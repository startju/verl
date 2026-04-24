# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from pprint import pprint
from typing import Optional

import ray
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl.experimental.partial_rollout.agent_loop.agent_loop import PRv3AgentLoopManager
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class PRv3RayPPOTrainer(SeparateRayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

    def init_workers(self):
        self.config.actor_rollout_ref.rollout["agent"]["agent_loop_manager_class"] = (
            f"{PRv3AgentLoopManager.__module__}.{PRv3AgentLoopManager.__qualname__}"
        )
        from verl.experimental.partial_rollout.prompt_manager import RolloutPromptManager

        self.rollout_prompt_manager = RolloutPromptManager.remote()
        RayPPOTrainer.init_workers(self)
        self.async_rollout_manager.init_agent_loop_workers(self.rollout_prompt_manager)

    def _fit_generate(self, data_loader_iter) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw

        need_more = False
        while need_more:
            try:
                batch_dict = next(data_loader_iter)
            except StopIteration:
                # make n length dummy
                gen_batch = DataProto(
                    batch=TensorDict({}, batch_size=(self.train_dataloader.batch_size,)), meta_info={}
                )
                break
            batch = self._fit_get_batch(batch_dict)
            gen_batch = self._get_gen_batch(batch)
            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps
            need_more = ray.get(self.rollout_prompt_manager.push_pending_prompts.remote(batch, gen_batch))

        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            if self.curr_step_profile:
                self.async_rollout_manager.start_profile(global_step=self.global_steps)
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
            self.checkpoint_manager.sleep_replicas()
            if self.curr_step_profile:
                self.async_rollout_manager.stop_profile()

            timing_raw.update(gen_batch_output.meta_info["timing"])
            gen_batch_output.meta_info.pop("timing", None)

        # TODO: support ReMax

        # we don't need these, just use gen_batch_output
        """
        # repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)
        """

        batch = gen_batch_output

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        # get images_seqlens
        images_seqlens_all = []
        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
            if "image_grid_thw" not in multi_modal_input.keys():
                continue
            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all
        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.

        !!!
        The logic of fit is consistent with that of fit_refactor;
        if any modifications are made, apply them to both methods simultaneously.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                self.epoch = epoch
                self.fit_step(batch_dict)
                if self.is_last_step:
                    return

    def fit_step(self, data_loader_iter):
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = self._fit_generate(data_loader_iter)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_weights()
            self._fit_dump_data(batch)

        self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_experimental(batch)
        self._fit_postprocess_step()
