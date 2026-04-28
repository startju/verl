# Copyright 2025 Meituan Ltd. and/or its affiliates
#
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

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.partial_rollout.prompt_manager import RolloutPrompt, RolloutPromptManager, is_prompt_done
from verl.experimental.partial_rollout.vllm_rollout.vllm_async_server import PRv3vLLMReplica
from verl.experimental.teacher_loop.teacher_model import MultiTeacherModelManager
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


@ray.remote
class PRv3AgentLoopWorker(AgentLoopWorker):
    def __init__(
        self,
        config: DictConfig,
        servers: list[tuple[str, ray.actor.ActorHandle]],
        load_balancer_handle: ray.actor.ActorHandle,
        prompt_manager_handle: ray.actor.ActorHandle,
        teacher_servers: Optional[dict[str, list[tuple[str, ray.actor.ActorHandle]]]] = None,
        teacher_load_balancer_handle: Optional[dict[str, ray.actor.ActorHandle]] = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        super().__init__(
            config,
            servers,
            load_balancer_handle,
            teacher_servers,
            teacher_load_balancer_handle,
            reward_loop_worker_handles,
        )
        self.prompt_manager_handle = prompt_manager_handle

    async def run_generate_sequences(self, max_inflight_prompts: int, global_steps: int):
        rollout_prompts: list[RolloutPrompt] = await self.prompt_manager_handle.pull_prompts.remote(
            max_inflight_prompts
        )

        running_set: set[asyncio.Task] = {
            asyncio.create_task(self._generate_sequences_for_prompt(rp, global_steps)) for rp in rollout_prompts
        }

        is_canceled = False
        while running_set:
            done, _ = await asyncio.wait(running_set, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                running_set.remove(task)
                rollout_prompt = task.result()
                self.prompt_manager_handle.push_prompts.remote([rollout_prompt])
                if not is_prompt_done(rollout_prompt):
                    is_canceled = True
            if is_canceled:
                continue
            new_rollout_prompts: list[RolloutPrompt] = await self.prompt_manager_handle.pull_prompts.remote(len(done))
            running_set.update(
                asyncio.create_task(self._generate_sequences_for_prompt(rp, global_steps)) for rp in new_rollout_prompts
            )

    # copy from AgentLoopWorker generate_sequences
    async def _generate_sequences_for_prompt(self, rollout_prompt: RolloutPrompt, global_steps: int) -> RolloutPrompt:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        batch = rollout_prompt.gen_batch_output
        agent_loop_output_list = rollout_prompt.agent_loop_output_list
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        # Track the weight-version range used to generate each sample across the
        # potentially many aborted-and-resumed worker rounds this prompt goes through.
        # min defaults to the current global_steps so a fresh sample records "started here";
        # on resume it carries over the earliest round's value from prior extra_fields.
        # max stays None until a round finishes non-aborted — aborted rounds intentionally
        # leave max unset so a later resume round can fill it in.
        min_global_steps: list[int] = [global_steps] * len(batch)
        max_global_steps: list[Optional[int]] = [None] * len(batch)
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            if "min_global_steps" in agent_loop_output_list[i].extra_fields:
                min_global_steps[i] = agent_loop_output_list[i].extra_fields["min_global_steps"]
            if "max_global_steps" in agent_loop_output_list[i].extra_fields:
                max_global_steps[i] = agent_loop_output_list[i].extra_fields["max_global_steps"]
            kwargs["last_agent_loop_output"] = agent_loop_output_list[i]
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop_no_post(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        rollout_prompt.agent_loop_output_list = await asyncio.gather(*tasks)
        for i in range(len(batch)):
            rollout_prompt.agent_loop_output_list[i].extra_fields["min_global_steps"] = min_global_steps[i]
            if max_global_steps[i] is not None:
                rollout_prompt.agent_loop_output_list[i].extra_fields["max_global_steps"] = max_global_steps[i]
            elif rollout_prompt.agent_loop_output_list[i].stop_reason != "aborted":
                rollout_prompt.agent_loop_output_list[i].extra_fields["max_global_steps"] = global_steps

        if is_prompt_done(rollout_prompt):
            coros = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                coros.append(
                    self._agent_loop_postprocess(
                        rollout_prompt.agent_loop_output_list[i], trajectory_info[i]["validate"], **kwargs
                    )
                )
            internal_agent_loop_output_list = await asyncio.gather(*coros)
            rollout_prompt.gen_batch_output = self._postprocess(
                internal_agent_loop_output_list,
                input_non_tensor_batch=batch.non_tensor_batch,
                validate=batch.meta_info.get("validate", False),
            )
            rollout_prompt.agent_loop_output_list = []
        return rollout_prompt

    # copy from AgentLoopWorker._run_agent_loop without call _agent_loop_postprocess
    async def _run_agent_loop_no_post(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> AgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )
            assert agent_name.startswith("prv3_"), (
                f"partial rollout requires a PRv3-aware agent loop (consumes `last_agent_loop_output` to resume "
                f"after abort), got {agent_name!r}; otherwise resume silently degrades to full rollout"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
            )
            return await agent_loop.run(sampling_params, **kwargs)


class PRv3AgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        teacher_model_manager: MultiTeacherModelManager = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.rollout_replica_class = PRv3vLLMReplica
        self.agent_loop_workers_class = PRv3AgentLoopWorker
        super().__init__(config, worker_group, rollout_resource_pool, teacher_model_manager, reward_loop_worker_handles)

    # copy from AgentLoopManager.create, not call _init_agent_loop_workers
    # we need call init_agent_loop_workers manually
    @classmethod
    @auto_await
    async def create(
        cls,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        teacher_model_manager: MultiTeacherModelManager = None,
    ):
        """Create agent loop manager."""
        instance = cls(config, worker_group, rollout_resource_pool, teacher_model_manager, reward_loop_worker_handles)
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        # await instance._init_agent_loop_workers()
        return instance

    @auto_await
    async def init_agent_loop_workers(self, rollout_prompt_manager: RolloutPromptManager):
        self.rollout_prompt_manager = rollout_prompt_manager
        await self._init_agent_loop_workers()

    # copy from AgentLoopManager._init_agent_loop_workers, add rollout_prompt_manager params to build agent_loop_worker
    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        load_balancer_handle = self.global_load_balancer
        servers = list(zip(self.server_addresses, self.server_handles, strict=True))

        if self.distillation_enabled:
            # teacher_model_manager exposes per-teacher dicts keyed by teacher key.
            teacher_servers = {
                key: list(
                    zip(
                        self.teacher_model_manager.server_addresses[key],
                        self.teacher_model_manager.server_handles[key],
                        strict=True,
                    )
                )
                for key in self.teacher_model_manager.server_addresses
            }
            teacher_load_balancer_handle = dict(self.teacher_model_manager.load_balancer_handle)
        else:
            teacher_servers = None
            teacher_load_balancer_handle = None

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    servers,
                    load_balancer_handle,
                    self.rollout_prompt_manager,
                    teacher_servers,
                    teacher_load_balancer_handle,
                    self.reward_loop_worker_handles,
                )
            )

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        await self.resume()
        if prompts.meta_info.get("validate", False):
            return await super().generate_sequences(prompts)

        assert "global_steps" in prompts.meta_info, (
            "PRv3 generate_sequences requires meta_info['global_steps'] to track per-sample weight-version range"
        )
        global_steps = prompts.meta_info["global_steps"]

        num_rollout_prompts = prompts.batch.size(0) // self.config.actor_rollout_ref.rollout.n

        max_inflight_prompts = (num_rollout_prompts + len(self.agent_loop_workers) - 1) // len(self.agent_loop_workers)
        worker_tasks = [
            worker.run_generate_sequences.remote(max_inflight_prompts, global_steps)
            for worker in self.agent_loop_workers
        ]

        while True:
            output = await self.rollout_prompt_manager.pull_batch.remote()
            if output:
                break
            await asyncio.sleep(0.01)
        await self.cancel()
        await asyncio.gather(*worker_tasks)

        # calculate performance metrics
        outputs = [output]
        metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])
