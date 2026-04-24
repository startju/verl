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

from collections import deque
from dataclasses import dataclass

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.fully_async_policy.detach_utils import RolloutSample, assemble_batch_from_rollout_samples
from verl.protocol import DataProto


@dataclass
class RolloutPrompt:
    """Enhanced rollout prompt (with n rollout samples) containing both original batch info and AgentLoopOutput"""

    batch: DataProto
    gen_batch_output: DataProto
    prompt_id: str

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]  # length: n


@ray.remote
class RolloutPromptManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        self.ongoing_set = set[str]()
        self.pending_queue = deque[RolloutPrompt]()
        self.done_queue = deque[RolloutPrompt]()

    def push_batch(self, batch: DataProto, gen_batch: DataProto):
        num_prompts = batch.batch.size(0)
        batch_list = batch.chunk(num_prompts)
        gen_batch_list = gen_batch.chunk(num_prompts)
        for i in range(num_prompts):
            batch = batch_list[i]
            gen_batch = gen_batch_list[i]
            gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            self.pending_queue.append(
                RolloutPrompt(
                    batch=batch,
                    gen_batch_output=gen_batch_output,
                    prompt_id=batch.non_tensor_batch["uid"][0],
                    agent_loop_output_list=[AgentLoopOutput(extra_fields={"stop_reason":"aborted"})]
                    * self.config.actor_rollout_ref.rollout.n,
                )
            )
        return len(self.pending_queue) < 2 * self.batch_size

    def pull_batch(self) -> DataProto:
        if len(self.done_queue) < self.batch_size:
            return DataProto()
        rollout_prompts = [self.done_queue.popleft() for _ in range(self.batch_size)]

        return assemble_batch_from_rollout_samples(
            [RolloutSample(full_batch=rp.gen_batch_output, sample_id=rp.prompt_id) for rp in rollout_prompts]
        )

    def pull_prompts(self, num_rollout_prompts: int) -> list[RolloutPrompt]:
        num_rollout_prompts = min(num_rollout_prompts, len(self.pending_queue))
        pending_prompts = [self.pending_queue.popleft() for _ in range(num_rollout_prompts)]
        for prompt in pending_prompts:
            assert prompt.prompt_id not in self.ongoing_set, f"prompt {prompt.prompt_id} already in ongoing_set"
            self.ongoing_set.add(prompt.prompt_id)
        return pending_prompts

    def push_prompts(self, prompts: list[RolloutPrompt]):
        for prompt in prompts:
            if is_prompt_done(prompt):
                self.done_queue.append(prompt)
            else:
                self.pending_queue.appendleft(prompt)


def is_prompt_done(prompt: RolloutPrompt) -> bool:
    for agent_loop_output in prompt.agent_loop_output_list:
        if agent_loop_output.extra_fields["stop_reason"] in ("abort", "aborted"):
            return False
    return True
