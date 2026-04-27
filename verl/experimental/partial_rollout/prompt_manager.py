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
from typing import Any

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.fully_async_policy.detach_utils import RolloutSample, assemble_batch_from_rollout_samples
from verl.protocol import DataProto

# Keep ~2x batch_size prompts buffered in pending_queue so rollout workers
# don't starve while the trainer is busy with the previous step.
_PREFETCH_FACTOR = 2


@dataclass
class RolloutPrompt:
    """Enhanced rollout prompt (with n rollout samples) carrying generation state across partial rollouts."""

    # Original (un-repeated) trainer batch. Reserved for downstream
    # reward / advantage / metric post-processing — not consumed inside
    # this manager. Don't drop without confirming the consumer.
    batch: DataProto
    gen_batch_output: DataProto
    prompt_id: str

    # AgentLoopOutput from generation
    agent_loop_output_list: list[AgentLoopOutput]  # length: n


def is_prompt_done(prompt: RolloutPrompt) -> bool:
    # A prompt is "done" iff none of its n samples were aborted mid-flight.
    # Any other stop_reason (including missing / None / unknown values) is
    # treated as a successful terminal state by design — only the explicit
    # "aborted" sentinel from PRv3vLLMHttpServer triggers a requeue.
    for agent_loop_output in prompt.agent_loop_output_list:
        if agent_loop_output.extra_fields.get("stop_reason") == "aborted":
            return False
    return True


@ray.remote
class RolloutPromptManager:
    """Ray actor coordinating partial-rollout prompt scheduling.

    Three data structures track each prompt's state:

      - pending_queue : prompts waiting for a worker. FIFO from the back
                        (push_batch.append) and LIFO from the front
                        (push_prompts.appendleft for aborted requeues, see
                        push_prompts).
      - ongoing_set   : prompt_ids currently being rolled out by some worker.
      - done_queue    : prompts whose n samples all reached a non-aborted
                        stop_reason and are ready to be assembled.

    Transitions:

      trainer → push_batch      : (new)         → pending_queue
      worker  → pull_prompts    : pending_queue → ongoing_set
      worker  → push_prompts    : ongoing_set   → done_queue       (if all done)
                                : ongoing_set   → pending_queue.left (if any aborted)
      trainer → pull_batch      : done_queue    → assembled DataProto

    Ray actors are single-threaded, so all method bodies execute serially —
    no locking is needed inside this class.
    """

    def __init__(self, config: DictConfig, tokenizer: Any):
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        self.ongoing_set: set[str] = set()
        self.pending_queue: deque[RolloutPrompt] = deque()
        self.done_queue: deque[RolloutPrompt] = deque()

    def push_batch(self, batch: DataProto, gen_batch: DataProto) -> bool:
        """Enqueue a trainer-supplied batch into pending_queue.

        Returns True iff the manager wants more prompts — i.e. pending_queue
        is still below the prefetch buffer. The trainer uses this as a
        backpressure signal to decide whether to keep feeding.
        """
        num_prompts = batch.batch.size(0)
        batch_list = batch.chunk(num_prompts)
        gen_batch_list = gen_batch.chunk(num_prompts)
        n = self.config.actor_rollout_ref.rollout.n

        # Validate all UIDs first, then mutate. This makes push_batch atomic:
        # if any UID collides (with another in-flight prompt or a duplicate
        # within this same batch) the assert fires before any state changes,
        # so the actor is left in a clean state for the caller to react to.
        in_flight_ids = (
            self.ongoing_set
            | {p.prompt_id for p in self.pending_queue}
            | {p.prompt_id for p in self.done_queue}
        )
        for sample_batch in batch_list:
            prompt_id = sample_batch.non_tensor_batch["uid"][0]
            assert prompt_id not in in_flight_ids, (
                f"push_batch received prompt_id {prompt_id!r} already in flight "
                "(ongoing/pending/done); UID collisions across calls are not supported"
            )
            in_flight_ids.add(prompt_id)

        for i in range(num_prompts):
            sample_batch = batch_list[i]
            sample_gen_batch = gen_batch_list[i]
            gen_batch_output = sample_gen_batch.repeat(repeat_times=n, interleave=True)
            self.pending_queue.append(
                RolloutPrompt(
                    batch=sample_batch,
                    gen_batch_output=gen_batch_output,
                    prompt_id=sample_batch.non_tensor_batch["uid"][0],
                    # Sentinel AgentLoopOutputs: only `extra_fields["stop_reason"]`
                    # is read by the first _run_agent_loop call (to detect that
                    # this is a fresh prompt that needs full rollout). The real
                    # AgentLoopOutput overwrites this whole list when the worker
                    # finishes — so we use model_construct to skip pydantic's
                    # required-field validation rather than fabricate dummy
                    # prompt_ids/response_ids/metrics.
                    agent_loop_output_list=[
                        AgentLoopOutput.model_construct(extra_fields={"stop_reason": "aborted"})
                        for _ in range(n)
                    ],
                )
            )
        return len(self.pending_queue) < _PREFETCH_FACTOR * self.batch_size

    def pull_batch(self) -> DataProto | None:
        """Assemble one training batch from done_queue, or return None if not enough done yet."""
        if len(self.done_queue) < self.batch_size:
            return None
        rollout_prompts = [self.done_queue.popleft() for _ in range(self.batch_size)]
        # Note: prompt_ids are already removed from ongoing_set by push_prompts
        # before they enter done_queue, so no cleanup is needed here.

        return assemble_batch_from_rollout_samples(
            [
                # epoch=0 is a placeholder: RolloutSample.epoch is required by
                # the dataclass but not read by assemble_batch_from_rollout_samples,
                # and partial rollout doesn't propagate epoch through this queue.
                RolloutSample(full_batch=rp.gen_batch_output, sample_id=rp.prompt_id, epoch=0, rollout_status={})
                for rp in rollout_prompts
            ],
            self.tokenizer,
            self.config,
        )

    def pull_prompts(self, max_count: int) -> list[RolloutPrompt]:
        """Hand at most max_count prompts to a worker; moves them pending → ongoing."""
        max_count = min(max_count, len(self.pending_queue))
        pending_prompts = [self.pending_queue.popleft() for _ in range(max_count)]
        for prompt in pending_prompts:
            # Belt-and-braces invariant check: by construction (push_batch UID
            # guard + push_prompts removing from ongoing_set before requeue)
            # this should never fire. Kept to catch internal bugs early.
            assert prompt.prompt_id not in self.ongoing_set, f"prompt {prompt.prompt_id} already in ongoing_set"
            self.ongoing_set.add(prompt.prompt_id)
        return pending_prompts

    def push_prompts(self, prompts: list[RolloutPrompt]) -> None:
        """Return prompts from a worker; routes done ones to done_queue, aborted ones back to pending."""
        # Validate first, then mutate — same atomicity discipline as push_batch.
        # Catches both "prompt was never pulled" and "same prompt pushed twice
        # in this call" before any state changes.
        seen: set[str] = set()
        for prompt in prompts:
            assert prompt.prompt_id in self.ongoing_set, (
                f"push_prompts: {prompt.prompt_id!r} was never pulled (not in ongoing_set)"
            )
            assert prompt.prompt_id not in seen, (
                f"push_prompts: {prompt.prompt_id!r} appears twice in the same call"
            )
            seen.add(prompt.prompt_id)

        for prompt in prompts:
            self.ongoing_set.discard(prompt.prompt_id)
            if is_prompt_done(prompt):
                self.done_queue.append(prompt)
            else:
                # appendleft (LIFO): an aborted prompt goes to the head so the
                # next pull_prompts resumes it while its KV cache may still be
                # live on the rollout server. Don't change to append() without
                # benchmarking — the LIFO bias is a deliberate cache-locality
                # optimization, not a fairness bug.
                self.pending_queue.appendleft(prompt)
