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

import itertools
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.fully_async_policy.detach_utils import RolloutSample, assemble_batch_from_rollout_samples
from verl.protocol import DataProto

# Cap total in-flight prompts (pending + ongoing + done) at ~2x batch_size.
# Originally sized to keep workers fed during long-running prompts; now also
# acts as a hard ceiling on done_queue growth — see push_batch's backpressure
# block for the trade-off rationale.
_PREFETCH_FACTOR = 2


@dataclass
class RolloutPrompt:
    """Enhanced rollout prompt (with n rollout samples) carrying generation state across partial rollouts."""

    # Original (un-repeated) trainer batch. Repeated to n rows and unioned
    # with gen_batch_output in pull_batch to form the final training batch
    # (uid / agent_name / raw prompt fields live here; generation tensors
    # live in gen_batch_output).
    batch: DataProto

    # Mutates across the prompt's lifetime:
    #   - push_batch  : raw gen-side fields (prompt tokens etc.), n rows.
    #   - worker side : replaced by AgentLoopWorker._postprocess(...) once the
    #                   rollout completes — schema becomes the standard
    #                   postprocessed one (prompts / responses / response_mask /
    #                   attention_mask / position_ids + meta_info["metrics"] +
    #                   fully_async fields like min_global_steps).
    #   - pull_batch  : unioned with rp.batch.repeat(n) to form the training batch.
    gen_batch_output: DataProto
    prompt_id: str

    # AgentLoopOutput from generation. Length is n while the prompt is being
    # rolled out; once all n samples finish successfully the worker postprocesses
    # them into gen_batch_output and clears this list to []. So:
    #   - len == n  : in-flight or freshly pushed (sentinels)
    #   - len == 0  : terminal, already postprocessed
    # Both are valid states. is_prompt_done relies on this: an empty list
    # vacuously returns True, which matches "fully done" semantics.
    agent_loop_output_list: list[AgentLoopOutput]


def is_prompt_done(prompt: RolloutPrompt) -> bool:
    # A prompt is "done" iff none of its n samples were aborted mid-flight.
    # Any other stop_reason (including missing / None / unknown values) is
    # treated as a successful terminal state by design — only the explicit
    # "aborted" sentinel from PRv3vLLMHttpServer triggers a requeue.
    #
    # Empty agent_loop_output_list also returns True (vacuous `not any([])`):
    # this is intentional — the worker clears the list after postprocessing,
    # so an empty list means "fully done, already postprocessed".
    return not any(
        output.extra_fields.get("stop_reason") == "aborted" for output in prompt.agent_loop_output_list
    )


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
        # tokenizer is passed straight through to assemble_batch_from_rollout_samples
        # in pull_batch; this class never reads it. Kept on self only because the
        # downstream call needs it. If assemble_batch_from_rollout_samples ever
        # stops needing a tokenizer, this field can be dropped entirely.
        self.tokenizer = tokenizer
        self.batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
        self.n = self.config.actor_rollout_ref.rollout.n
        # Fail-fast: invalid batch_size makes pull_batch silently never return,
        # and n < 1 would make push_batch create RolloutPrompts with an empty
        # sentinel list — which is_prompt_done treats as "fully done" (the
        # postprocessed-terminal state), so a fresh prompt would skip rollout
        # entirely and land in done_queue with no generation.
        assert self.batch_size > 0, f"batch_size must be > 0, got {self.batch_size}"
        assert self.n >= 1, f"rollout.n must be >= 1, got {self.n}"
        self.ongoing_set: set[str] = set()
        self.pending_queue: deque[RolloutPrompt] = deque()
        self.done_queue: deque[RolloutPrompt] = deque()

    def push_batch(self, batch: DataProto, gen_batch: DataProto) -> bool:
        """Enqueue a trainer-supplied batch into pending_queue.

        Returns True iff the manager wants more prompts — i.e. total in-flight
        (pending + ongoing + done) is still below the prefetch buffer. The
        trainer uses this as a backpressure signal to decide whether to keep
        feeding.
        """
        num_prompts = batch.batch.size(0)
        # gen_batch is split row-wise in lockstep with batch; mismatched row
        # counts would silently mis-pair prompts with their generation inputs.
        assert gen_batch.batch.size(0) == num_prompts, (
            f"gen_batch row count {gen_batch.batch.size(0)} != batch row count {num_prompts}"
        )
        batch_list = batch.chunk(num_prompts)
        gen_batch_list = gen_batch.chunk(num_prompts)
        n = self.n

        # Validate all UIDs first, then mutate. This makes push_batch atomic:
        # if any UID collides (with another in-flight prompt or a duplicate
        # within this same batch) the assert fires before any state changes,
        # so the actor is left in a clean state for the caller to react to.
        in_flight_ids = (
            self.ongoing_set | {p.prompt_id for p in self.pending_queue} | {p.prompt_id for p in self.done_queue}
        )
        # Element type is whatever non_tensor_batch["uid"] holds — usually
        # numpy.str_ (a str subclass), occasionally a python str depending on
        # how the upstream dataset built the column. Annotated as Any to avoid
        # implying a stricter contract than what we actually rely on (just
        # hashability + equality for ongoing/pending/done lookups).
        prompt_ids: list[Any] = []
        for sample_batch in batch_list:
            prompt_id = sample_batch.non_tensor_batch["uid"][0]
            assert prompt_id not in in_flight_ids, (
                f"push_batch received prompt_id {prompt_id!r} already in flight "
                "(ongoing/pending/done); UID collisions across calls are not supported"
            )
            in_flight_ids.add(prompt_id)
            prompt_ids.append(prompt_id)

        for i in range(num_prompts):
            prompt_batch = batch_list[i]
            prompt_gen_batch = gen_batch_list[i]
            self.pending_queue.append(
                RolloutPrompt(
                    batch=prompt_batch,
                    gen_batch_output=prompt_gen_batch.repeat(repeat_times=n, interleave=True),
                    prompt_id=prompt_ids[i],
                    # Sentinel AgentLoopOutputs: only `extra_fields["stop_reason"]`
                    # is read by the first _run_agent_loop call (to detect that
                    # this is a fresh prompt that needs full rollout). The real
                    # AgentLoopOutput overwrites this whole list when the worker
                    # finishes — so we use model_construct to skip pydantic's
                    # required-field validation rather than fabricate dummy
                    # prompt_ids/response_ids/metrics.
                    # NOTE: model_construct is a pydantic-internal API; if
                    # AgentLoopOutput migrates off pydantic this needs revisiting.
                    agent_loop_output_list=[
                        AgentLoopOutput.model_construct(extra_fields={"stop_reason": "aborted"}) for _ in range(n)
                    ],
                )
            )
        # Backpressure: count total in-flight (pending + ongoing + done), not
        # just pending. Otherwise a fast worker / slow trainer pattern keeps
        # pending empty while done_queue grows unbounded.
        #
        # This is a deliberate trade-off: bounding total in-flight prioritizes
        # memory safety over keeping every worker saturated. In steady state
        # workers may briefly idle when done_queue is near the cap — that is
        # expected, not a starvation bug. Don't "fix" it by reverting to a
        # pending-only check without addressing the OOM risk.
        in_flight = len(self.pending_queue) + len(self.ongoing_set) + len(self.done_queue)
        return in_flight < _PREFETCH_FACTOR * self.batch_size

    def pull_batch(self) -> DataProto | None:
        """Assemble one training batch from done_queue, or return None if not enough done yet."""
        if len(self.done_queue) < self.batch_size:
            return None
        # Peek-then-commit: assemble can OOM during repeat/union/cat. If we
        # popleft up front and then fail, the prompts are lost; the caller
        # has no way to retry. Instead build the result first, only mutate
        # done_queue once assemble has returned successfully.
        rollout_prompts = list(itertools.islice(self.done_queue, self.batch_size))
        rollout_samples = []
        for rp in rollout_prompts:
            full_batch = rp.batch.repeat(repeat_times=self.n, interleave=True).union(rp.gen_batch_output)
            # epoch=0 is a placeholder: RolloutSample.epoch is required by the
            # dataclass but not read by assemble_batch_from_rollout_samples,
            # and partial rollout doesn't propagate epoch through this queue.
            rollout_samples.append(
                RolloutSample(full_batch=full_batch, sample_id=rp.prompt_id, epoch=0, rollout_status={})
            )

        result = assemble_batch_from_rollout_samples(
            rollout_samples,
            self.tokenizer,
            self.config,
        )
        for _ in range(self.batch_size):
            self.done_queue.popleft()
        return result

    def pull_prompts(self, max_count: int) -> list[RolloutPrompt]:
        """Hand at most max_count prompts to a worker; moves them pending → ongoing."""
        take = min(max_count, len(self.pending_queue))
        # Validate first, then mutate — same atomicity discipline as push_batch
        # / push_prompts. Belt-and-braces: by construction (push_batch UID
        # guard + push_prompts removing from ongoing_set before requeue) this
        # assert should never fire; kept to catch internal bugs early without
        # leaving the actor with prompts popped but not tracked.
        # Use islice instead of indexed access — deque[i] is O(i), so an
        # indexed loop here would be O(take^2).
        for prompt in itertools.islice(self.pending_queue, take):
            assert prompt.prompt_id not in self.ongoing_set, (
                f"prompt {prompt.prompt_id} already in ongoing_set"
            )

        pending_prompts = [self.pending_queue.popleft() for _ in range(take)]
        self.ongoing_set.update(p.prompt_id for p in pending_prompts)
        return pending_prompts

    def push_prompts(self, prompts: list[RolloutPrompt]) -> None:
        """Return prompts from a worker; routes done ones to done_queue, aborted ones back to pending."""
        # Validate first, then mutate — same atomicity discipline as push_batch.
        # Catches both "prompt was never pulled" and "same prompt pushed twice
        # in this call" before any state changes. is_prompt_done is computed
        # here too so the mutate phase below is pure dict/deque ops with no
        # logic that could raise mid-loop.
        seen: set[str] = set()
        done_flags: list[bool] = []
        for prompt in prompts:
            assert prompt.prompt_id in self.ongoing_set, (
                f"push_prompts: {prompt.prompt_id!r} was never pulled (not in ongoing_set)"
            )
            assert prompt.prompt_id not in seen, f"push_prompts: {prompt.prompt_id!r} appears twice in the same call"
            seen.add(prompt.prompt_id)
            done_flags.append(is_prompt_done(prompt))

        for prompt, is_done in zip(prompts, done_flags, strict=True):
            # remove (not discard): assert above guarantees membership, so a
            # KeyError here would mean an internal bug we want to surface.
            self.ongoing_set.remove(prompt.prompt_id)
            if is_done:
                self.done_queue.append(prompt)
            else:
                # appendleft (LIFO): an aborted prompt goes to the head so the
                # next pull_prompts resumes it while its KV cache may still be
                # live on the rollout server. Don't change to append() without
                # benchmarking — the LIFO bias is a deliberate cache-locality
                # optimization, not a fairness bug.
                #
                # Side effect for multi-prompt pushes: appendleft applied per
                # iteration reverses intra-batch order in pending_queue. Today
                # callers always push a single prompt (agent_loop.py wraps each
                # rollout result in [prompt]) so it's invisible. If batch sizes
                # grow, switch to extendleft(reversed(aborted_subset)) to
                # preserve order — or decide that the reversal is fine.
                self.pending_queue.appendleft(prompt)
