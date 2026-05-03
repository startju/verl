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

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Snapshot key in AgentLoopOutput.extra_fields holding the AgentData state
# needed to resume a multi-turn rollout aborted mid-generation. Only fields not
# already recoverable from AgentLoopOutput's typed fields are stashed here.
_SNAPSHOT_KEY = "_prv3_tool_agent_snapshot"


@register("prv3_tool_agent")
class PRv3ToolAgentLoop(ToolAgentLoop):
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        last_agent_loop_output: AgentLoopOutput = kwargs.get("last_agent_loop_output")

        if last_agent_loop_output is None:
            assert kwargs.get("_prv3_is_validate") is True, (
                "last_agent_loop_output=None outside validate path "
                f"(kwargs.get('_prv3_is_validate')={kwargs.get('_prv3_is_validate')!r})"
            )
            return await super().run(sampling_params, **kwargs)

        if not last_agent_loop_output.prompt_ids:
            agent_data = await self._init_agent_data_fresh(**kwargs)
            initial_state = AgentState.PENDING
        else:
            if last_agent_loop_output.extra_fields["stop_reason"] != "aborted":
                return last_agent_loop_output
            agent_data = await self._restore_agent_data(last_agent_loop_output, **kwargs)
            # Prompt is already templated and conversation tokens are intact;
            # jump straight back into another generate call.
            initial_state = AgentState.GENERATING

        state = initial_state
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        output = self._build_output(agent_data)
        # Stash AgentData state when terminating on abort so the next round can
        # resume the exact conversation/turn-count/metrics state. Skipped for
        # clean terminations to keep extra_fields lean for the postprocessor.
        if output.extra_fields.get("stop_reason") == "aborted":
            output.extra_fields[_SNAPSHOT_KEY] = {
                "messages": agent_data.messages,
                "user_turns": agent_data.user_turns,
                "assistant_turns": agent_data.assistant_turns,
                "metrics": dict(agent_data.metrics),
                "tools_kwargs": agent_data.tools_kwargs,
                "request_id": agent_data.request_id,
            }
        return output

    async def _init_agent_data_fresh(self, **kwargs) -> AgentData:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics={},
            request_id=uuid4().hex,
            tools_kwargs=kwargs.get("tools_kwargs", {}),
        )
        self._apply_per_sample_tool_selection(agent_data, kwargs)
        return agent_data

    async def _restore_agent_data(self, last: AgentLoopOutput, **kwargs) -> AgentData:
        snapshot = last.extra_fields.get(_SNAPSHOT_KEY, {})
        multi_modal = last.multi_modal_data or {}
        images = multi_modal.get("images")
        videos = multi_modal.get("videos")
        agent_data = AgentData(
            messages=snapshot.get("messages") or list(kwargs["raw_prompt"]),
            image_data=images,
            video_data=videos,
            metrics=dict(snapshot.get("metrics", {})),
            # Reuse the prior round's request_id so vLLM can match preemption /
            # KV-cache state on resume.
            request_id=snapshot.get("request_id") or uuid4().hex,
            tools_kwargs=snapshot.get("tools_kwargs") or kwargs.get("tools_kwargs", {}),
        )
        # Token-level state lives on AgentLoopOutput typed fields, not snapshot:
        # rebuild the running conversation by concatenating prior prompt+response.
        agent_data.prompt_ids = list(last.prompt_ids) + list(last.response_ids)
        agent_data.response_mask = list(last.response_mask)
        agent_data.response_logprobs = list(last.response_logprobs or [])
        agent_data.routed_experts = last.routed_experts
        agent_data.user_turns = snapshot.get("user_turns", 0)
        # Compensate the upstream `assistant_turns += 1` that runs at the top of
        # `_handle_generating_state` (before the abort check, line ~229 of upstream
        # tool_agent_loop.py): when generation is cancelled mid-flight, the snapshot
        # already reflects the optimistic +1, and on resume the same line will fire
        # again for the *same* logical assistant turn. Subtract once to undo the
        # double-count. Floor at 0 in case snapshot was 0 (defensive).
        agent_data.assistant_turns = max(0, snapshot.get("assistant_turns", 0) - 1)
        # turn_scores / tool_rewards are written to extra_fields by upstream's
        # _build_output tail; pull them back so cumulative state survives.
        agent_data.turn_scores = list(last.extra_fields.get("turn_scores", []))
        agent_data.tool_rewards = list(last.extra_fields.get("tool_rewards", []))
        # Carry over user-defined extra_fields, dropping our internal snapshot
        # so it doesn't leak into the next output's payload.
        agent_data.extra_fields = {
            k: v for k, v in last.extra_fields.items() if k not in (_SNAPSHOT_KEY, "turn_scores", "tool_rewards")
        }
        self._apply_per_sample_tool_selection(agent_data, kwargs)
        return agent_data

    def _apply_per_sample_tool_selection(self, agent_data: AgentData, kwargs: dict[str, Any]) -> None:
        # Same routing as upstream ToolAgentLoop.run: filter tools by per-sample
        # extra_info.tool_selection, falling back to the full registered set.
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

    def _build_output(self, agent_data: AgentData) -> AgentLoopOutput:
        # Same tail as upstream ToolAgentLoop.run; lifted out so resume path can
        # reuse it without re-entering the state machine wrapper.
        response_ids = (
            agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        )
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update(
            {"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards}
        )
        return output
