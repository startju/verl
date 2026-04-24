from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler.performance import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput


@register("prv3_single_turn_agent")
class PRv3SingleTurnAgentLoop(SingleTurnAgentLoop):
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        last_agent_loop_output: AgentLoopOutput = kwargs.get("last_agent_loop_output")

        metrics = {}
        if not last_agent_loop_output.prompt_ids:
            messages = list(kwargs["raw_prompt"])

            # 1. extract images and videos from messages
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")

            # 2. apply chat template and tokenize
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
            )
        else:
            if last_agent_loop_output.extra_fields.get("stop_reason") not in ("abort", "aborted"):
                return last_agent_loop_output
            prompt_ids = last_agent_loop_output.prompt_ids + last_agent_loop_output.response_ids
            metrics["generate_sequences"] = last_agent_loop_output.metrics.generate_sequences

        # 3. generate sequences
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

        response_mask = [1] * len(output.token_ids)
        if not last_agent_loop_output.prompt_ids:
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            output: AgentLoopOutput = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=output.token_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
                routed_experts=(
                    output.routed_experts[: len(prompt_ids) + self.response_length]
                    if output.routed_experts is not None
                    else None
                ),
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=metrics,
                extra_fields=output.extra_fields,
            )
        else:
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
            elif output.num_preempted is not None:
                if metrics["num_preempted"] == -1:
                    metrics["num_preempted"] = output.num_preempted
                else:
                    metrics["num_preempted"] += output.num_preempted
            else:
                metrics["num_preempted"] = -1

            prompt_ids = last_agent_loop_output.prompt_ids
            response_ids = last_agent_loop_output.response_ids + output.token_ids
            response_mask = last_agent_loop_output.response_mask + response_mask
            response_logprobs = last_agent_loop_output.response_logprobs
            if response_logprobs is not None or output.response_logprobs is not None:
                response_logprobs = (response_logprobs or []) + (output.response_logprobs or [])

            output: AgentLoopOutput = AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids[: self.response_length],
                response_mask=response_mask[: self.response_length],
                response_logprobs=response_logprobs[: self.response_length] if output.log_probs else None,
                routed_experts=(
                    output.routed_experts[: len(prompt_ids) + self.response_length]
                    if output.routed_experts is not None
                    else None
                ),
                multi_modal_data=last_agent_loop_output.multi_modal_data,
                num_turns=2,
                metrics=metrics,
                extra_fields=output.extra_fields,
            )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
