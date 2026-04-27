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
from typing import Any, Optional

import ray
from ray.actor import ActorHandle

from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.replica import RolloutMode, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


@ray.remote
class PRv3vLLMHttpServer(vLLMHttpServer):
    def __init__(
        self,
        config,
        model_config,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        super().__init__(
            config,
            model_config,
            rollout_mode,
            workers,
            replica_rank,
            node_rank,
            gpus_per_node,
            nnodes,
            cuda_visible_devices,
        )

        # for cancel LLMServer
        self.paused = False
        self.lock = asyncio.Lock()
        self.cancel_event_dict: dict[str, asyncio.Event] = {}
        self.token_output_dict: dict[str, Optional[TokenOutput]] = {}

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        async with self.lock:
            if self.paused:
                return TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"stop_reason": "aborted"})
            self.token_output_dict[request_id] = None
            self.cancel_event_dict[request_id] = asyncio.Event()

            async def _generate():
                self.token_output_dict[request_id] = await vLLMHttpServer.generate(
                    self, prompt_ids, sampling_params, request_id, image_data, video_data, priority
                )

            generate_handle = asyncio.create_task(_generate())
            cancel_handle = asyncio.create_task(self.cancel_event_dict[request_id].wait())

        try:
            done, pend = await asyncio.wait([generate_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await task
            for task in pend:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        finally:
            async with self.lock:
                token_output = self.token_output_dict.pop(request_id, None)
                self.cancel_event_dict.pop(request_id, None)

        if token_output is None:
            await self.abort_request(request_id, True)
            return TokenOutput(token_ids=[], stop_reason="aborted", extra_fields={"stop_reason": "aborted"})
        if token_output.stop_reason == "abort":
            token_output.stop_reason = "aborted"
        token_output.extra_fields["stop_reason"] = token_output.stop_reason
        return token_output

    async def cancel(self):
        async with self.lock:
            self.paused = True
            for cancel_event in self.cancel_event_dict.values():
                cancel_event.set()

    async def resume(self):
        async with self.lock:
            self.paused = False


class PRv3vLLMReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        name_suffix: str = "",
    ):
        super().__init__(
            replica_rank, config, model_config, gpus_per_node, is_reward_model, is_teacher_model, name_suffix
        )
        self.server_class = PRv3vLLMHttpServer

    async def cancel(self):
        """Cancel each rollout server."""
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
