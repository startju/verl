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
        self.inflight = 0

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        if self.paused:
            # Carry global_steps even on abort so downstream `_postprocess`
            # produces a consistent non_tensor_batch schema across prompts.
            # Without this, prompts whose first generation aborts permanently
            # lose `global_steps` (PRv3ToolAgentLoop's upstream
            # _handle_generating_state only takes extra_fields wholesale on
            # the first call), and DataProto.concat in pull_batch fails with
            # "key global_steps length N != batch_size".
            return TokenOutput(
                token_ids=[],
                stop_reason="aborted",
                extra_fields={"stop_reason": "aborted", "global_steps": self.global_steps},
            )
        self.inflight += 1
        token_output = await vLLMHttpServer.generate(
                    self, prompt_ids, sampling_params, request_id, image_data, video_data, priority
                )
        self.inflight -= 1
        if token_output.stop_reason == "abort":
            token_output.stop_reason = "aborted"
        token_output.extra_fields["stop_reason"] = token_output.stop_reason
        return token_output

    async def cancel(self):
        import os
        import time
        wpid = os.getpid()
        t0 = time.perf_counter()
        self.paused = True
        n_inflight_start = self.inflight
        drain_iters = 0
        while self.inflight:
            await self.abort_all_requests(reset_prefix_cache=False)
            await asyncio.sleep(0)
            drain_iters += 1
        t1 = time.perf_counter()
        print(
            f"[CANCEL-DBG pid={wpid}] server.cancel "
            f"total={t1 - t0:.3f}s n_inflight_start={n_inflight_start} drain_iters={drain_iters}",
            flush=True,
        )

    async def resume(self):
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
        import time
        t0 = time.perf_counter()
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])
        t1 = time.perf_counter()
        print(
            f"[CANCEL-DBG] replica.cancel ray_total={t1 - t0:.3f}s n_servers={len(self.servers)}",
            flush=True,
        )

    async def resume(self):
        """Resume each rollout server."""
        await asyncio.gather(*[server.resume.remote() for server in self.servers])
