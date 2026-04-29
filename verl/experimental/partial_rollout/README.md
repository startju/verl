# Recipe: Partial Rollout (PRv3)

English | [简体中文](README_zh.md)

**Authors:** Yue Wang, Zhipeng Ma, Yi Yan, Hang Xu, Yang Li, Bo Qian, Peng Chen, Xingyu Zhao

A partial-rollout pipeline for **synchronous RL training**, designed to reclaim GPU bubbles caused by **long-tail response lengths** via sample supplementation and mid-generation interruption with cross-step resume.

> ⚠️ **Don't confuse this with the fully-async framework's partial rollout.** This pipeline still runs the synchronous loop "rollout → wait for batch → one train step"; the only twist is that long-tail samples can be interrupted mid-rollout and resumed in a later step. Trainer and rollout phases remain serial. If you want the trainer/rollout fully decoupled and advancing concurrently, see `verl/experimental/fully_async_policy/` — not here.

Background and method comparison (APRIL) live in [REFERENCE.md](REFERENCE.md). This README only covers **when to use it, the architecture, and how to run it**.

---

## When to use

Use it when:

- The dataset has a **long-tailed response length distribution** (a small fraction of very long samples drags down each step).
- Synchronous PPO/GRPO shows a visible GPU bubble waiting on those long-tail samples.
- Training tolerates **mild off-policy drift** — partial rollout inherently spans multiple weight versions; pair it with IS correction.
- Multi-turn / tool-call workloads where the vLLM server can be interrupted (requires `PRv3vLLMHttpServer`, not the upstream server).

Skip it when:

- Response lengths are uniform with no long-tail bubble — the plain synchronous trainer is simpler.
- Strict on-policy is required (every trajectory must come from the current weights).
- You need to mix this pipeline with the upstream sync trainer's batch-shape assumptions — PRv3-specific fields (dummy `gen_batch`, `last_agent_loop_output`, …) would break them.

---

## Architecture

Three actors coordinated by one Ray actor scheduler:

```
                 trainer (PRv3RayPPOTrainer)
                        │
            push_batch  │  pull_batch
                        ▼
         ┌──────────────────────────────────┐
         │   RolloutPromptManager (Ray)     │
         │                                  │
         │   pending  ─pull─►  ongoing      │
         │      ▲                │          │
         │      └─aborted──push──┴─done─►   │
         └──────────────────────────────────┘
                        │
            pull_prompts│push_prompts
                        ▼
              PRv3AgentLoopWorker  ×N
                        │
              vLLM HTTP │ generate / cancel / resume
                        ▼
              PRv3vLLMHttpServer  ×replicas
```

| Component | File | Role |
|---|---|---|
| `PRv3RayPPOTrainer` | `ray_trainer.py` | Main trainer loop. `_fit_generate` pushes prompts into the manager, awaits one full batch via `async_rollout_manager.generate_sequences`, then runs log_prob / advantage / policy update. |
| `RolloutPromptManager` | `prompt_manager.py` | Single-threaded Ray actor holding the three queues (pending / ongoing / done). `pull_batch` is async + `asyncio.Event`-driven, no busy polling. |
| `PRv3AgentLoopManager` / `PRv3AgentLoopWorker` | `agent_loop/agent_loop.py` | Manager dispatches and drives vLLM cancel/resume; workers pull prompts, run the agent loop, push results back. |
| `PRv3SingleTurnAgentLoop` / `PRv3ToolAgentLoop` | `agent_loop/{single_turn,tool}_agent_loop.py` | Partial-rollout-aware versions of the upstream agent loops. Inspect `last_agent_loop_output`: pass through if done, resume from the interruption point if aborted. |
| `PRv3vLLMHttpServer` / `PRv3vLLMReplica` | `vllm_rollout/vllm_async_server.py` | Upstream vLLM HTTP server extended with `cancel()` / `resume()` — lets the manager interrupt in-flight generations once the batch is complete and write back partial token output. |

---

## Key invariants

1. **Prompt ownership during scheduling**: at any instant a prompt lives in exactly one of pending / ongoing / done. `pull_prompts` moves pending→ongoing; `push_prompts` decides ongoing→done or ongoing→pending (aborted resume) based on `is_prompt_done`.
2. **`stop_reason == "aborted"` is the resume signal**: anything else (including `missing` / `None`) is treated as done. This is intentional, not a bug — read the comments in `prompt_manager.py` before changing `is_prompt_done`.
3. **PRv3 agent loops resume via kwargs**: the training path always injects `kwargs["last_agent_loop_output"]`; the validate path doesn't, and PRv3 subclasses fall back to the upstream implementation by checking `kwargs["_prv3_is_validate"]`.
4. **The dummy `gen_batch` must carry `uid`**: when the dataloader is exhausted at the end of an epoch, the placeholder batch built to drain in-flight prompts still needs `non_tensor_batch["uid"]`, otherwise the manager can't compute the row count.
5. **Stateful dataloader resume**: `PRv3RayPPOTrainer.fit()` resumes via the stateful dataloader automatically — **don't** add manual skip-on-resume logic.

---

## Quick start

### Partial-rollout runs

Single-turn:
```bash
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo.sh
```

Multi-turn with tool calls — first generate the tool-agent dataset:
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
then launch:
```bash
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### Baseline runs (vanilla GRPO, for A/B against partial rollout)

Same model / data / batch / hyperparameters; the only differences from the PR variants are the upstream entry, the upstream agent loop, and no IS correction:
```bash
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo_baseline.sh
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh
```

### PRv3-specific Hydra overrides

| Key | Value | Notes |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `prv3_single_turn_agent` / `prv3_tool_agent` | Must be a `prv3_`-prefixed agent loop — worker asserts to prevent silent fallback. |
| `algorithm.rollout_correction.rollout_is` | `token` (recommended) / `sequence` | Sequence-level ratios easily hit the `exp(±20)` safety clamp once rollouts span several weight versions. |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS upper bound; for IcePop pass `"0.5_5.0"`. |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True` *(tool variant only)* | Enables multi-turn. |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML path | Tool registry; shared across backends. |

---

## File layout

```
partial_rollout/
├── README.md / README_zh.md / REFERENCE.md
├── main_ppo.py                # @hydra entry; wraps PRv3TaskRunner
├── ray_trainer.py             # PRv3RayPPOTrainer
├── prompt_manager.py          # RolloutPromptManager (Ray actor)
├── agent_loop/
│   ├── agent_loop.py          # PRv3AgentLoopManager / PRv3AgentLoopWorker
│   ├── single_turn_agent_loop.py   # PRv3SingleTurnAgentLoop  (@register prv3_single_turn_agent)
│   └── tool_agent_loop.py          # PRv3ToolAgentLoop        (@register prv3_tool_agent)
├── vllm_rollout/
│   └── vllm_async_server.py   # PRv3vLLMHttpServer / PRv3vLLMReplica
├── run_qwen3-0.6b_gsm8k_grpo.sh                 # PR, single-turn
├── run_qwen3-0.6b_gsm8k_grpo_tool.sh            # PR, tool-call
├── run_qwen3-0.6b_gsm8k_grpo_baseline.sh        # baseline, single-turn
├── run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh   # baseline, tool-call
└── run_{dapomath,gsm8k}_{nopr,pr}_grpo_4b_*.sh  # 4B Qwen3 ports of the recipe PR
```
