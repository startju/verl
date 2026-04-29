# Recipe: Partial Rollout (PRv3)

[English](README.md) | 简体中文

**作者：** Yue Wang, Zhipeng Ma, Yi Yan, Hang Xu, Yang Li, Bo Qian, Peng Chen, Xingyu Zhao

**同步 RL 训练模式**下的 partial-rollout 流水线，针对**长尾响应长度**带来的 GPU 闲置问题做样本补充与中断续跑。

> ⚠️ **不要跟全异步框架的 partial rollout 混淆**：本流水线仍走「rollout → 等批次齐 → 一次性 train step」的同步循环，只是允许在 rollout 阶段中断长尾样本、跨 step 续跑；trainer 与 rollout 之间是顺序串行的。如果你需要的是 trainer / rollout 完全解耦异步推进的版本，看 `verl/experimental/fully_async_policy/`，不是这里。

学术背景与方法对照（APRIL）放在 [REFERENCE.md](REFERENCE.md)，本 README 只讲**何时用、整体架构、怎么跑**。

---

## 使用场景

适用：

- 数据集**响应长度分布长尾**（少量超长样本拖慢整批 step）
- 同步 PPO/GRPO 因等待长尾样本导致 GPU bubble 明显
- 训练对**轻微 off-policy** 容忍（partial rollout 必然引入 weight-version 跨越，需配合 IS 修正）
- 多轮 / tool-call 场景里 vLLM 服务端能被中断（需要 PRv3vLLMHttpServer 替代上游 server）

不适用：

- 响应长度均匀、没有 long-tail bubble — 同步 trainer 更简单
- 严格 on-policy 必须保证（每个 trajectory 只能由当前权重产出）
- 同时使用本流水线 + 上游 sync trainer 的 batch shape 假设（dummy gen_batch / `last_agent_loop_output` 等 PRv3 专属字段会破坏）

---

## 整体架构

三个角色，靠一个 Ray actor 做调度：

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

- **`PRv3RayPPOTrainer`** (`ray_trainer.py`)：trainer 主循环。`_fit_generate` 把 prompt push 进 manager，调 `async_rollout_manager.generate_sequences` 等一个完整 batch 回来，再走 log_prob / advantage / policy update。
- **`RolloutPromptManager`** (`prompt_manager.py`)：单线程 Ray actor，维护三个数据结构（pending / ongoing / done_queue），保证不丢 prompt、不重复入队。`pull_batch` 是 async + `asyncio.Event` 驱动，无 busy poll。
- **`PRv3AgentLoopManager` / `PRv3AgentLoopWorker`** (`agent_loop/agent_loop.py`)：管理 agent loop worker 池。manager 负责调度 + cancel/resume vLLM；worker 拉 prompt、跑 agent loop、把结果回 push。
- **`PRv3SingleTurnAgentLoop` / `PRv3ToolAgentLoop`** (`agent_loop/`)：上游 agent loop 的 partial-rollout 改造版。每次执行前看 `last_agent_loop_output`：完成态直接放行；aborted 状态从中断点续跑（拼接前轮 prompt+response 当新输入）。
- **`PRv3vLLMHttpServer` / `PRv3vLLMReplica`** (`vllm_rollout/vllm_async_server.py`)：上游 vLLM HTTP server 的扩展，加 `cancel()` / `resume()` —— 让 manager 在拿到一个完整 batch 后中断尚未完成的生成、把 token 回写当部分 rollout。

---

## 关键不变量

1. **prompt 流转的所有权**：每个 prompt 同一时刻只在 pending / ongoing / done 之一。`pull_prompts` 移 pending→ongoing；`push_prompts` 按 `is_prompt_done` 决定 ongoing→done 还是 ongoing→pending（aborted 续跑）。
2. **`stop_reason == "aborted"` 是续跑信号**：其它（含 `missing` / `None`）一律视为完成（设计如此，不是 bug）。改 `is_prompt_done` 前先看 `prompt_manager.py` 注释。
3. **PRv3 agent loop 通过 kwargs 续跑**：训练路径 `kwargs["last_agent_loop_output"]` 必传；validate 路径不传，PRv3 子类靠 `kwargs["_prv3_is_validate"]` 退化到上游实现。
4. **dummy gen_batch 也要带 uid**：epoch 末 dataloader 耗尽时构造的占位 batch 必须填 `non_tensor_batch["uid"]`，否则 manager 端取不到行数。
5. **stateful dataloader 续训**：`PRv3RayPPOTrainer.fit()` 走 stateful loader 自动恢复进度，**不要**手工加 skip-on-resume 逻辑。

---

## 快速上手

### 单轮 (single-turn)
```bash
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo.sh
```

### 多轮工具调用 (tool agent)
先生成 tool-agent 数据集：
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
再启动：
```bash
bash verl/experimental/partial_rollout/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### 关键 Hydra override

| 项 | 值 | 说明 |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `prv3_single_turn_agent` / `prv3_tool_agent` | 必须用 `prv3_` 开头的 agent loop（worker 里有 assert 防退化） |
| `algorithm.rollout_correction.rollout_is` | `token` 或 `sequence` | 推荐 `token`（partial rollout 跨权重版本，sequence 级 ratio 易被 clamp） |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS 上限；想用 IcePop 写 `"0.5_5.0"` |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True`（仅 tool 版） | 启用多轮 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML 路径 | tool registry，跨 backend 共用 |
