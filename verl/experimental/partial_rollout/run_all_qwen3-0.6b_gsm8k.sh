#!/usr/bin/env bash
#
# Chain runner: launches all four 0.6B GSM8K experiments sequentially.
#
# Order (PR variants first so the partial-rollout curves land before baselines):
#   1. PR single-turn          (~4h25m typical)
#   2. PR tool                 (~5h-7h typical)
#   3. Baseline single-turn    (~4h25m typical)
#   4. Baseline tool           (~5h-7h typical)
#
# fail-fast: any non-zero exit aborts the chain and skips the remaining runs.
# Each run's per-script log is preserved (verl_*.log via the run script's own
# tee). Chain-level [CHAIN] timestamps land in the bash task output stream.
#
# Usage:
#   bash verl/experimental/partial_rollout/run_all_qwen3-0.6b_gsm8k.sh

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_one() {
    local script="$1"
    local label="$2"
    echo "[CHAIN] === START $label at $(date '+%Y-%m-%d %H:%M:%S') ==="
    bash "$DIR/$script"
    echo "[CHAIN] === END   $label at $(date '+%Y-%m-%d %H:%M:%S') ==="
}

run_one run_qwen3-0.6b_gsm8k_grpo.sh                "PR single-turn"
run_one run_qwen3-0.6b_gsm8k_grpo_tool.sh           "PR tool-call"
run_one run_qwen3-0.6b_gsm8k_grpo_baseline.sh       "baseline single-turn"
run_one run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh  "baseline tool-call"

echo "[CHAIN] All four runs complete at $(date '+%Y-%m-%d %H:%M:%S')"
