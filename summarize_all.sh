#!/usr/bin/env bash
set -euo pipefail

summarize_group() {
    local group="$1"
    local pattern="$2"

    echo "${group}"
    echo "total_time | seed | coverage"
    echo "--------------------------------"

    for seed in 41 42 43 44 45; do
        printf -v dir "${pattern}" "${seed}"
        if [ -d "${dir}" ]; then
            # 1) JSON에서 total_execution_time_formatted 가져오기
            total_time=$(
                python - "$dir" << 'PY'
import json, os, sys
logdir = sys.argv[1]
cfg = os.path.join(logdir, "test_config.json")

if not os.path.exists(cfg):
    print("N/A")
    sys.exit(0)

with open(cfg, "r", encoding="utf-8") as f:
    data = json.load(f)

print(data.get("total_execution_time_formatted", "N/A"))
PY
            )

            # 2) coverage_generator.py 실행해서 Final coverage 한 줄 뽑기
            cov_line=$(python coverage_generator.py "${dir}" 2>/dev/null | grep "Final coverage" | tail -n 1 || true)

            # cov_line 예: "Final coverage: 3447% / 3800%"
            if [ -n "${cov_line}" ]; then
                coverage=${cov_line#*Final coverage: }
            else
                coverage="N/A"
            fi

            # 3) 출력
            printf "%9s | %3d | %s\n" "${total_time}" "${seed}" "${coverage}"
        fi
    done
    echo
}

echo "=== Summary (all existing logs) ==="

# Biased
summarize_group "HCC (biased)" "benchmark_log_1_biased_%d"
summarize_group "HC (biased)"  "benchmark_log_2_biased_%d"
summarize_group "GA (biased)"  "benchmark_log_ga_biased_%d"

# Random-init
summarize_group "HCC (random)" "benchmark_log_1_random_%d"
summarize_group "HC (random)"  "benchmark_log_2_random_%d"
summarize_group "GA (random)"  "benchmark_log_ga_random_%d"
