set -euo pipefail

MODE="${1:-all}"

LOG_LIST=()

############################################
# BIASED VERSIONS
############################################

run_hcc() {
    echo "=== Running HCC (biased) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_1_biased_${seed}"
        python test_benchmark_parallel_csv.py --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

run_hc() {
    echo "=== Running HC (biased) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_2_biased_${seed}"
        python test_benchmark_base_parallel_csv.py --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

run_ga() {
    echo "=== Running GA (biased) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_ga_biased_${seed}"
        python test_benchmark_ga_csv.py --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

############################################
# RANDOM-INIT VERSIONS
############################################

run_hcc_random() {
    echo "=== Running HCC (random-init) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_1_random_${seed}"
        python test_benchmark_parallel_csv.py --random-init --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

run_hc_random() {
    echo "=== Running HC (random-init) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_2_random_${seed}"
        python test_benchmark_base_parallel_csv.py --random-init --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

run_ga_random() {
    echo "=== Running GA (random-init) ==="
    for seed in 46 47 48 49 50; do
        out="benchmark_log_ga_random_${seed}"
        python test_benchmark_ga_csv.py --random-init --output "${out}" --seed "${seed}"
        LOG_LIST+=("${out}")
    done
}

############################################
# SELECT WORKLOAD
############################################

case "${MODE}" in
    hcc) run_hcc ;;
    hc) run_hc ;;
    ga) run_ga ;;

    hcc-random) run_hcc_random ;;
    hc-random) run_hc_random ;;
    ga-random) run_ga_random ;;

    all)
        run_hcc
        run_hc
        run_ga
        ;;

    all-random)
        run_hcc_random
        run_hc_random
        run_ga_random
        ;;

    full)
        run_hcc
        run_hc
        run_ga
        run_hcc_random
        run_hc_random
        run_ga_random
        ;;

    *)
        echo "Usage: $0 {hcc|hc|ga|hcc-random|hc-random|ga-random|all|all-random|full}"
        exit 1
        ;;
esac


############################################
# RUN COVERAGE GENERATOR
############################################

echo
echo "=== Running coverage_generator.py on ${#LOG_LIST[@]} logs ==="
for logdir in "${LOG_LIST[@]}"; do
    python coverage_generator.py "${logdir}"
done

############################################
# SUMMARY OUTPUT
############################################

summarize_group() {
    local group="$1"
    local pattern="$2"

    echo "${group}"
    echo "total_time | seed | coverage"
    echo "--------------------------------"

    for seed in 46 47 48 49 50; do
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

echo "=== Summary ==="
case "${MODE}" in
    hcc) summarize_group "HCC (biased)" "benchmark_log_1_biased_%d" ;;
    hc)  summarize_group "HC (biased)"  "benchmark_log_2_biased_%d" ;;
    ga)  summarize_group "GA (biased)"  "benchmark_log_ga_biased_%d" ;;

    hcc-random) summarize_group "HCC (random)" "benchmark_log_1_random_%d" ;;
    hc-random)  summarize_group "HC (random)"  "benchmark_log_2_random_%d" ;;
    ga-random)  summarize_group "GA (random)"  "benchmark_log_ga_random_%d" ;;

    all)
        summarize_group "HCC (biased)" "benchmark_log_1_biased_%d"
        summarize_group "HC (biased)"  "benchmark_log_2_biased_%d"
        summarize_group "GA (biased)"  "benchmark_log_ga_biased_%d"
        ;;

    all-random)
        summarize_group "HCC (random)" "benchmark_log_1_random_%d"
        summarize_group "HC (random)"  "benchmark_log_2_random_%d"
        summarize_group "GA (random)"  "benchmark_log_ga_random_%d"
        ;;

    full)
        summarize_group "HCC (biased)" "benchmark_log_1_biased_%d"
        summarize_group "HC (biased)"  "benchmark_log_2_biased_%d"
        summarize_group "GA (biased)"  "benchmark_log_ga_biased_%d"
        summarize_group "HCC (random)" "benchmark_log_1_random_%d"
        summarize_group "HC (random)"  "benchmark_log_2_random_%d"
        summarize_group "GA (random)"  "benchmark_log_ga_random_%d"
        ;;
esac

