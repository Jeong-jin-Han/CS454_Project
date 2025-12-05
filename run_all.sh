#!/bin/bash
caffeinate -i &
CAFFEINATE_PID=$!

echo "===== RUN 1: HCC (random init) ====="
python test_4_hcc_csv.py \
    --seeds 41 42 43 44 45 \
    --random-init \
    --source ./benchmark \
    --output benchmark_log_test4_hcc_random_test

echo "===== RUN 1: HCC (biased init) ====="
python test_4_hcc_csv.py \
    --seeds 41 42 43 44 45 \
    --source ./benchmark \
    --output benchmark_log_test4_hcc_biased_test

echo "===== RUN 2: GA (biased init) ====="
python test_4_ga_csv.py \
    --seeds 41 42 43 44 45 \
    --source ./benchmark \
    --output benchmark_log_test4_ga_biased_test

echo ""
echo "===== RUN 3: GA (random init) ====="
python test_4_ga_csv.py \
    --seeds 41 42 43 44 45 \
    --random-init \
    --source ./benchmark \
    --output benchmark_log_test4_ga_random_test

echo "===== RUN 1: HC (biased init) ====="
python test_4_base_csv.py \
    --seeds 41 42 43 44 45 \
    --source ./benchmark \
    --output benchmark_log_test4_hc_biased_test

echo "===== RUN 2: HC (random init) ====="
python test_4_base_csv.py \
    --seeds 41 42 43 44 45 \
    --random-init \
    --source ./benchmark \
    --output benchmark_log_test4_hc_random_test

echo ""
echo "===== ALL DONE ====="

kill $CAFFEINATE_PID