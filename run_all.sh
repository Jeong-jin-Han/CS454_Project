# !/bin/bash

# echo "===== RUN 1: HC (random init) ====="
# python test_4_hcc_csv.py \
#     --seeds 0 1 2 3 4 \
#     --random-init \
#     --source ./benchmark \
#     --output benchmark_log_test4_hcc_random_test

# echo ""
# echo "===== RUN 2: GA (biased init) ====="
# python test_4_ga_csv.py \
#     --seeds 0 1 2 3 4 \
#     --source ./benchmark \
#     --output benchmark_log_test4_ga_biased_test

# echo ""
# echo "===== RUN 3: GA (random init) ====="
# python test_4_ga_csv.py \
#     --seeds 0 1 2 3 4 \
#     --random-init \
#     --source ./benchmark \
#     --output benchmark_log_test4_ga_random_test

# echo ""
# echo "===== ALL DONE ====="

!/bin/bash

echo "===== RUN 1: HC (biased init) ====="
python test_4_base_csv.py \
    --seeds 0 1 2 3 4 \
    --source ./benchmark \
    --output benchmark_log_test4_base_biased_test

echo "===== RUN 2: HC (random init) ====="
python test_4_base_csv.py \
    --seeds 0 1 2 3 4 \
    --random-init \
    --source ./benchmark \
    --output benchmark_log_test4_base_random_test

echo ""
echo "===== ALL DONE ====="
