#!/bin/bash
caffeinate -i &
CAFFEINATE_PID=$!

echo "===== RUN 1: HCC ====="
python3 test_3_hcc_csv.py

echo "===== RUN 2: HC ====="
python3 test_3_hc_csv.py

echo "===== RUN 3: GA ====="
python3 test_3_ga_csv.py

kill $CAFFEINATE_PID