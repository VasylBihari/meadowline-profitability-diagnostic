#!/usr/bin/env bash
# Runs the full pipeline in order. Needs: python3, pandas, numpy (see requirements.txt).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p docs/logs
for s in 01_data_quality 02_clean 03_revenue_baseline 04_product_discount_return_economics 05_followup_checks 06_customers_marketing 07_qa_and_sizing; do
  echo ">>> $s"; python3 src/$s.py > docs/logs/$s.log
done
echo "Done. Tables in outputs/, logs in docs/logs/."
