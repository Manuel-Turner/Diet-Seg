$ErrorActionPreference = "Stop"

python scripts/00_check_env.py
python scripts/01_prepare_data.py --auto --num_cases 6
python scripts/02_run_monai_inference.py
python scripts/03_run_kaist_inference.py
python scripts/04_normalize_predictions.py
python scripts/05_evaluate.py
python scripts/06_visualize_success_failure.py
python scripts/07_make_report.py
