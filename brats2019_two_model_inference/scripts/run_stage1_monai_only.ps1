$ErrorActionPreference = "Stop"

python scripts/00_check_env.py
python scripts/01_prepare_data.py --auto --num_cases 2
python scripts/02_run_monai_inference.py
python scripts/04_normalize_predictions.py --models monai
python scripts/05_evaluate.py --models monai
python scripts/06_visualize_success_failure.py --models monai
python scripts/07_make_report.py --models monai
