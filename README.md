TBD

curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync -U

task1 - 1_dataset_preparation.ipynb

task2 - 2_descriptor_calculation.ipynb

task3 - 3_model_training.ipynb

task4 - uv run python -m src.run_task4 --input data/cox2_final_dataset.csv --model_type qed_optimized --num_generated 20000 --num_selected 200 --output selected_hits_sa_fixed.csv && uv run python analyze_results.py
