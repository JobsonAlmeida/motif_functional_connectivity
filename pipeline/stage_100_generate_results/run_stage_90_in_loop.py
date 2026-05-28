
from pathlib import Path
import os

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

result_name = "run_stage_90_in_loop"
output_path = project_root / "results" / current_file.parents[0].name / result_name

from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM import (
    run_maximum_matrix_based_SVM
)

def run_stage_90_in_loop(
        k_features = [3,4,5],
        output_path = output_path,
        experiment_name = "experiment_base" 
):
    
    experiment_output_path = output_path/experiment_name

    try:
        experiment_output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f'Error: The experiment "{experiment_name}" already exists in "{output_path}". '
            f'Delete the folder or change the name.'
        ) 


    for k in k_features:

        print(f"\nExecutando experimento com k_features={k}\n")

        run_maximum_matrix_based_SVM(
            k_features=k,
            output_path= experiment_output_path,
        )

if __name__ == "__main__":

    run_stage_90_in_loop()