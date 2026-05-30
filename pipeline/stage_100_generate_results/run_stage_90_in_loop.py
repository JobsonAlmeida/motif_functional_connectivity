
from pathlib import Path
import os
import json

from pipeline.config.build_feature_selector import build_feature_selector

from datetime import datetime
import logging

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

result_name = "run_stage_90_in_loop"
output_path = project_root / "results" / current_file.parents[0].name / result_name

from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM import (
    run_maximum_matrix_based_SVM
)



def run_stage_90_in_loop(
        feature_selector = "f_classif",
        k_features = [20, 40, 60, 80, 100],
        output_path = output_path,
        experiment_name = "experiment_base" 
):
    
    k_features = list(k_features)
    
    experiment_output_path = output_path/experiment_name

    try:
        experiment_output_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f'Error: The experiment "{experiment_name}" already exists in "{output_path}". '
            f'Delete the folder or change the name.'
        ) 

    log_path = experiment_output_path / "execution.log"

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    start_time = datetime.now()
    logging.info("===== EXPERIMENT START =====")

    _ , selector_config = build_feature_selector(feature_selector, k_features)

    config_path = experiment_output_path / "experiment_config.json"

    with open(config_path, "w") as f:
        json.dump(selector_config, f, indent=4)

    for k in k_features:

        print(f"\nExecutando experimento com k_features={k}\n")
        logging.info(f"Executando k_features={k}")


        run_maximum_matrix_based_SVM(
            feature_selector = feature_selector,
            k_features=k,
            output_path= experiment_output_path,
        )

    end_time = datetime.now()

    logging.info("===== EXPERIMENT END =====")
    logging.info(f"Start: {start_time}")
    logging.info(f"End: {end_time}")
    logging.info(f"Execution time: {end_time - start_time}")


if __name__ == "__main__":

    run_stage_90_in_loop()