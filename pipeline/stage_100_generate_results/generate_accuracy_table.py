import os
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
import networkx as nx

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


import pickle
import pandas as pd
import matplotlib.pyplot as plt



current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

result_name = "accuracy_table"
base_path = project_root / "processed_data" / "stage_80_lag_based_SVM"
output_path = project_root / "results" / current_file.parents[0].name / result_name

base_path_maximum_matrix = project_root / "processed_data" / "stage_90_maximum_matrix_based_SVM"


os.makedirs(output_path, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, 11)]
sessions = [f"ses-{i:02d}" for i in range(1, 4)]

def run_generate_accuracy_table():

    rows = []

    for subject in subjects:

        file_path = os.path.join(
            base_path,
            f"{subject}_svm_results.pkl"
        )

        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            continue

        with open(file_path, "rb") as f:
            results = pickle.load(f)

        file_path_max_matrices = os.path.join(
            base_path_maximum_matrix,
            f"{subject}_maximum_matrices_svm_results_k_2816.pkl"
        )

        if not os.path.exists(file_path_max_matrices):
            print(f"Arquivo não encontrado: {file_path_max_matrices}")
            continue
        
        with open(file_path_max_matrices, "rb") as f:
            results_max_matrices = pickle.load(f)

        row = {
            "Subject": subject
        }


        # Obtendo as acurácias para cada lag
        for lag_name in results[subject]:

            mean_acc = results[subject][lag_name]["mean_accuracy"]

            row[lag_name] = mean_acc * 100


        # obtendo as acurácias da matriz de máximo
        for max_type in results_max_matrices[subject]:

            mean_acc = results_max_matrices[subject][max_type]["mean_accuracy"]

            row[max_type] = mean_acc * 100        

        rows.append(row) 


    df = pd.DataFrame(rows) 

    # Anexando linhas de media e desvio padrao na tabela
    mean_row = {
        "Subject": "Mean"
    }

    std_row = {
        "Subject": "Std"
    }

    lag_columns = [col for col in df.columns if col != "Subject"]

    for col in lag_columns:

        mean_row[col] = df[col].mean()
        std_row[col] = df[col].std()

    df = pd.concat([
        df,
        pd.DataFrame([mean_row]),
        pd.DataFrame([std_row])
    ], ignore_index=True) 

    df[lag_columns] = df[lag_columns].round(2) # arredondando valores

    desired_order = [
        "Subject",
        "max_0_1_2_3",
        "max_1_2_3",
        "lag_0",
        "lag_1",
        "lag_2",
        "lag_3"
    ]

    df = df[desired_order]

    # Renomeando colunas
    df = df.rename(columns={
        "max_0_1_2_3": "Max (0,1,2,3) %",
        "max_1_2_3": "Max (1,2,3) %",
        "lag_0": "Lag 0 (%)",
        "lag_1": "Lag 1 (%)",
        "lag_2": "Lag 2 (%)",
        "lag_3": "Lag 3 (%)"
    })

    # renomeando linhas
    df["Subject"] = df["Subject"].replace({
        "sub-01": "S1",
        "sub-02": "S2",
        "sub-03": "S3",
        "sub-04": "S4",
        "sub-05": "S5",
        "sub-06": "S6",
        "sub-07": "S7",
        "sub-08": "S8",
        "sub-09": "S9",
        "sub-10": "S10"
    })

    save_path = os.path.join(
        output_path,
        f"{result_name}.csv"
    )

    df.to_csv(save_path, index=False)


    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 4))

    # Remover eixos
    ax.axis("off")

    # Criar tabela
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    # Ajustes visuais
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)


    # Colocando maior valor de cada linha em negrito

    # Colunas numéricas (ignorando Subject)
    numeric_cols = df.columns[1:]

    # Percorrendo linhas do dataframe
    for row_idx in range(len(df)):

        # Valores numéricos da linha
        values = df.loc[row_idx, numeric_cols].astype(float)

        # Índice da maior coluna
        max_col_idx = values.argmax()

        # Ajuste porque a coluna 0 é "Subject"
        table_col_idx = max_col_idx + 1

        # +1 porque a linha 0 da tabela é o cabeçalho
        table_row_idx = row_idx + 1

        # Aplicando negrito
        table[(table_row_idx, table_col_idx)].set_text_props(
            fontweight="bold"
        )

    # Título
    ax.set_title(
        "Table 1 - Mean SVM classification accuracy (%) obtained using 5-fold cross-validation",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.figtext(
        0.05,
        0.01,
        "Values represent the mean classification accuracy (%) across 5-fold cross-validation.\n"
        "For each lag condition, the training and test data were obtained using all filtered frequency bands.",
        ha="left",
        fontsize=10
    )
    
    save_path = os.path.join(
            output_path,
            f"{result_name}.png"
        )
    # Salvar
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    



if __name__ == "__main__":

    run_generate_accuracy_table()