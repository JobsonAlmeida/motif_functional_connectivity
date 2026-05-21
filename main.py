from pipeline.stage_20_band_filtering_and_epoch.band_filtering_and_epoch import run_band_filtering_and_epoch
from pipeline.stage_30_motif_sequence.motif_sequence import run_motif_sequence
from pipeline.stage_40_synchronization_matrix.synchronization_matrix import run_synchronization_matrix
from pipeline.stage_50_maximum_matrix.maximum_matrix import run_maximum_matrix
from pipeline.stage_60_synchronization_matrix_graph_measure.graph_measure import run_synchronization_matrix_graph_measure
from pipeline.stage_70_maximum_matrix_graph_measure.maximum_matrix_graph_measure import run_maximum_matrix_graph_measure
from pipeline.stage_80_lag_based_SVM.lag_based_SVM import run_lag_based_SVM
from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM import run_maximum_matrix_based_SVM

from pipeline.stage_100_generate_results.generate_accuracy_table import run_generate_accuracy_table
from pipeline.stage_100_generate_results.generate_confusion_matrix_for_maximum_matrix import run_generate_confusion_matrix_for_maximum_matrix
from pipeline.stage_100_generate_results.run_stage_90 import run_stage_90

from pipeline.config.specific_group_channels import only_channels_D

def main():

    #run_band_filtering_and_epoch()
    #run_motif_sequence()
    #run_synchronization_matrix()

    #run_maximum_matrix()

    #run_maximum_matrix()

    #run_synchronization_matrix_graph_measure()
    #run_maximum_matrix_graph_measure()

    #run_lag_based_SVM()

    #run_generate_accuracy_table()

    #run_maximum_matrix()

    #run_maximum_matrix_graph_measure()

    # run_maximum_matrix_based_SVM()


    run_stage_90(
        experiment_name = "experiment_4",

        use_feature_selector=True, #False or True
        specific_k_features = False,    # False o um número 
        max_k_features=None,        #None significa usar o numero máximo possível de features 
        

        selected_epochs=None,
        selected_bands=None,
        selected_channels=only_channels_D,
        selected_measures=None
    )

    

    


if __name__ == "__main__":
    main()