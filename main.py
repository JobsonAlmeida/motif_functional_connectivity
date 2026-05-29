from pipeline.stage_20_band_filtering_and_epoch.band_filtering_and_epoch import run_band_filtering_and_epoch
from pipeline.stage_30_motif_sequence.motif_sequence import run_motif_sequence
from pipeline.stage_40_synchronization_matrix.synchronization_matrix import run_synchronization_matrix
from pipeline.stage_50_maximum_matrix.maximum_matrix import run_maximum_matrix
from pipeline.stage_60_synchronization_matrix_graph_measure.graph_measure import run_synchronization_matrix_graph_measure
from pipeline.stage_70_maximum_matrix_graph_measure.maximum_matrix_graph_measure import run_maximum_matrix_graph_measure
from pipeline.stage_80_lag_based_SVM.lag_based_SVM import run_lag_based_SVM

#from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM_old import run_maximum_matrix_based_SVM
#from pipeline.stage_90_1_maximum_matrix_based_SVM_sequential_feature_selector.maximum_matrix_based_SVM_sequential_feature_selector import run_maximum_matrix_based_SVM_sequential_feature_selector
from pipeline.stage_90_maximum_matrix_based_SVM.maximum_matrix_based_SVM import run_maximum_matrix_based_SVM

from pipeline.stage_100_generate_results.run_stage_90_in_loop import run_stage_90_in_loop
# # from pipeline.stage_100_generate_results.generate_accuracy_table import run_generate_accuracy_table
# from pipeline.stage_100_generate_results.generate_confusion_matrix_for_maximum_matrix import run_generate_confusion_matrix_for_maximum_matrix
# # from pipeline.stage_100_generate_results.run_stage_90 import run_stage_90
# # from pipeline.stage_100_generate_results.run_stage_90_1 import run_stage_90_1
# # from pipeline.stage_100_generate_results.run_stage_90_v2_experiments import run_stage_90_v2_experiments
# from pipeline.stage_100_generate_results.generate_topographic_plot import plot_topographic
# from pipeline.stage_100_generate_results.generate_band_contributions_across_k import plot_band_contributions_across_k
from pipeline.stage_100_generate_results.generate_accuracy_accross_k_plot import generate_accuracy_accross_k_plot


from pipeline.config.specific_group_channels import only_channels_D

from pathlib import Path
import os

current_file = Path(__file__).resolve()
project_root = current_file.parents[0]


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

    #run_maximum_matrix_based_SVM()
    #run_generate_confusion_matrix_for_maximum_matrix()

    #run_maximum_matrix_based_SVM_sequential_feature_selector()

    # run_stage_90_1(
    #     experiment_name = "experiment_1",
    #     use_feature_selector = True,
    #     max_k_features = 3        
    #     )


    # run_stage_90(
    #     experiment_name = "experiment_3",
    #     use_feature_selector=True, #False or True
    #     max_k_features=None,        #None significa usar o numero máximo possível de features 
    # )


    # run_maximum_matrix_based_SVM(feature_selector = "mutual_info",
    #                              k_features= 3)

    # run_stage_90_in_loop(
    #     feature_selector = "f_classif",
    #     k_features= range(1, 11),
    #     experiment_name="experiment_1", 
    # )

    generate_accuracy_accross_k_plot(k_features= range(1, 11), experiment_name="experiment_1")

    #plot_topographic()

    # plot_band_contributions_across_k()

    #generate_accuracy_accross_k_plot()
    
   


if __name__ == "__main__":
    main()