from pipeline.stage_20_band_filtering.band_filtering import run_band_filtering
from pipeline.stage_30_motif_sequence.motif_sequence import run_motif_sequence
from pipeline.stage_40_synchronization_matrix.synchronization_matrix import run_synchronization_matrix
from pipeline.stage_60_synchronization_matrix_graph_measure.graph_measure import run_synchronization_matrix_graph_measure
from pipeline.stage_80_lag_based_SVM.lag_based_SVM import run_lag_based_SVM

def main():

    #run_band_filtering()
    #run_motif_sequence()
    #run_synchronization_matrix()
    #run_synchronization_matrix_graph_measure()
    run_lag_based_SVM()


if __name__ == "__main__":
    main()