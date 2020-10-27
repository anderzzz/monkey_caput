'''Bla bla

'''
from la_learner import LALearner

learner_1 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64, iselector=[0,1,2,3,4,5],
                      lr_init=0.01,
                      temperature=1.0, k_nearest_neighbours=100, clustering_repeats=5, number_of_centroids=100,
                      memory_mixing=0.5, n_samples=216)

def train_simple():
    learner_1.load_encoder('ae_learner_run_2')


if __name__ == '__main__':
    train_simple()