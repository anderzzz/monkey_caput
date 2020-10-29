'''Bla bla

'''
from pandas import IndexSlice
from numpy.random import shuffle

from ic_learner import ICLearner

label_binary_cf = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')

tt = IndexSlice[:, :, :, :, :, ['Cantharellaceae', 'Amanitaceae'], :, :, :]
v1 = list(range(759))
v2 = list(range(759, 2429))
shuffle(v1)
shuffle(v2)
vv = v1[:350] + v2[:700]
learner_1 = ICLearner(run_label='simple classification test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64,
                      selector=tt, iselector=vv,
                      label_keys=label_binary_cf,
                      lr_init=0.01,
                      random_seed=79)

def train_simple_ic():
    learner_1.train(2)
    learner_1.save_model('ic_run_1')

if __name__ == '__main__':
    train_simple_ic()