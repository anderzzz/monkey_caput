'''Bla bla

'''
from pandas import IndexSlice
from numpy.random import shuffle

from torch.utils.data import DataLoader

from ic_learner import ICLearner
from fungidata import factory

label_binary_cf = ('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')

tt = IndexSlice[:, :, :, :, :, ['Cantharellaceae', 'Amanitaceae'], :, :, :]
v1 = list(range(759))
v2 = list(range(759, 2429))
shuffle(v1)
shuffle(v2)
vv = v1[:20] + v2[:40]
vv_test = v1[20:30] + v2[40:55]
dataset2 = factory.create('full basic labelled', csv_file='../../Desktop/Fungi/toc_full.csv',
                         img_root_dir='../../Desktop/Fungi', label_keys=label_binary_cf,
                         selector=tt, iselector=vv_test, min_dim=299)
dataloader_test = DataLoader(dataset2, batch_size=16, shuffle=False)

learner_1 = ICLearner(run_label='simple classification test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64,
                      selector=tt, iselector=vv,
                      label_keys=label_binary_cf,
                      lr_init=0.01,
                      random_seed=79)

learner_2 = ICLearner(run_label='simple classification test run with data augmentation',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64,
                      selector=tt, iselector=vv,
                      label_keys=label_binary_cf,
                      dataset_type='full aug labelled',
                      ic_model='inception_v3',min_dim=299,
                      aug_multiplicity=1, aug_label='random_resized_crop_rotation',
                      lr_init=0.01,
                      random_seed=79, test_dataloader=dataloader_test)

def train_simple_ic():
    learner_1.train(2)
    learner_1.save_model('ic_run_1')

def train_aug_ic():
    learner_2.train(5)
    learner_2.save_model('ic_run_2')

if __name__ == '__main__':
    #train_simple_ic()
    train_aug_ic()