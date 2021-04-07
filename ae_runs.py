'''Bla bla

'''
from pandas import IndexSlice
from numpy.random import shuffle

from ae_learner import AELearner
from img_transforms import UnNormalizeTransform


#learner_1 = AELearner(run_label='simple test run',
#                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
#                      dataset_type='grid basic',
#                      loader_batch_size=64, iselector=[0,1,2,3,4,5],
#                      lr_init=0.01, freeze_encoder=True,
#                      random_seed=79)
#
#tt = IndexSlice[:, :, :, :, :, ['Cantharellaceae'], :, :, :]
#learner_2 = AELearner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
#                      loader_batch_size=128, selector=tt,
#                      iselector=list(range(120)),
#                      lr_init=0.03, scheduler_step_size=12,
#                      freeze_encoder=False,
#                      random_seed=79)

tt = IndexSlice[:, :, :, :, :, ['Cantharellaceae', 'Amanitaceae'], :, :, :]
v1 = list(range(759))
v2 = list(range(759, 2429))
shuffle(v1)
shuffle(v2)
vv = v1[:100] + v2[:300]
learner_3 = AELearner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      dataset_type='grid basic',
                      loader_batch_size=64, selector=tt,
                      iselector=vv,
                      lr_init=0.01, scheduler_step_size=8, scheduler_gamma=0.1,
                      freeze_encoder=False,
                      random_seed=79)

def train_from_scratch():
    learner_1.train(6)
    learner_1.save_model('ae_learner_run_1')

def train_from_existing():
    learner_1.load_model('ae_learner_run_1')
    learner_1.train(6)
    learner_1.save_model('ae_learner_run_2')

def eval_from_existing():
    learner_1.load_model('ae_learner_run_2')
    for out in learner_1.eval_model(untransform=UnNormalizeTransform()):
        print (out.shape)

def train_bigger():
    learner_3.load_model('ae_learner_2_bigger')
    learner_3.train(16)
    learner_3.save_model('ae_learner_2_bigger_2')


if __name__ == '__main__':
    #train_from_scratch()
    #train_from_existing()
    #eval_from_existing()
    train_bigger()
