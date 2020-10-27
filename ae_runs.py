'''Bla bla

'''
from pandas import IndexSlice

from ae_learner import AELearner

learner_1 = AELearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64, iselector=[0,1,2,3,4,5],
                      lr_init=0.01, freeze_encoder=True,
                      random_seed=79)

tt = IndexSlice[:, :, :, :, :, ['Cantharellaceae'], :, :, :]
learner_2 = AELearner(raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=128, selector=tt,
                      iselector=list(range(120)),
                      lr_init=0.03, scheduler_step_size=12,
                      freeze_encoder=False,
                      random_seed=79)

def train_from_scratch():
    learner_1.train(3)
    learner_1.save_model('ae_learner_run_1')

def train_from_existing():
    learner_1.load_model('ae_learner_run_1')
    learner_1.train(3)
    learner_1.save_model('ae_learner_run_2')

def eval_from_existing():
    learner_1.load_model('ae_learner_run_2')
    learner_1.eval_model(eval_img_prefix='./save_dummy/eval_img')

def train_bigger():
    learner_2.load_model('model_in_progress')
    learner_2.train(30)
    learner_2.save_model('ae_learner_2_bigger')


if __name__ == '__main__':
    train_from_scratch()
    #train_from_existing()
    #eval_from_existing()
    #train_bigger()