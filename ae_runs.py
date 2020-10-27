'''Bla bla

'''
from ae_learner import AELearner

learner_1 = AELearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64, iselector=[0,1,2,3,4,5],
                      lr_init=0.01, freeze_encoder=True,
                      random_seed=79)

def train_from_scratch():
    learner_1.train(3)
    learner_1.save_ae('ae_learner_run_1')

def train_from_existing():
    learner_1.load_ae('ae_learner_run_1')
    learner_1.train(3)
    learner_1.save_ae('ae_learner_run_2')


if __name__ == '__main__':
    train_from_scratch()
    train_from_existing()