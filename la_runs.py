'''Bla bla

'''
import pandas as pd
from numpy.random import shuffle

from pathlib import Path
from torchvision.utils import save_image
from sklearn.cluster import KMeans

from fungiimg import UnNormalizeTransform
from la_learner import LALearner

uu = UnNormalizeTransform()
eval_clusterer = KMeans(n_clusters=100)

def get_learner_1():
    learner_1 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64, iselector=[0,1,2,3,4,5,6,7,8,9],
                      dataset_type='grid basic idx',
                      lr_init=0.01,
                      temperature=0.07, k_nearest_neighbours=100, clustering_repeats=5, number_of_centroids=100,
                      memory_mixing=0.5, n_samples=360)
    return learner_1

def get_learner_2():
    v1 = list(range(759))
    v2 = list(range(759, 2429))
    shuffle(v1)
    shuffle(v2)
    vv = v1[:50] + v2[:100]
    learner_2 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=128, iselector=vv,
                      selector=pd.IndexSlice[:, :, :, :, :, ['Cantharellaceae', 'Amanitaceae'], :, :, :],
                      lr_init=0.03, scheduler_step_size=5, scheduler_gamma=0.3,
                      temperature=0.07, k_nearest_neighbours=500, clustering_repeats=6, number_of_centroids=20,
                      memory_mixing=0.5, n_samples=5400)
    return learner_2

def get_learner_3():
    v1 = list(range(759))
    v2 = list(range(759, 2429))
    shuffle(v1)
    shuffle(v2)
    vv = v1[:100] + v2[:300]
    learner = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=128, iselector=vv,
                      selector=pd.IndexSlice[:, :, :, :, :, ['Cantharellaceae', 'Amanitaceae'], :, :, :],
                      lr_init=0.03, scheduler_step_size=5, scheduler_gamma=0.3,
                      temperature=0.07, k_nearest_neighbours=1000, clustering_repeats=6, number_of_centroids=2000,
                      memory_mixing=0.5, n_samples=14400)
    return learner

def saver_func(dloader, c_labels, root_dir, img_key):
    Path(root_dir).mkdir(parents=False, exist_ok=False)

    n_img = 0
    for data in dloader:
        image_tensor = data[img_key]
        for image in image_tensor:
            img_ = uu(image)
            the_cluster = c_labels[n_img]

            subfolder = '{}/cluster_{}'.format(root_dir, the_cluster)
            Path(subfolder).mkdir(parents=False, exist_ok=True)
            save_image(img_, '{}/{}.png'.format(subfolder, n_img))
            n_img += 1

def train_simple():
    learner_1 = get_learner_1()
    learner_1.load_model('ae_learner_2_bigger')
    learner_1.train(4)
    learner_1.save_model('la_simple')

def eval_simple():
    learner_2 = get_learner_2()
    learner_2.load_model('la_bigger')
    cluster_labels = learner_2.eval(clusterer=eval_clusterer.fit_predict)
    saver_func(learner_2.dataloader, cluster_labels, './cluster_imgs', learner_2.dataset.getkeys.image)

def train_bigger():
    learner_2 = get_learner_2()
    learner_2.load_model('ae_learner_2_bigger')
    learner_2.train(20)
    learner_2.save_model('la_bigger')

def train_verybig():
    learner_3 =  get_learner_3()
    learner_3.load_model('ae_learner_2_bigger')
    learner_3.train(20)
    learner_3.save_model('la_verybig')

if __name__ == '__main__':
    train_simple()
    #eval_simple()
    #train_bigger()
    #train_verybig()