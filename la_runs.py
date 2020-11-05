'''Bla bla

'''
import pandas as pd
import numpy as np
from numpy.random import shuffle
from sklearn.preprocessing import normalize

from pathlib import Path
from torchvision.utils import save_image
from sklearn.cluster import KMeans

from img_transforms import UnNormalizeTransform
from la_learner import LALearner

uu = UnNormalizeTransform()
eval_clusterer = KMeans(n_clusters=20)

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
    #v2 = list(range(759, 2429))
    #shuffle(v1)
    #shuffle(v2)
    #vv = v1[:50] + v2[:100]
    vv = v1[:50]
    learner_2 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=128, iselector=vv,
                      dataset_type='grid basic idx',
                      selector=pd.IndexSlice[:, :, :, :, :, ['Cantharellaceae'], :, :, :],
                      lr_init=0.03, scheduler_step_size=5, scheduler_gamma=0.3,
                      temperature=0.07, k_nearest_neighbours=100, clustering_repeats=6, number_of_centroids=100,
                      memory_mixing=0.5, n_samples=1800)
    return learner_2

def get_learner_2x():
    v1 = list(range(759))
    v2 = list(range(759, 2429))
    shuffle(v1)
    shuffle(v2)
    vv = v1[:50] + v2[:150]
    #vv = v1[:100]
    learner_2 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=16, iselector=vv,
                      dataset_type='full basic idx',
                      selector=pd.IndexSlice[:, :, :, :, :, ['Cantharellaceae','Amanitaceae'], :, :, :],
                      lr_init=0.03, scheduler_step_size=5, scheduler_gamma=0.3,
                      temperature=0.07, k_nearest_neighbours=199, clustering_repeats=6, number_of_centroids=20,
                      memory_mixing=0.5, n_samples=200)
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
    learner_2.save_model('la_bigger_2')

def train_biggerx():
    learner_2 = get_learner_2x()
    learner_2.load_model('ae_kantflue_fullimage_wellconverged')
    img_array = []
    idx_array = []
    for dd in learner_2.dataloader:
        image = dd[learner_2.dataset.returnkey.image]
        idx = dd[learner_2.dataset.returnkey.idx]
        out = learner_2.model(image)
        out_npy = normalize(out.detach().numpy(), axis=1)
        img_array.append(out_npy)
        idx_array.append(idx.detach().numpy())
        print (idx.detach().numpy())
        print (out_npy.shape)
    img_vecs = np.concatenate(img_array, axis=0)
    idx_vecs = np.concatenate(idx_array, axis=0)
    print (img_vecs.shape)
    learner_2.criterion.memory_bank.memory_mixing_rate=1.0
    learner_2.criterion.memory_bank.update_memory(img_vecs, idx_vecs)
    learner_2.criterion.memory_bank.memory_mixing_rate=0.5
    learner_2.train(8)
    learner_2.save_model('la_bigger_2x')

def eval_bigger():
    learner_2 = get_learner_2x()
    learner_2.load_model('la_bigger_2x')
    cluster_labels = learner_2.eval(clusterer=eval_clusterer.fit_predict)
    saver_func(learner_2.dataloader, cluster_labels, './cluster_imgs', learner_2.dataset.returnkey.image)

def train_verybig():
    learner_3 =  get_learner_3()
    learner_3.load_model('ae_learner_2_bigger')
    learner_3.train(20)
    learner_3.save_model('la_verybig')

if __name__ == '__main__':
    #train_simple()
    #eval_simple()
    train_biggerx()
    eval_bigger()
    #train_verybig()