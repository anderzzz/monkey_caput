'''Bla bla

'''
from pathlib import Path
from torchvision.utils import save_image
from sklearn.cluster import KMeans

from fungiimg import UnNormalizeTransform
from la_learner import LALearner

learner_1 = LALearner(run_label='simple test run',
                      raw_csv_toc='../../Desktop/Fungi/toc_full.csv', raw_csv_root='../../Desktop/Fungi',
                      loader_batch_size=64, iselector=[0,1,2,3,4,5,6,7,8,9],
                      lr_init=0.01,
                      temperature=0.07, k_nearest_neighbours=100, clustering_repeats=5, number_of_centroids=100,
                      memory_mixing=0.5, n_samples=360)
eval_clusterer = KMeans(n_clusters=100)
uu = UnNormalizeTransform()

def saver_func(dloader, c_labels, root_dir):
    Path(root_dir).mkdir(parents=False, exist_ok=False)

    n_img = 0
    for data in dloader:
        image_tensor = data[learner_1.dataset.getkeys.image]
        for image in image_tensor:
            img_ = uu(image)
            the_cluster = c_labels[n_img]

            subfolder = '{}/cluster_{}'.format(root_dir, the_cluster)
            Path(subfolder).mkdir(parents=False, exist_ok=True)
            save_image(img_, '{}/{}.png'.format(subfolder, n_img))
            n_img += 1

def train_simple():
    learner_1.load_model('ae_learner_2_bigger')
    learner_1.train(4)
    learner_1.save_model('la_simple')

def eval_simple():
    learner_1.load_model('la_simple')
    cluster_labels = learner_1.eval(clusterer=eval_clusterer.fit_predict)
    saver_func(learner_1.dataloader, cluster_labels, './cluster_imgs')

if __name__ == '__main__':
    #train_simple()
    eval_simple()