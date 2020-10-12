'''Fungi Image Dataset class

The class presupposes that the fungi data is organized in a table with headers:

    Kingdom, Division, Subdivision, Class, Order, Family, Genus, Species, InstanceIndex, ImageName

Written By: Anders Ohrn, September 2020

'''
import torch
import pandas as pd
import os
from skimage import io

from enum import Enum

from torch.utils.data import Dataset
from torchvision import transforms

class RawData(Enum):
    '''Number of rows in the image raw data'''
    N_ROWS = 15695
    '''Number of rows in the image raw data'''
    HEADERS = ['Kingdom', 'Division', 'Subdivision', 'Class', 'Order', 'Family', 'Genus', 'Species', 'InstanceIndex', 'ImageName']

class FungiImg(Dataset):
    '''The Fungi Image Dataset Class

    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        selector (IndexSlice, optional): Pandas IndexSlice that can select a subset of images
            on basis of MultiIndex values
        transform (callable, optional): Optional transform to be applied
            on a sample.

    '''
    def __init__(self, csv_file, root_dir, selector=None, iselector=None, transform=None, label_keys=None):

        self.img_toc = pd.read_csv(csv_file, index_col=(0,1,2,3,4,5,6,7,8))
        self.root_dir = root_dir
        self.transform = transform

        # Discard data as if never present, like in creation of test and train data sets, either
        # by row index or by an IndexSlice on the semantics of the MultiIndex
        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]
        elif not iselector is None:
            self.img_toc = self.img_toc.iloc[iselector]

        # Assign labels to data. This does not control for disjoint definitions or completeness
        if not label_keys is None:
            self.label_keys = label_keys
        else:
            species = self.img_toc.index.unique(level='Species')
            self.label_keys = ['Species == "{}"'.format(sss) for sss in species]
        self.img_toc = pd.concat(self._assign_label(self.label_keys))

        self.n_species = self._n_x('Species')
        self.n_genus = self._n_x('Genus')
        self.n_family = self._n_x('Family')
        self.n_order = self._n_x('Order')
        self.n_instance_species = self._n_instances_x('Species')
        self.n_instance_genus = self._n_instances_x('Genus')
        self.n_instance_family = self._n_instances_x('Family')
        self.n_instance_order = self._n_instances_x('Order')

    def __len__(self):
        return len(self.img_toc)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_toc.iloc[idx]
        img_name = row[0]
        label = row[1]
        rel_path = list(row.name)[1:-1]
        rel_path.append(img_name)
        img_name = os.path.join(self.root_dir, *tuple(rel_path))
        image = io.imread(img_name)

        if not self.transform is None:
            image = self.transform(image)

        return image, label

    def _assign_label(self, l_keys):
        '''Assign label to data based on family, genus, species selections'''
        category_slices = []
        for label_int, query_label in enumerate(l_keys):
            subset_label = self.img_toc.query(query_label)

            if len(subset_label) > 0:
                subset_label.loc[:, 'ClassLabel'] = label_int
                subset_label = subset_label.astype({'ClassLabel': 'int64'})
                category_slices.append(subset_label)

        return category_slices

    def _n_x(self, x_label):
        '''Compute number of distinct types of fungi at a given step in the hierarchy'''
        return len(self.img_toc.groupby(x_label))

    def _n_instances_x(self, x_label):
        '''Compute number of images for each type of fungi at a given step in the hierarchy'''
        return self.img_toc.groupby(x_label).count()[RawData.HEADERS.value[-1]].to_dict()

    @property
    def label_semantics(self):
        '''The dictionary that maps '''
        return dict([(count, label_select) for count, label_select in enumerate(self.label_keys)])

class StandardTransform(object):
    '''Standard Image Transforms, typically instantiated and provided to the DataSet class

    '''
    def __init__(self, min_dim=300, to_tensor=True, square=False,
                 normalize=True, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.transforms = []
        self.transforms.append(transforms.ToPILImage())
        self.transforms.append(transforms.Resize(min_dim))
        if square:
            self.transforms.append(transforms.CenterCrop(min_dim))
        if to_tensor:
            self.transforms.append(transforms.ToTensor())
        if normalize:
            self.transforms.append(transforms.Normalize(norm_mean, norm_std))

        self.t_total = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_total(img)

class UnTransform(object):

    def __init__(self, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.transforms = []
        self.transforms.append(transforms.Normalize(mean=[-m / s for m, s in zip(norm_mean, norm_std)],
                                                    std=[1.0 / s for s in norm_std]))
        #self.transforms.append(transforms.ToPILImage())
        self.t_total = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_total(img)

class DataAugmentTransform(object):
    '''Random Image Transforms for the purpose of data augmentation

    '''
    def __init__(self, augmentation_label, min_dim=300, to_tensor=True,
                 normalize=True, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.basic_transform = StandardTransform(min_dim, to_tensor, normalize, norm_mean, norm_std)

        self.transforms = []
        self.transforms.append(transforms.ToPILImage())
        if augmentation_label == 'random_resized_crop':
            self.transforms.append(transforms.RandomResizedCrop((300, 450), scale=(0.67,1.0)))
        elif augmentation_label == 'random_rotation':
            self.transforms.append(transforms.RandomRotation(180.0))
        self.transforms.append(transforms.ToTensor())
        self.t_aug = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_aug(self.basic_transform(img))

def test1():
    fds = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi')
    xx, label = fds[1]
    print (fds.label_semantics)

def test2():
    tt = StandardTransform(300, to_tensor=True, normalize=False)
    fds = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi', transform=tt)
    xx, label = fds[1000]
    io.imsave('dummy.png', xx.permute(1,2,0))
    print (label)

def test3():
    fds = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
                   label_keys=('Genus == "Cantharellus"', 'Genus == "Amanita"'))
    print (fds.label_semantics)

def test4():
    print (FungiImg.raw_table_rows())

def test5():
    from numpy import random
    img_items = list(range(RawData.N_ROWS.value))
    random.shuffle(img_items)
    test_mask = img_items[:200]
    train_mask = img_items[200:]
    fds_test = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
                        iselector=test_mask)
    print (fds_test.img_toc.shape)
    fds_train = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
                         iselector=train_mask)
    print (fds_train.img_toc.shape)

