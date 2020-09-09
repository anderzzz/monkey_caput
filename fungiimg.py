'''Fungi Image Dataset class

The class presupposes that the fungi data is organized in a table with headers:

    Kingdom, Division, Subdivision, Class, Order, Family, Genus, Species, InstanceIndex, ImageName

Written By: Anders Ohrn, September 2020

'''
import torch
import pandas as pd
import os
from skimage import io

from torch.utils.data import Dataset
from torchvision import transforms

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
    RAW_TABLE_ROWS = 15695

    def __init__(self, csv_file, root_dir, selector=None, iselector=None, transform=None, label_keys=None):

        self.img_toc = pd.read_csv(csv_file, index_col=(0,1,2,3,4,5,6,7,8))
        print (self.img_toc)
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
                subset_label = subset_label.astype({'ClassLabel': 'int32'})
                category_slices.append(subset_label)

        return category_slices

    @property
    def label_semantics(self):
        '''The dictionary that maps '''
        return dict([(count, label_select) for count, label_select in enumerate(self.label_keys)])

    @classmethod
    def raw_table_rows(cls):
        return cls.RAW_TABLE_ROWS


class StandardTransform(object):
    '''Standard Image Transforms, typically instantiated and provided to the DataSet class

    '''
    def __init__(self, min_dim=300, to_tensor=True,
                 normalize=True, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.transforms = []
        self.transforms.append(transforms.ToPILImage())
        self.transforms.append(transforms.Resize(min_dim))
        if to_tensor:
            self.transforms.append(transforms.ToTensor())
        if normalize:
            self.transforms.append(transforms.Normalize(norm_mean, norm_std))

        self.t_total = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_total(img)


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
    test_mask = random.randint(low=0, high=FungiImg.raw_table_rows(), size=200)
    train_mask = list(set(range(FungiImg.raw_table_rows())) - set(test_mask))
    fds = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi',
                   iselector=test_mask)
    print (fds.img_toc.shape)
    print (fds.img_toc)

test5()