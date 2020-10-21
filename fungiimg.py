'''Fungi Image Dataset class

The class presupposes that the fungi data is organized in a table with headers:

    Kingdom, Division, Subdivision, Class, Order, Family, Genus, Species, InstanceIndex, ImageName

Written By: Anders Ohrn, September 2020

'''
import torch
import pandas as pd
import numpy as np
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
        self.label_keys = label_keys

        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]
        elif not iselector is None:
            self.img_toc = self.img_toc.iloc[iselector]

        # Extend the data table with labels if requested. This changes what the __getitem__ returns
        if not self.label_keys is None:
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
        '''Get data of given index

        Note that the method returns the image and/or the associated ground truth label depending on the `label_keys`
        argument during initialization.

        Returns:
            image (Tensor): an image or images in the dataset
            label: ground truth label or labels that specifies ground truth class of the image or images

        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_toc.iloc[idx]
        img_name = row[0]

        rel_path = list(row.name)[1:-1]
        rel_path.append(img_name)
        img_name = os.path.join(self.root_dir, *tuple(rel_path))
        image = io.imread(img_name)

        if not self.transform is None:
            image = self.transform(image)

        if not self.label_keys is None:
            label = row[1]
            return image, label
        else:
            return image

    def info_on_(self, idx):
        return self.img_toc.iloc[idx].name, self.img_toc.iloc[idx][0]

    def _assign_label(self, l_keys, int_start=0):
        '''Assign label to data based on family, genus, species selections

        The label keys are query strings for Pandas, as described here:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html

        Each query string define a class. The query string can refer to individual species, genus, family etc. or
        collections thereof. For example, the tuple `('Family == "Cantharellaceae"', 'Family == "Amanitaceae"')`
        defines two labels for all fungi in either of the two families.

        Args:
            l_keys (iterable): list of query strings for Pandas DataFrame, where each query string defines a class to be
                assigned a unique integer label.
            int_start (int, optional): the first integer class label. Defaults to 0.

        Returns:
            category_slices (list): List of DataFrames each corresponding to the categories. The list can be
                concatenated in order to form a single DataFrame

        '''
        category_slices = []
        for label_int, query_label in enumerate(l_keys):
            subset_label = self.img_toc.query(query_label)

            if len(subset_label) > 0:
                subset_label.loc[:, 'ClassLabel'] = label_int + int_start
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


class FungiImgGridCrop(Dataset):
    '''Fungi image data set class, with each image a grid unit from the source image

    Args:
        Bla bla

    '''
    def __init__(self, csv_file, root_dir, selector=None, iselector=None,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):

        self.fungiimg = FungiImg(csv_file=csv_file, root_dir=root_dir,
                                 selector=selector, iselector=iselector,
                                 transform=None,
                                 label_keys=None)
        self.cropper = OverlapGridTransform(img_input_dim, img_n_splits, crop_step_size, crop_dim)

    def __len__(self):
        return self.cropper.n_blocks * self.fungiimg.__len__()

    def __getitem__(self, idx):
        '''Get data of given index

        '''
        idx_fungi = int(np.floor(idx / self.cropper.n_blocks))
        idx_sub = idx % self.cropper.n_blocks
        img = self.fungiimg.__getitem__(idx_fungi)
        img_crops = self.cropper(img)

        return img_crops[idx_sub]

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
    '''Invert standard image normalization

    '''
    def __init__(self, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.transforms = []
        self.transforms.append(transforms.Normalize(mean=[-m / s for m, s in zip(norm_mean, norm_std)],
                                                    std=[1.0 / s for s in norm_std]))
        self.t_total = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_total(img)

class DataAugmentTransform(object):
    '''Random Image Transforms for the purpose of data augmentation

    This class is not fully general, and assumes the input images have width 50% greater than height. Reuse
    this class with caution.

    '''
    def __init__(self, augmentation_label, min_dim=300, to_tensor=True,
                 normalize=True, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        self.basic_transform = StandardTransform(min_dim, to_tensor, normalize, norm_mean, norm_std)

        self.transforms = []
        self.transforms.append(transforms.ToPILImage())
        if augmentation_label == 'random_resized_crop':
            self.transforms.append(transforms.RandomResizedCrop((min_dim, int(min_dim * 1.5)), scale=(0.67,1.0)))
        elif augmentation_label == 'random_rotation':
            self.transforms.append(transforms.RandomRotation(180.0))
        self.transforms.append(transforms.ToTensor())
        self.t_aug = transforms.Compose(self.transforms)

    def __call__(self, img):
        return self.t_aug(self.basic_transform(img))

class OverlapGridTransform(object):
    '''Transformer of image to multiple images on partially overlapping grids

    '''
    def __init__(self, img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):

        if not crop_dim + (img_n_splits - 1) * crop_step_size == img_input_dim:
            raise ValueError('Image grid crop not possible: crop_dim + (img_n_splits - 1) * crop_step_size != img_input_dim')

        pre_transforms = []
        pre_transforms.append(transforms.ToPILImage())
        pre_transforms.append(transforms.Resize(img_input_dim))
        pre_transforms.append(transforms.CenterCrop(img_input_dim))
        self.pre_transforms = transforms.Compose(pre_transforms)

        post_transforms = []
        post_transforms.append(transforms.ToTensor())
        post_transforms.append(transforms.Normalize(norm_mean, norm_std))
        self.post_transforms = transforms.Compose(post_transforms)

        self.kwargs = []
        h_indices = range(img_n_splits)
        w_indices = range(img_n_splits)
        for h in h_indices:
            for w in w_indices:
                self.kwargs.append({'top' : h * crop_step_size,
                                    'left' : w * crop_step_size,
                                    'height' : crop_dim,
                                    'width' : crop_dim})

        self.n_blocks = len(self.kwargs)

    def __call__(self, img):

        img_ = self.pre_transforms(img)
        ret_imgs = []
        for kwarg in self.kwargs:
            img_crop = self.post_transforms(transforms.functional.crop(img_, **kwarg))
            ret_imgs.append(img_crop)

        return ret_imgs