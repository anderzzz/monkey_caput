'''Fungi Image Dataset classes, including image transformations

Written By: Anders Ohrn, September 2020

'''
import pandas as pd
import numpy as np
import os
from skimage import io

from enum import Enum
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GridMakerError(Exception):
    pass

class RawData(Enum):
    '''Number of rows in the image raw data'''
    N_ROWS = 15695
    '''Number of rows in the image raw data'''
    HEADERS = ['Kingdom', 'Division', 'Subdivision', 'Class', 'Order', 'Family', 'Genus', 'Species', 'InstanceIndex', 'ImageName']

@dataclass
class DataGetKeys:
    '''Shared keys for the return of the datasets __getitem__ dictionary'''
    image: str = 'image'
    label: str = 'label'
    idx: str = 'idx'

class FungiImg(Dataset):
    '''The Fungi Image Dataset Class

    The class presupposes that the fungi data is organized in a CSV table with the headers:

        `Kingdom, Division, Subdivision, Class, Order, Family, Genus, Species, InstanceIndex, ImageName`

    An example instantiation that can be used for supervised training of binary classifier:
        `FungiImg(csv_file='toc.csv', root_dir='./fungi_imgs',
                  selector=pd.IndexSlice[:,:,:,:,:,['Cantharellaceae', 'Amanitaceae'],:,:,:],
                  transform=StandardTransform(244, to_tensor=True, normalize=True),
                  label_keys=('Family == "Cantharellaceae"', 'Family == "Amanitaceae"'))`

    An example instantiation that can be used to access the first 100 images:
        `FungiImg(csv_file='toc.csv', root_dir='./fungi_imgs',
                  iselector=list(range(100)),
                  transform=StandardTransform(244, to_tensor=True, normalize=True))`

    Args:
        csv_file (string): Path to the csv file with image meta data.
        root_dir (string): Directory with all the images organized in subfolders.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        transform (callable, optional): Optional image transform to be applied. Defaults to None.
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.
        index_return (bool, optional): If True, the `__getitem__` method returns in addition to other things,
            the index associated with its other returned objects. Defaults to False.

    Attributes:
        n_species (int): Number of fungi species in DataSet
        n_genus (int) : Number of fungi genus in DataSet
        n_family (int) : Number of fungi families in DataSet
        n_order (int) : Number of fungi orders in DataSet
        n_instance_species (dict) : Number of instances of each fungi species in DataSet
        n_instance_genus (dict) : Number of instances of each fungi genus in DataSet
        n_instance_family (dict) : Number of instances of each fungi family in DataSet
        n_instance_order (dict) : Number of instances of each fungi orders in DataSet

    '''
    def __init__(self, csv_file, root_dir, selector=None, iselector=None, transform=None,
                 label_keys=None, index_return=False):

        self.img_toc = pd.read_csv(csv_file, index_col=(0,1,2,3,4,5,6,7,8))
        self.root_dir = root_dir
        self.transform = transform
        self.label_keys = label_keys
        self.index_return = index_return

        self.getkeys = DataGetKeys()

        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]

        if not iselector is None:
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
        '''Get data of given index.

        Note that the method returns the image and/or the associated ground truth label depending on the `label_keys`
        argument during initialization. Optionally the indeces are returned as well, which means a DataLoader will
        return these as well.

        Returns:
            return_data (dict): Dictionary with return data. The content of the dictionary depends on the
                initialization. At least it contains the image tensor. Optionally image label and
            image (Tensor): an image or images in the dataset
            label: ground truth label or labels that specifies ground truth class of the image or images. This is
                only returned if `label_keys` is not `None` in initialization.
            idx (Tensor): one index or several indices of the images retrieved. This is only returned if `index_return`
                is `True` in initialization.

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

        return_ = {self.getkeys.image : image}
        if not self.label_keys is None:
            return_[self.getkeys.label] = row[1]
        if self.index_return:
            return_[self.getkeys.idx] = idx

        return return_

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
    '''Fungi image data set class, with each image a grid square slice from the source image

    The class presupposes that the fungi data is organized in a CSV table with the headers:

        `Kingdom, Division, Subdivision, Class, Order, Family, Genus, Species, InstanceIndex, ImageName`

    Args:
        csv_file (string): Path to the csv file with image meta data.
        root_dir (string): Directory with all the images organized in subfolders.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        index_return (bool, optional): If True, the `__getitem__` method returns in addition to other things,
            the index associated with its other returned objects. Defaults to False.
        img_input_dim (int): Length and height of square of source image to be sliced by grid.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`.
        crop_step_size (int): Number of pixels between grid lines
        crop_dim (int): Length and height of grid squares.

    Raises:
        GridMakerError: In case the grid cropping specifications are not adding up

    '''
    def __init__(self, csv_file, root_dir, selector=None, iselector=None, index_return=False,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):

        self.fungiimg = FungiImg(csv_file=csv_file, root_dir=root_dir,
                                 selector=selector, iselector=iselector, index_return=index_return,
                                 transform=None,
                                 label_keys=None)
        self.index_return = self.fungiimg.index_return
        self.getkeys = self.fungiimg.getkeys

        self.cropper = OverlapGridTransform(img_input_dim, img_n_splits, crop_step_size, crop_dim)

    def __len__(self):
        return self.cropper.n_blocks * self.fungiimg.__len__()

    def __getitem__(self, idx):
        '''Get data of given index

        Returns:
            image (Tensor): one or multiple image grid square units in the dataset

        '''
        idx_fungi = int(np.floor(idx / self.cropper.n_blocks))
        idx_sub = idx % self.cropper.n_blocks
        out_full_img = self.fungiimg.__getitem__(idx_fungi)

        return_ = {}
        if self.index_return:
            return_[self.getkeys.idx] = idx
        img_crops = self.cropper(out_full_img[self.getkeys.image])
        return_[self.getkeys.image] = img_crops[idx_sub]

        return return_


class ZScoreConsts(Enum):
    '''Mean value to use for standard Z-score normalization, taken from https://pytorch.org/docs/stable/torchvision/models.html'''
    Z_MEAN = [0.485, 0.456, 0.406]
    '''Standard deviation values to use for standard Z-score normalization, taken from https://pytorch.org/docs/stable/torchvision/models.html'''
    Z_STD = [0.229, 0.224, 0.225]


class StandardTransform(object):
    '''Standard Image Transforms for pre-processing source image

    Args:
        min_dim (int): Length of shortest dimension of transformed image
        to_tensor (bool): If True, the output will be a PyTorch tensor, else PIL Image
        square (bool): If True, the source image (after resizing of shortest dimension) is cropped at the centre
            such that output image is square
        normalize (bool): If True, Z-score normalization is applied
        norm_mean : mean value for normalization of the R,G,B channels
        norm_std : std value for normalization of the R,G,B channels

    '''
    def __init__(self, min_dim=300, to_tensor=True, square=False,
                 normalize=True, norm_mean=ZScoreConsts.Z_MEAN.value, norm_std=ZScoreConsts.Z_STD.value):

        ts = [transforms.ToPILImage(), transforms.Resize(min_dim)]
        if square:
            ts.append(transforms.CenterCrop(min_dim))
        if to_tensor:
            ts.append(transforms.ToTensor())
        if normalize:
            ts.append(transforms.Normalize(norm_mean, norm_std))

        self.transform_total = transforms.Compose(ts)

    def __call__(self, img):
        return self.transform_total(img)


class UnNormalizeTransform(object):
    '''Invert standard image normalization. Typically used in order to create image representation to be saved for
    visualization

    Args:
        norm_mean : mean value for normalization of the R,G,B channels
        norm_std : std value for normalization of the R,G,B channels

    '''
    def __init__(self, norm_mean=ZScoreConsts.Z_MEAN.value, norm_std=ZScoreConsts.Z_STD.value):
        self.transform_total = transforms.Normalize(mean=[-m / s for m, s in zip(norm_mean, norm_std)],
                                                    std=[1.0 / s for s in norm_std])

    def __call__(self, img):
        return self.transform_total(img)


class DataAugmentTransform(object):
    '''Random Image Transforms for the purpose of data augmentation

    This class is not fully general, and assumes the input images have width 50% greater than height, which
    is true for fungi image dataset. Reuse this class with caution.

    Args:
        augmentation_label (str): Short-hand label for the type of augmentation transform to perform
        min_dim (int): Length of shortest dimension of transformed image
        to_tensor (bool): If True, the output will be a PyTorch tensor, else PIL Image
        normalize (bool): If True, Z-score normalization is applied
        norm_mean : mean value for normalization of the R,G,B channels
        norm_std : std value for normalization of the R,G,B channels

    '''
    def __init__(self, augmentation_label, min_dim=300, to_tensor=True,
                 normalize=True, norm_mean=ZScoreConsts.Z_MEAN.value, norm_std=ZScoreConsts.Z_STD.value):

        ts = [transforms.ToPILImage(), transforms.Resize(min_dim)]
        if augmentation_label == 'random_resized_crop':
            ts.append(transforms.RandomResizedCrop((min_dim, int(min_dim * 1.5)), scale=(0.67,1.0)))
        elif augmentation_label == 'random_rotation':
            ts.append(transforms.RandomRotation(180.0))
        else:
            raise ValueError('Unknown augmentation label: {}'.format(augmentation_label))

        if to_tensor:
            ts.append(transforms.ToTensor())
        if normalize:
            ts.append(transforms.Normalize(norm_mean, norm_std))
        self.transform_total = transforms.Compose(ts)

    def __call__(self, img):
        return self.transform_total(img)

class OverlapGridTransform(object):
    '''Transformer of image to multiple image slices on a grid. The images slices can be overlapping.

    In order for the slicing of the image to add up the following equality must hold:
        `crop_dim + (img_n_splits - 1) * crop_step_size == img_input_dim`

    Args:
        img_input_dim (int): Length and height of square of source image to be sliced by grid. Defaults to 224.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`. Defaults to 6.
        crop_step_size (int): Number of pixels between grid lines. Defaults to 32.
        crop_dim (int): Length and height of grid squares. Defaults to 64.
        norm_mean : mean value for normalization of the R,G,B channels
        norm_std : std value for normalization of the R,G,B channels

    Raises:
        GridMakerError: In case the grid cropping specifications are not adding up

    '''
    def __init__(self, img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64,
                 norm_mean=ZScoreConsts.Z_MEAN.value, norm_std=ZScoreConsts.Z_STD.value):

        if not crop_dim + (img_n_splits - 1) * crop_step_size == img_input_dim:
            raise GridMakerError('Image grid crop not possible: crop_dim + (img_n_splits - 1) * crop_step_size != img_input_dim')

        # Transformations of the source image: To PIL Image -> Resize shortest dimension -> Crop square at centre
        pre_transforms = []
        pre_transforms.append(transforms.ToPILImage())
        pre_transforms.append(transforms.Resize(img_input_dim))
        pre_transforms.append(transforms.CenterCrop(img_input_dim))
        self.pre_transforms = transforms.Compose(pre_transforms)

        # Transformations of the sliced grid image: To Tensor -> Z-Score Normalize RGB Channels
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
        return [self.post_transforms(transforms.functional.crop(img_, **kwarg)) for kwarg in self.kwargs]
