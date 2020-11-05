'''Fungi Image Dataset classes and factory methods for their creation

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

import img_transforms

class RawData(Enum):
    '''Number of rows in the image raw data'''
    N_ROWS = 15695
    '''Name of headers in raw data file'''
    HEADERS = ['Kingdom', 'Division', 'Subdivision', 'Class', 'Order', 'Family', 'Genus', 'Species', 'InstanceIndex', 'ImageName']
    '''Fungi level specification names'''
    LEVELS = HEADERS[:-2]

@dataclass
class DataGetKeys:
    '''Shared keys for the return of the datasets __getitem__ dictionary'''
    image: str = 'image'
    label: str = 'label'
    idx: str = 'idx'

#
# Various Fungi Datasets. These are accessed in the Learner via the factory function, `factory`, see below
#
class FungiFullBasicData(Dataset):
    '''Fungi Dataset. Properties: full image, basic transformation of image channels, no appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None, min_dim=224, square=True):
        super(FungiFullBasicData, self).__init__()

        self._core = _FungiDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.label
        del self.returnkey.idx
        self._transform = img_transforms.StandardTransform(min_dim=min_dim, square=square)

    def __len__(self):
        return self._core.__len__()

    def __getitem__(self, idx):
        image = self._transform(self._core[idx]['image'])
        return {self.returnkey.image : image}


class FungiFullBasicLabelledData(Dataset):
    '''Fungi Dataset. Properties: full image, basic transformation of image channels, label appended to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        min_dim (int, optional): min_dim (int): Length of shortest dimension of transformed output image
        square (bool, optional): If True, the source image (after resizing of shortest dimension) is cropped at
            the centre such that output image is square. Defaults to False

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, label_keys, selector=None, iselector=None, min_dim=224, square=False):
        super(FungiFullBasicLabelledData, self).__init__()

        self._core = _FungiDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                    label_keys=label_keys)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.idx
        self._transform = img_transforms.StandardTransform(min_dim=min_dim, square=square)

    def __len__(self):
        return self._core.__len__()

    def __getitem__(self, idx):
        raw_out = self._core[idx]
        image = self._transform(raw_out['image'])
        label = raw_out['label']
        return {self.returnkey.image : image, self.returnkey.label : label}


class FungiFullAugLabelledData(Dataset):
    '''Fungi Dataset. Properties: full image, augmentation transformation of image channels, label appended to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, label_keys, aug_multiplicity, aug_label, min_dim=224, square=False,
                 selector=None, iselector=None):
        super(FungiFullAugLabelledData, self).__init__()

        self._core = _FungiDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                    label_keys=label_keys)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.idx

        self._transform = [img_transforms.StandardTransform(min_dim=min_dim, square=square)]

        self.aug_multiplicity = aug_multiplicity
        for k_aug_transform in range(self.aug_multiplicity):
            self._transform.append(img_transforms.DataAugmentTransform(augmentation_label=aug_label,
                                                                       min_dim=min_dim, square=square))

    def __len__(self):
        return self._core.__len__() * self.aug_multiplicity

    def __getitem__(self, idx):
        idx_fungi = int(np.floor(idx / (1 + self.aug_multiplicity)))
        idx_aug_transform = idx % (1 + self.aug_multiplicity)
        raw_out = self._core[idx_fungi]
        image = self._transform[idx_aug_transform](raw_out['image'])
        label = raw_out['label']
        return {self.returnkey.image : image, self.returnkey.label : label}

class FungiFullBasicIdxData(FungiFullBasicData):
    '''Fungi Dataset. Properties: full image, basic transformation of image channels, image index appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None):
        super(FungiFullBasicIdxData, self).__init__(csv_file=csv_file, img_root_dir=img_root_dir,
                                                    selector=selector, iselector=iselector,
                                                    square=True)

        self.returnkey = DataGetKeys()
        del self.returnkey.label

    def __getitem__(self, idx):
        ret_dict = super().__getitem__(idx)
        ret_dict[self.returnkey.idx] = idx
        return ret_dict


class FungiGridBasicData(Dataset):
    '''Fungi Dataset. Properties: grid image, basic transformation of image channels, no appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        img_input_dim (int): Length and height of square of source image to be sliced by grid. Defaults to 224.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`. Defaults to 6.
        crop_step_size (int): Number of pixels between grid lines. Defaults to 32.
        crop_dim (int): Length and height of grid squares. Defaults to 64.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):
        super(FungiGridBasicData, self).__init__()

        self._core = _FungiDataCore(csv_file, img_root_dir, selector=selector, iselector=iselector)
        self._core._set_level_attr(self)
        self.returnkey = DataGetKeys()
        del self.returnkey.label
        del self.returnkey.idx
        self._transform = img_transforms.OverlapGridTransform(img_input_dim=img_input_dim,
                                                             img_n_splits=img_n_splits,
                                                             crop_step_size=crop_step_size,
                                                             crop_dim=crop_dim)

    def __len__(self):
        return self._core.__len__() * self._transform.n_blocks

    def __getitem__(self, idx):
        idx_fungi = int(np.floor(idx / self._transform.n_blocks))
        idx_sub = idx % self._transform.n_blocks
        raw_out = self._core[idx_fungi]
        img_crops = self._transform(raw_out['image'])
        return {self.returnkey.image : img_crops[idx_sub]}


class FungiGridBasicIdxData(FungiGridBasicData):
    '''Fungi Dataset. Properties: grid image, basic transformation of image channels, image index appended data to __getitem__

    Args:
        csv_file (str): Path to CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        img_input_dim (int): Length and height of square of source image to be sliced by grid. Defaults to 224.
        img_n_splits (int): Number of slices per side, thus total number of slices for one source image
            will be `img_n_splits * img_n_splits`. Defaults to 6.
        crop_step_size (int): Number of pixels between grid lines. Defaults to 32.
        crop_dim (int): Length and height of grid squares. Defaults to 64.

    Attributes:
          returnkey : Keys to access values of return dictionary for __getitem__

    '''
    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, img_n_splits=6, crop_step_size=32, crop_dim=64):
        super(FungiGridBasicIdxData, self).__init__(csv_file, img_root_dir, selector=selector, iselector=iselector,
                                                 img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                                 crop_step_size=crop_step_size, crop_dim=crop_dim)

        self.returnkey = DataGetKeys()
        del self.returnkey.label

    def __getitem__(self, idx):
        ret_dict = super().__getitem__(idx)
        ret_dict[self.returnkey.idx] = idx
        return ret_dict


class _FungiDataCore(object):
    '''The core data class that contains all logic related to the raw data files and their construction.

    Args:
        csv_file (str): CSV file with table-of-contents of the fungi raw data
        img_root_dir (str): Path to the root directory of fungi images
        selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
            order to select a subset of images on basis of MultiIndex values. Defaults to None.
        iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
            method in order to select a subset of images. This is applied after any `selector` filtering.
        label_keys (iterable of str): Collection of strings pass to the Pandas `.query` method in order
            to define subsets of the data that should be assigned integer class labels. If None, the
            indexing of the class returns the image tensor object, if not None, the indexing of the class
            returns the image tensor object and the integer class label.

    '''
    def __init__(self, csv_file, img_root_dir, selector=None, iselector=None, label_keys=None):

        self.img_toc = pd.read_csv(csv_file, index_col=(0,1,2,3,4,5,6,7,8))
        self.img_root_dir = img_root_dir
        self.label_keys = label_keys

        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]

        if not iselector is None:
            self.img_toc = self.img_toc.iloc[iselector]

        if not label_keys is None:
            self.img_toc = pd.concat(self._assign_label(self.label_keys))

    def _set_level_attr(self, obj):
        '''Add attributes to input object that denotes how numerous different levels of fungi data are.

        Args:
            obj : Object to add attributes to. Typically the `self` of a Fungi Dataset

        '''
        for level in RawData.LEVELS.value:
            setattr(obj, 'n_{}'.format(level.lower()), self._n_x(level))
            setattr(obj, 'n_instances_{}'.format(level.lower()), self._n_instances_x(level))

    def __getitem__(self, idx):
        '''Retrieve raw data from disk

        Args:
            idx: index to retrieve

        Returns:
            raw_data (dict): Raw data retrieved, which is a dictionary with keys "image" and "label", the former
                with value the image raw data, as represented by `skimage.io.imread`, the latter with value the
                associated image label (if applicable).

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.img_toc.iloc[idx]
        img_name = row[0]

        rel_path = list(row.name)[1:-1]
        rel_path.append(img_name)
        img_name = os.path.join(self.img_root_dir, *tuple(rel_path))
        image = io.imread(img_name)

        if not self.label_keys is None:
            label = row[1]
        else:
            label = None

        return {'image' : image, 'label' : label}

    def __len__(self):
        return len(self.img_toc)

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


#
# Various Fungi Dataset builders, which instantiate a Fungi Dataset. The builders are used by the factory
# function `factory`, see below.
#

class FungiFullBasicDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, selector=None, iselector=None,
                 img_input_dim=224, square=False, **_ignored):
        self._instance = FungiFullBasicData(csv_file=csv_file, img_root_dir=img_root_dir,
                                            selector=selector, iselector=iselector, square=square,
                                            min_dim=img_input_dim)
        return self._instance

class FungiFullBasicLabelledDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, label_keys, selector=None, iselector=None,
                       min_dim=224, square=False, **_ignored):
        self._instance = FungiFullBasicLabelledData(csv_file=csv_file, img_root_dir=img_root_dir,
                                                    label_keys=label_keys,
                                                    selector=selector, iselector=iselector,
                                                    min_dim=min_dim, square=square)
        return self._instance

class FungiFullAugLabelledDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, label_keys, aug_multiplicity, aug_label,
                 min_dim=224, square=False, selector=None, iselector=None, **_ignored):
        self._instance = FungiFullAugLabelledData(csv_file=csv_file, img_root_dir=img_root_dir,
                                                  label_keys=label_keys,
                                                  min_dim=min_dim, square=square,
                                                  aug_multiplicity=aug_multiplicity,
                                                  aug_label=aug_label,
                                                  selector=selector, iselector=iselector)
        return self._instance

class FungiFullBasicIdxDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir, selector=None, iselector=None, **_ignored):
        self._instance = FungiFullBasicIdxData(csv_file=csv_file, img_root_dir=img_root_dir,
                                               selector=selector, iselector=iselector)
        return self._instance

class FungiGridBasicDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir,
                 img_input_dim, img_n_splits, crop_step_size, crop_dim,
                 selector=None, iselector=None, **_ignored):
        self._instance = FungiGridBasicData(csv_file=csv_file, img_root_dir=img_root_dir,
                                            selector=selector, iselector=iselector,
                                            img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                            crop_step_size=crop_step_size, crop_dim=crop_dim)
        return self._instance

class FungiGridBasicIdxDataBuilder(object):
    def __init__(self):
        self._instance = None

    def __call__(self, csv_file, img_root_dir,
                 img_input_dim, img_n_splits, crop_step_size, crop_dim,
                 selector=None, iselector=None, **_ignored):
        self._instance = FungiGridBasicIdxData(csv_file=csv_file, img_root_dir=img_root_dir,
                                               selector=selector, iselector=iselector,
                                               img_input_dim=img_input_dim, img_n_splits=img_n_splits,
                                               crop_step_size=crop_step_size, crop_dim=crop_dim)
        return self._instance

class FungiDataFactory(object):
    '''Interface to fungi data factories.

    Typical usage involves the invocation of the `create` method, which returns a specific Fungi dataset.

    '''
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        '''Register a builder

        Args:
            key (str): Key to the builder, which can be invoked by `create` method
            builder: A Fungi Data Builder instance

        '''
        self._builders[key] = builder

    @property
    def keys(self):
        return self._builders.keys()

    def create(self, key, csv_file, img_root_dir, selector=None, iselector=None, **kwargs):
        '''Method to create a fungi data set through a uniform interface

        Args:
            key (str): The name of the type of dataset to create. The available keys available in attribute `keys`
            csv_file (str): CSV file with table-of-contents of the fungi raw data
            img_root_dir (str): Path to the root directory of fungi images
            selector (optional): Pandas IndexSlice or callable that is passed to the Pandas `.loc` method in
                order to select a subset of images on basis of MultiIndex values. Defaults to None.
            iselector (optional): Colletion of integer indices or callable that is passed to the Pandas `.iloc`
                method in order to select a subset of images. This is applied after any `selector` filtering.
            **kwargs: Additional arguments to be passed to the specific dataset builder.
        '''
        try:
            builder = self._builders[key]
        except KeyError:
            raise ValueError('Unregistered data builder: {}'.format(key))
        return builder(csv_file=csv_file, img_root_dir=img_root_dir, selector=selector, iselector=iselector,
                       **kwargs)

# The available pre-registrered fungi data set factory method. It can be imported and the `create` method has a
# uniform interface for the creation of one of many possible variants of a fungi data set.
factory = FungiDataFactory()
factory.register_builder('full basic', FungiFullBasicDataBuilder())
factory.register_builder('full basic labelled', FungiFullBasicLabelledDataBuilder())
factory.register_builder('full aug labelled', FungiFullAugLabelledDataBuilder())
factory.register_builder('full basic idx', FungiFullBasicIdxDataBuilder())
factory.register_builder('grid basic', FungiGridBasicDataBuilder())
factory.register_builder('grid basic idx', FungiGridBasicIdxDataBuilder())
