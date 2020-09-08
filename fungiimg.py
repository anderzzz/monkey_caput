'''Fungi Image Dataset class

'''
import torch
import pandas as pd
import os
from skimage import io

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
    def __init__(self, csv_file, root_dir, selector=None, transform=None):

        self.img_toc = pd.read_csv(csv_file, index_col=(0,1,2,3,4,5,6,7,8))
        self.root_dir = root_dir
        self.transform = transform

        if not selector is None:
            self.img_toc = self.img_toc.loc[selector]

    def __len__(self):
        return len(self.img_toc)

    def __getitem__(self, idx):

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

        return image


def test1():
    fds = FungiImg('../../Desktop/Fungi/toc_full.csv', '../../Desktop/Fungi')
    xx = fds[1]

test1()