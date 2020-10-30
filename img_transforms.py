'''Image data transforms that can be applied after reading of raw data, before the application of the model

Written By: Anders Ohrn, October 2020

'''
from enum import Enum
from torchvision import transforms

class GridMakerError(Exception):
    pass

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
    def __init__(self, augmentation_label, min_dim=300, to_tensor=True, square=False,
                 normalize=True, norm_mean=ZScoreConsts.Z_MEAN.value, norm_std=ZScoreConsts.Z_STD.value):

        ts = [transforms.ToPILImage(), transforms.Resize(min_dim)]
        if square:
            ts.append(transforms.CenterCrop(min_dim))

        if augmentation_label == 'random_resized_crop':
            ts.append(transforms.RandomResizedCrop((min_dim, int(min_dim * 1.5)), scale=(0.67,1.0)))
        elif augmentation_label == 'random_rotation':
            ts.append(transforms.RandomRotation(180.0))
        elif augmentation_label == 'random_resized_crop_rotation':
            ts.append(transforms.RandomResizedCrop((min_dim, int(min_dim * 1.5)), scale=(0.67, 1.0)))
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
