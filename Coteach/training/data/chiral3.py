from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
from .utils import noisify, flip_error, flip_error_test
import h5py


class CHIRAL(data.Dataset):
    """

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    data_file = 'split_Chiral_D_Large_TIFF_Cropped_front_only_combo_coteaching.h5'

    def __init__(self, root, train=True, transform=None, target_transform=None,noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset='chiral'



        if self.train:
            self.train_data = (h5py.File(os.path.join(self.root,self.data_file),'r')['train_images'][:,:,:,0]*255).astype('uint8')
            self.train_data = np.concatenate([self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data,self.train_data],axis=0)
            self.train_labels = h5py.File(os.path.join(self.root,self.data_file),'r')['train_labels']
            self.train_noisy_labels, self.actual_noise_rate = flip_error(self.train_labels,noise_rate)
            self.train_noisy_labels = np.concatenate([self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels,self.train_noisy_labels],axis=0)
            self.train_labels = np.concatenate([self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels,self.train_labels],axis=0)
            self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
            _train_labels=[i[0] for i in self.train_labels]
            self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
        else:
            self.test_data = (h5py.File(os.path.join(self.root,self.data_file),'r')['test_images'][:,:,:,0]*255).astype('uint8')
            self.test_labels = h5py.File(os.path.join(self.root,self.data_file),'r')['test_labels'][:,0]
            self.train_labels, self.actual_noise_rate = flip_error_test(self.test_labels,noise_rate)
            # self.test_labels=[i[0] for i in self.test_labels]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img, target = self.train_data[index], self.train_noisy_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
