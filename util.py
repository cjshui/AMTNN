"""TODO: docstring
"""

import os
import sys
import shutil
import tarfile
import urllib.request
import pickle as pkl
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import  Image
import skimage
from skimage.transform import resize as skresize
from skimage.io import imread as skimread
from skimage.color import rgb2gray as sk_rgb2gray

import torch
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision
from torchvision import datasets, transforms


"""
UTILS FUNCTIONS
"""


def in_feature_size (ft_extrctor_p, img_size):
    w = img_size
    for lyr_name, lyr_func in ft_extrctor_p.items():
        for oprt in lyr_func.keys():
            prmtrs = lyr_func[oprt]

            if oprt == 'conv':
                num_filters = prmtrs[1]
                w = int ((w - prmtrs[2] + 2*prmtrs[4]) /prmtrs[3])+1

            elif oprt =='maxpool':
                w = int((w - prmtrs[0] + 2 * prmtrs[2]) / prmtrs[1])+1

    return w*w*num_filters


"""
UTILS CLASSES
"""


class feature_extractor():
    def __init__(self, ft_extrctr_p):
        self.ft_extrctr_p = ft_extrctr_p

    def construct(self):
        layers = []
        for lyr_name, lyr_func in self.ft_extrctr_p.items():
            for oprt in lyr_func.keys():
                prmtrs = lyr_func[oprt]
                if oprt == 'conv':
                    # in_channels, out_channels, kernel_size, stride = 1, padding = 0
                    layers += [
                        nn.Conv2d(in_channels=prmtrs[0], out_channels=prmtrs[1], kernel_size=prmtrs[2],
                                  stride=prmtrs[3],
                                  padding=prmtrs[4])]

                if oprt == 'relu':
                    layers += [(nn.ReLU(True))]
                if oprt=='elu':
                    layers += [(nn.ELU(True))]

                if oprt == 'maxpool':
                    layers += [nn.MaxPool2d(kernel_size=prmtrs[0], stride=prmtrs[1], padding=prmtrs[2])]
        return nn.Sequential(*layers)


class Local_Dataset_digit(Dataset):
    def __init__(self, data_name, set,  data_path, transform, num_samples=None):
        super(Local_Dataset_digit, self).__init__()
        self.data_path = data_path
        self.data_name = data_name
        self.set = set
        self.transform = transform
        self.num_samples = num_samples


        if self.data_name == 'usps':

            self.inputs , self.targets =  self._USPS()
            self.inputs, self.targets = self._select_data()


        elif self.data_name=='m_mnist' or self.data_name =='mnist' :
            if self.set =='train' or self.set=='validation':
                self.inputs, self.targets = torch.load(open('data/mnist/processed/training.pt','rb'))

            elif self.set == 'test':
                self.inputs, self.targets = torch.load(open('data/mnist/processed/test.pt','rb'))


            self.inputs, self.targets = self._select_data()

            if self.data_name== 'm_mnist':

                self.inputs , self.targets = self._create_m_mnist( self.inputs,self.targets)

        print(self.data_name+' size '+str(self.inputs.size())+str( self.targets.size())+str([torch.min(self.inputs), torch.max(self.inputs)]))


    def __getitem__(self, index):

        img = self.inputs[index]
        lbl = self.targets[index]
        # img from tensor converted to numpy for applying a transformation:
        img = Image.fromarray(img.numpy())
        # img = self._checklist(img)

        if self.transform is not None:
            # convert back to tensor will be done as transform
            img = self.transform(img)

        return img, lbl

    def __len__(self):

        return len(self.inputs)

    # def _checklist(self, img):
    #     if torch.max(img)>=255.0: self.inputs/=255.0
    #     # img =()
    #     if img.ndim==4 and img.shape[1] ==1: img = np.squeeze(img, axis=1)
    #     return  img

    def _select_data (self):
        try :
            inputs , targets = self.inputs.numpy(), self.targets.numpy()
        except AttributeError:
            inputs, targets = self.inputs, self.targets
            pass
        if len(
            inputs) < self.num_samples:
            print ("! requested number of samples {:d} exceed the available data {:d}!! The maximum number to request is {:d}".format(self.num_samples, len(inputs), len(inputs)))
            self.num_samples = len(inputs)
        s_inputs, s_targets = [],[]
        for i in range(10):
            indx = np.where((targets).astype('uint8')==i)[0]

            if self.set == 'validation':
                s_inputs.append(inputs[indx][-100:])
                s_targets.append(targets[indx][-100:])
            elif self.set=='train' or self.set=='test':

                s_inputs.append(inputs[indx][:int(self.num_samples/10)])
                s_targets.append(targets[indx][:int(self.num_samples/10)])

        s_inputs = np.concatenate(s_inputs, axis=0)
        s_targets = np.concatenate(s_targets, axis=0)
        s_inputs = torch.tensor(s_inputs)
        s_targets = torch.tensor(s_targets, dtype=torch.long)
        return s_inputs,s_targets

    def _create_m_mnist(self,imgs,lbls):
        imgs, lbls = imgs.numpy(), lbls.numpy()
        print ('----> m_mnist'+str(imgs.shape))
        assert  len(imgs) == len(lbls)

        def _compose_image(digit, background):
            """Difference-blend a digit and a random patch from a background image."""

            w, h, _ = background.shape
            dw, dh, _ = digit.shape
            x = np.random.randint(0, w - dw)
            y = np.random.randint(0, h - dh)

            bg = background[x:x + dw, y:y + dh]
            return np.abs(bg - digit).astype(np.uint8)

        def _mnist_to_img(x):
            """Binarize MNIST digit and convert to RGB."""
            x = (x > 0).astype(np.float32)
            d = x.reshape([28, 28, 1]) * 255
            return np.concatenate([d, d, d], 2)

        def _create_mnistm(X, background_data):
            """
            Give an array of MNIST digits, blend random background patches to
            build the MNIST-M dataset as described in
            http://jmlr.org/papers/volume17/15-239/15-239.pdf
            """
            rand = np.random.RandomState(42)
            X_ = np.zeros([X.shape[0], 28, 28, 3], np.uint8)
            for i in range(X.shape[0]):

                bg_img = rand.choice(background_data)
                while bg_img is None: bg_img = rand.choice(background_data)
                d = _mnist_to_img(X[i])
                d = _compose_image(d, bg_img)
                X_[i] = d

            return X_

        # # import pdb ;pdb.set_trace()
        # if  not os.path.isfile(self.data_path+'/mnist_m_data.pt'):

        BST_PATH = self.data_path+'/BSR_bsds500.tgz'

        if 'BSR_bsds500.tgz' not in os.listdir(self.data_path):
            urllib.request.urlretrieve('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz',self.data_path+'/BSR_bsds500.tgz')

        f = tarfile.open(BST_PATH)
        train_files = []
        for name in f.getnames():
            if self.set=='train' or self.set=='validation':
                the_set = 'train'
            else: the_set='test'
            if name.startswith('BSR/BSDS500/data/images/'+the_set+'/'):
                    train_files.append(name)


        background_data = []
        for name in train_files:
            try:
                fp = f.extractfile(name)
                bg_img = Image.open(fp)
                background_data.append(np.array(bg_img))
            except:
                continue

        # os.remove(self.data_path+'/BSR_bsds500.tgz')

        train = _create_mnistm(imgs, background_data)
        train = np.mean(train, axis=3)
        train = train.reshape(-1, 28, 28)
        train = train/255.0

        # if self.set=='train':train, lbls = self._select_data(train, lbls)
        train = torch.tensor(train)
        lbls = torch.tensor(lbls, dtype=torch.long)

        return train, lbls


    def _USPS (self):
        def resize_and_scale(img, size, scale):
            img = skresize(img, size)
            return 1 - (np.array(img, "float32") / scale)

        # if  os.path.isfile(self.data_path+'/USPS'+set+'.pt'):
        sz = (28, 28)
        imgs_usps = []
        lbl_usps = []
        if 'USPdata.zip' not in os.listdir(self.data_path): urllib.request.urlretrieve('https://github.com/darshanbagul/USPS_Digit_Classification/raw/master/USPSdata/USPSdata.zip', self.data_path+'/USPSdata.zip')
        zip_ref = zipfile.ZipFile(self.data_path + '/USPSdata.zip', 'r')
        zip_ref.extractall(self.data_path)
        zip_ref.close()
        if self.set =='train' or self.set=='validation':
            for i in range(10):
                label_data = self.data_path+'/Numerals/'+ str(i) + '/'
                img_list = os.listdir(label_data)
                for name in img_list:
                    if '.png' in name:
                        img = skimread(label_data + name)
                        img = sk_rgb2gray(img)
                        resized_img = resize_and_scale(img, sz, 255)
                        imgs_usps.append(resized_img.flatten())
                        lbl_usps.append(i)

        elif self.set =='test':
            test_path = self.data_path+'/Test/'
            strt = 1
            for lbl, cntr in enumerate(range(151,1651, 150)):

                    for i in range(strt, cntr):
                        i = format(i, '04d')
                        img = skimread(os.path.join(test_path, 'test_'+str(i)+'.png'))
                        img = sk_rgb2gray(img)
                        resized_img = resize_and_scale(img, sz, 255)
                        imgs_usps.append(resized_img.flatten())
                        lbl_usps.append(9-lbl)
                    strt= cntr

        # os.remove(self.data_path+'/USPSdata.zip')
        shutil.rmtree(self.data_path+'/Numerals')
        shutil.rmtree(self.data_path + '/Test')
        imgs_usps, lbl_usps = np.asarray(imgs_usps).reshape(-1,28,28), np.asarray(lbl_usps)
        lbl_usps = torch.tensor(lbl_usps,dtype= torch.long)
        imgs_usps = torch.tensor(imgs_usps)

        # torch.save((imgs_usps,lbl_usps), open(self.data_path+'/USPS'+set+'.pt','wb'))
        #
        # else:
        #     imgs_usps, lbl_usps = torch.load(open(self.data_path+'/USPS'+set+'.pt','rb'))

        return imgs_usps,lbl_usps




class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = - lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        # initialize with x
        return GradientReversalFunction.apply(x, self.lambda_)

class classifier(nn.Module):
    def __init__(self, classifier_p):
        super(classifier, self).__init__()
        layers = []
        for lyr_name, lyr_func in classifier_p.items():
            for oprt in lyr_func.keys():
                prmtrs = lyr_func[oprt]
                if oprt == 'fc':
                    layers += [nn.Linear(prmtrs[0], prmtrs[1])]
                if oprt =='act_fn':
                    if prmtrs =='relu': layers += [nn.ReLU(True)]

                    elif prmtrs =='elu': layers += [nn.ELU(True)]

                    elif prmtrs =='sigm': layers += [nn.Sigmoid()]

                    elif prmtrs == 'softmax': layers += [nn.Softmax(dim = 1)]

                if oprt == 'dropout':
                    layers += [nn.Dropout(p=prmtrs[0])]
                if oprt == 'gradient_revers':
                    layers += [GradientReversal()]
        self.hypothsis = nn.Sequential(*layers)

    def forward(self, x):

        x = self.hypothsis(x)
        return x


class Local_SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, num_smpl=None):
        super(Local_SVHN, self).__init__(
            root, split, transform, target_transform, download)
        self.data = self.data[:num_smpl]
        self.labels = self.labels[:num_smpl]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # Convert to grayscale
        img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target





