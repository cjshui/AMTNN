"""TODO: docstring
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

import util


def data_loading (img_size, num_tr_smpl,num_test_smpl, tsk_list ):
    our_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()])
    source_dataset, source_dataset_test, source_dataset_validation = [], [] , []

    for tsk in tsk_list:
        print ('LLLLLLLLLLLLLLL loading the current task '+tsk)
        if tsk =='mnist':
            source_dataset.append(util.Local_Dataset_digit(data_name='mnist', set='train', data_path='data/mnist', transform=our_transform,
                                 num_samples=num_tr_smpl))
            source_dataset_test.append(
                util.Local_Dataset_digit(data_name='mnist', set='test', data_path='data/mnist', transform=our_transform,
                                         num_samples=num_test_smpl))
            source_dataset_validation.append(
                util.Local_Dataset_digit(data_name='mnist', set='validation', data_path='data/mnist',
                                         transform=our_transform,
                                         num_samples=1000))
        if tsk == 'm_mnist':

            source_dataset.append(util.Local_Dataset_digit(data_name='m_mnist', set='train', data_path='data/mnist_m',
                                                           transform=our_transform,
                                                           num_samples=num_tr_smpl))
            source_dataset_test.append(
                util.Local_Dataset_digit(data_name='m_mnist', set='test', data_path='data/mnist_m', transform=our_transform,
                                         num_samples=num_test_smpl))
            source_dataset_validation.append(
                util.Local_Dataset_digit(data_name='m_mnist', set='validation', data_path='data/mnist_m',
                                         transform=our_transform,
                                         num_samples=1000))
        if tsk =='usps':
            source_dataset.append(util.Local_Dataset_digit(data_name='usps', set='train', data_path='data/USPSdata',
                                                           transform=our_transform,
                                                           num_samples=num_tr_smpl))
            source_dataset_test.append(
                util.Local_Dataset_digit(data_name='usps', set='test', data_path='data/USPSdata', transform=our_transform,
                                         num_samples=num_test_smpl))
            source_dataset_validation.append(
                util.Local_Dataset_digit(data_name='usps', set='validation', data_path='data/USPSdata',
                                         transform=our_transform,
                                         num_samples=1000))
        if tsk =='svhn':

            source_dataset.append(util.Local_SVHN(root='data/SVHN', split='train', transform=our_transform, download=True,
                            num_smpl=num_tr_smpl))
            source_dataset_test.append(
                util.Local_SVHN(root='data/SVHN', split='test', transform=our_transform, download=True,
                                num_smpl=num_test_smpl))
            source_dataset_validation.append(
                util.Local_SVHN(root='data/SVHN', split='extra', transform=our_transform, download=True,
                                    num_smpl=1000))

    train_loader = [DataLoader(source_dataset[t], batch_size=16, shuffle=True, num_workers=0)
                    for t in range(len(source_dataset_test))]
    test_loader = [
        DataLoader(source_dataset_test[t], batch_size=128, shuffle=False, num_workers=0) for t in
        range(len(source_dataset_test))]


    validation_loader = [
        DataLoader(source_dataset_validation[t], batch_size=128, shuffle=False, num_workers=0) for t in
        range(len(source_dataset_test))]

    return train_loader,test_loader,validation_loader
