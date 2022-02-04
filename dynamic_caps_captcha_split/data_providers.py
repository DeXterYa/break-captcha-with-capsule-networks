import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from skimage import io
import random
from PIL import Image


class CaptchaDigit(Dataset):
    def __init__(self, root_dir, set_name, transform, index, test_name='/test'):
        # self.root_dir = root_dir
        self.transform = transform
        self.set_name = set_name
        self.data = index

        if set_name == "train" or set_name == "val":
            self.data_dir = root_dir + '/train'
        elif set_name == "test":
            self.data_dir = root_dir + test_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # acquire the folder name containing the image and its mask
        folder_name = self.data[item]
        folder_dir = self.data_dir + '/' + folder_name
        # get the names of two image files inside the folder
        image_names = os.listdir(folder_dir)
        # the longer name is the mask the shorter name is the original image
        if len(image_names[0]) < len(image_names[1]):
            img_name = image_names[0]
            # img_mask_name = image_names[1]
        else:
            img_name = image_names[1]
            # img_mask_name = image_names[0]

        img = Image.open(folder_dir + '/' + img_name)
        # img_mask = Image.open(folder_dir + '/' + img_mask_name)
        digit_0 = torch.tensor([int(img_name[0])])
        digit_1 = torch.tensor([int(img_name[1])])
        digit_2 = torch.tensor([int(img_name[2])])
        digit_3 = torch.tensor([int(img_name[3])])

        if self.transform:
            img = self.transform(img)
            # img_mask = self.transform(img_mask)

        return img, torch.cat((digit_0, digit_1, digit_2, digit_3))


class CaptchaDigit90(Dataset):
    def __init__(self, root_dir, set_name, index=None, num_partition=6):
        super(CaptchaDigit90, self).__init__()
        self.images = []
        self.labels = []
        if set_name == "train" or set_name == "val":
            data_dir = root_dir + '/train'
            for p in range(1, num_partition + 1):
                partition_name = data_dir + '/images_' + str(p) + '.pt'
                data_partition = torch.load(partition_name)
                self.images += data_partition['images']
                self.labels += data_partition['labels']
            self.images = [self.images[i] for i in index]
            self.labels = [self.labels[i] for i in index]

        elif set_name == "test":
            data_dir = root_dir + '/test/images.pt'
            data = torch.load(data_dir)
            self.images = data['images']
            self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.images[item], self.labels[item]