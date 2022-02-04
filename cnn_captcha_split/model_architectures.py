import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import coord_addition
import random

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, num_char, num_dim, coord=False, random_order=True, random_rotation=True):
        """
        Convolutional neural network to recognise the captcha character sequence
        :param input_shape: The shape of the input image (b, c, h, w)
        :param num_char: The number of characters in the character sequence
        :param num_dim: The size of character vocabulary
        """
        super(ConvolutionalNetwork, self).__init__()
        self.input_shape = input_shape
        self.num_char = num_char
        self.num_dim = num_dim
        self.coord = coord
        self.random_order = random_order
        self.random_rotation = random_rotation
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            print("The input shape is: ", self.input_shape)
            x = torch.zeros(self.input_shape)

            # First convolutional layer
            self.layer_dict['conv1_1'] = nn.Conv2d(in_channels=self.input_shape[1],
                                                 out_channels=32, kernel_size=5,
                                                 padding=2)

            self.layer_dict['conv1_2'] = nn.Conv2d(in_channels=self.input_shape[1],
                                                   out_channels=32, kernel_size=5,
                                                   padding=2)
            out = self.layer_dict['conv1_1'](x)
            out = F.relu(out)
            print("The shape of feature maps after 1st convolutional layer: ", out.shape)
            # First max pooling layer
            self.layer_dict['max_pooling'] = nn.MaxPool2d(kernel_size=2)
            out = self.layer_dict['max_pooling'](out)
            print("The shape of feature maps after 1st max pooling layer: ", out.shape)

            # Second convolutional layer
            self.layer_dict['conv2_1'] = nn.Conv2d(in_channels=out.shape[1],
                                                 out_channels=48, kernel_size=5,
                                                 padding=2)
            self.layer_dict['conv2_2'] = nn.Conv2d(in_channels=out.shape[1],
                                                 out_channels=48, kernel_size=5,
                                                 padding=2)

            out = self.layer_dict['conv2_1'](out)
            out = F.relu(out)
            print("The shape of feature maps after 2nd convolutional layer: ", out.shape)
            # Second max pooling layer

            out = self.layer_dict['max_pooling'](out)
            print("The shape of feature maps after 2nd max pooling layer: ", out.shape)

            # Add coordinates
            # if self.coord:
            #     out = coord_addition(out)

            # Third convolutional layer
            self.layer_dict['conv3_1'] = nn.Conv2d(in_channels=out.shape[1],
                                                 out_channels=64, kernel_size=5,
                                                 padding=2)
            self.layer_dict['conv3_2'] = nn.Conv2d(in_channels=out.shape[1],
                                                   out_channels=64, kernel_size=5,
                                                   padding=2)
            out = self.layer_dict['conv3_1'](out)
            out = F.relu(out)
            print("The shape of feature maps after 3rd convolutional layer: ", out.shape)
            # Third max pooling layer

            out = self.layer_dict['max_pooling'](out)
            print("The shape of feature maps after 3rd max pooling layer: ", out.shape)

            out = out.view(self.input_shape[0], -1)
            out = torch.cat ((out, out), dim=1)
            self.layer_dict['fcc1'] = nn.Linear(in_features=out.shape[1],
                                                out_features=512)
            out = self.layer_dict['fcc1'](out)
            out = F.relu(out)

            self.layer_dict['fcc2'] = nn.Linear(in_features=512,
                                                out_features=self.num_char * self.num_dim)
            out = self.layer_dict['fcc2'](out)

            out = out.view(self.input_shape[0], self.num_char, self.num_dim)
            print("The output shape is: ", out.shape)

    def forward(self, x):
        b, _, _, _ = x.shape
        if self.random_order:

            if random.random() > 0.5:
                x_1 = x[:, :, :, :28]
                x_2 = x[:, :, :, 28:]
            else:
                x_1 = x[:, :, :, 28:]
                x_2 = x[:, :, :, :28]
        else:
            x_1 = x[:, :, :, :28]
            x_2 = x[:, :, :, 28:]

        if self.random_rotation:
            r1 = random.randint(0, 3)
            if r1 == 1:
                x_1 = torch.rot90(x_1, 1, [2, 3])
            if r1 == 2:
                x_1 = torch.rot90(x_1, 1, [2, 3])
                x_1 = torch.rot90(x_1, 1, [2, 3])
            if r1 == 3:
                x_1 = torch.rot90(x_1, -1, [2, 3])

            r2 = random.randint(0, 3)
            if r2 == 1:
                x_2 = torch.rot90(x_2, 1, [2, 3])
            if r2 == 2:
                x_2 = torch.rot90(x_2, 1, [2, 3])
                x_2 = torch.rot90(x_2, 1, [2, 3])
            if r2 == 3:
                x_2 = torch.rot90(x_2, -1, [2, 3])

        out_1 = self.layer_dict['conv1_1'](x_1)
        out_2 = self.layer_dict['conv1_2'](x_2)
        out_1 = F.relu(out_1)
        out_2 = F.relu(out_2)
        out_1 = self.layer_dict['max_pooling'](out_1)
        out_2 = self.layer_dict['max_pooling'](out_2)

        out_1 = self.layer_dict['conv2_1'](out_1)
        out_2 = self.layer_dict['conv2_2'](out_2)
        out_1 = F.relu(out_1)
        out_2 = F.relu(out_2)
        out_1 = self.layer_dict['max_pooling'](out_1)
        out_2 = self.layer_dict['max_pooling'](out_2)

        # Add coordinates
        # if self.coord:
        #     out = coord_addition(out)

        out_1 = self.layer_dict['conv3_1'](out_1)
        out_2 = self.layer_dict['conv3_2'](out_2)
        out_1 = F.relu(out_1)
        out_2 = F.relu(out_2)
        out_1 = self.layer_dict['max_pooling'](out_1)
        out_2 = self.layer_dict['max_pooling'](out_2)

        out = torch.cat((out_1.view(b, -1), out_2.view(b, -1)), dim=1)
        out = self.layer_dict['fcc1'](out)
        out = F.relu(out)
        out = self.layer_dict['fcc2'](out)
        out = out.view(b, self.num_char, self.num_dim)

        return out