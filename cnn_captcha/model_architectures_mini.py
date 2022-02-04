import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_shape, num_char, num_dim, coord=False):
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
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            print("The input shape is: ", self.input_shape)
            x = torch.zeros(self.input_shape)

            # First convolutional layer
            self.layer_dict['conv1'] = nn.Conv2d(in_channels=self.input_shape[1],
                                                 out_channels=8, kernel_size=5,
                                                 padding=2)
            out = self.layer_dict['conv1'](x)
            out = F.relu(out)
            print("The shape of feature maps after 1st convolutional layer: ", out.shape)
            # First max pooling layer
            self.layer_dict['max_pooling1'] = nn.MaxPool2d(kernel_size=2)
            out = self.layer_dict['max_pooling1'](out)
            print("The shape of feature maps after 1st max pooling layer: ", out.shape)

            # Second convolutional layer
            self.layer_dict['conv2'] = nn.Conv2d(in_channels=out.shape[1],
                                                 out_channels=12, kernel_size=5,
                                                 padding=2)
            out = self.layer_dict['conv2'](out)
            out = F.relu(out)
            print("The shape of feature maps after 2nd convolutional layer: ", out.shape)
            # Second max pooling layer
            self.layer_dict['max_pooling2'] = nn.MaxPool2d(kernel_size=2)
            out = self.layer_dict['max_pooling2'](out)
            print("The shape of feature maps after 2nd max pooling layer: ", out.shape)

            # Third convolutional layer
            self.layer_dict['conv3'] = nn.Conv2d(in_channels=out.shape[1],
                                                 out_channels=16, kernel_size=5,
                                                 padding=2)
            out = self.layer_dict['conv3'](out)
            out = F.relu(out)
            print("The shape of feature maps after 3rd convolutional layer: ", out.shape)
            # Third max pooling layer
            self.layer_dict['max_pooling3'] = nn.MaxPool2d(kernel_size=2)
            out = self.layer_dict['max_pooling3'](out)
            print("The shape of feature maps after 3rd max pooling layer: ", out.shape)

            out = out.view(self.input_shape[0], -1)
            self.layer_dict['fcc1'] = nn.Linear(in_features=out.shape[1],
                                                out_features=64)
            out = self.layer_dict['fcc1'](out)
            out = F.relu(out)

            self.layer_dict['fcc2'] = nn.Linear(in_features=64,
                                                out_features=self.num_char * self.num_dim)
            out = self.layer_dict['fcc2'](out)

            out = out.view(self.input_shape[0], self.num_char, self.num_dim)
            print("The output shape is: ", out.shape)

    def forward(self, x):
        b, _, _, _ = x.shape

        out = self.layer_dict['conv1'](x)
        out = F.relu(out)
        out = self.layer_dict['max_pooling1'](out)

        out = self.layer_dict['conv2'](out)
        out = F.relu(out)
        out = self.layer_dict['max_pooling2'](out)

        out = self.layer_dict['conv3'](out)
        out = F.relu(out)
        out = self.layer_dict['max_pooling3'](out)

        out = out.view(b, -1)
        out = self.layer_dict['fcc1'](out)
        out = F.relu(out)
        out = self.layer_dict['fcc2'](out)
        out = out.view(b, self.num_char, self.num_dim)

        return out