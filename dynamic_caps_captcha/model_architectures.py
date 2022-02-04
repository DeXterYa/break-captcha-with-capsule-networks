import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
from utils import coord_addition, coord_addition_cap

def squash(s, dim=-1):
    """
    A non-linear squashing function to ensure that vector length is between 0 and 1. (Eq. (1) from the original paper)
    :param s: A weighted sum over all prediction vectors.
    :param dim: Dimension along which the length of a vector is calculated.
    :return: Squashed vector.
    """
    squared_length = torch.sum(s ** 2, dim=dim, keepdim=True)

    return squared_length / (1 + squared_length) * s / (torch.sqrt(squared_length) + 1e-8)  # avoid zero denominator


class PrimaryCapsules(nn.Module):
    def __init__(self, input_shape, out_channels, dim_caps=8, kernel_size=9, stride=2, padding=0, coord=False, attention=False):
        """
        Initialise a Primary Capsules layer.
        :param input_shape: The shape of the inputs going in to the network.
        :param out_channels: The number of output channels.
        :param dim_caps: The dimension of a capsule vector.
        :param kernel_size: The size of convolutional kernels
        :param stride: The amount of movement between kernel applications
        :param padding: The addition of pixels to the edge of the feature maps
        """
        super(PrimaryCapsules, self).__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.dim_caps = dim_caps
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.coord = coord
        self.attention = attention
        # initialize a module dict, which is effectively a dictionary that can collect layers
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Build neural networks and print out the shape of feature maps.
        """
        with torch.no_grad():
            print("Building a Primary Capsules layer with input shape: ", self.input_shape)
            x = torch.zeros(self.input_shape)
            # Compute the number of capsules in the Primary Capsules layer
            self.num_caps = int(self.out_channels / self.dim_caps)
            self.layer_dict['conv'] = nn.Conv2d(in_channels=self.input_shape[1],
                                                out_channels=self.out_channels,
                                                kernel_size=self.kernel_size,
                                                stride=self.stride,
                                                padding=self.padding)

            out = self.layer_dict['conv'](x)
            print("The shape of feature maps after convolution: ", out.shape)
            # Group dim_caps pixel values together across channels as elements of a capsule vector
            h = out.size(2)
            w = out.size(3)
            out = out.view(out.size(0), self.num_caps, self.dim_caps, out.size(2), out.size(3))
            # attention for the spatial dimension
            # if self.attention:
            #     mask = out.sum((2, 1)).view(out.size(0), -1) / (self.num_caps * self.dim_caps)
            #
            #     self.fc_1 = nn.Linear(mask.size(-1), int(mask.size(-1)/2))
            #     mask = F.relu(self.fc_1(mask))
            #     self.fc_2 = nn.Linear(mask.size(-1), h*w)
            #     mask = torch.sigmoid(self.fc_2(mask))
            #     mask = mask.view(out.size(0), 1, 1, h, w)
            #
            #     out = mask * out + 1e-6

            out = out.permute(0, 1, 3, 4, 2).contiguous()
            # attention for capsule type
            if self.attention:
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                mask = self.avg_pool(out).squeeze()
                self.fc_1 = nn.Linear(mask.size(-1), 16)
                mask = F.relu(self.fc_1(mask))
                self.fc_2 = nn.Linear(16, out.size(1))
                mask = torch.sigmoid(self.fc_2(mask))
                mask = mask.view(out.size(0), out.size(1), 1, 1, 1)
                out = mask * out

            # if self.coord:
            #     out = coord_addition_cap(out)
            # Sequentialise capsule vectors
            out = out.view(out.size(0), -1, out.size(-1))
            out = squash(out)
            print("The shape of the layer output: ", out.shape)

    def forward(self, x):
        out = self.layer_dict['conv'](x)
        h = out.size(2)
        w = out.size(3)
        out = out.view(out.size(0), self.num_caps, self.dim_caps, out.size(2), out.size(3))
        # attention for the spatial dimension
        # if self.attention:
        #     mask = out.sum((2, 1)).view(out.size(0), -1) / (self.num_caps * self.dim_caps)
        #     mask = F.relu(self.fc_1(mask))
        #     mask = torch.sigmoid(self.fc_2(mask))
        #     mask = mask.view(out.size(0), 1, 1, h, w)
        #
        #     out = mask * out + 1e-6

        out = out.permute(0, 1, 3, 4, 2).contiguous()
        _, _, h, w, dim = out.shape

        # attention for capsule type
        if self.attention:
            mask = out.sum((4, 3, 2)) / (h * w * dim)
            mask = F.relu(self.fc_1(mask))
            mask = torch.sigmoid(self.fc_2(mask))
            mask = mask.view(out.size(0), out.size(1), 1, 1, 1)
            out = mask * out + 1e-6

        # if self.coord:
        #     out = coord_addition_cap(out)
        out = out.view(out.size(0), -1, out.size(-1))

        return squash(out)


class RoutingCapsules(nn.Module):
    def __init__(self, input_shape, num_caps, dim_caps, num_iter, device: torch.device):
        """
        Initialise a Routing Capsule layer. We only have one such layer (DigitCaps) in the original paper.
        :param input_shape: The input shape of the Routing Capsules layer. (batch_size, num_caps_input, dim_caps_input)
        :param num_caps: The number of output capsules.
        :param dim_caps: The dimension of each output capsule.
        :param num_iter: The number of routing iterations.
        :param device: Store tensor variables in CPU or GPU.
        """
        super(RoutingCapsules, self).__init__()
        self.input_shape = input_shape
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_iter = num_iter
        self.device = device
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            print("Building a RoutingCapsules layer with input shape: ", self.input_shape)

            x = torch.zeros(self.input_shape)  # (batch_size, num_caps_input, dim_caps_input)
            x = x.unsqueeze(1).unsqueeze(4)    # (batch_size, 1, num_caps_input, dim_caps_input, 1)
            # Weight matrix (1, num_caps_output, num_caps_input, dim_caps_output, dim_caps_input)
            self.W = nn.Parameter(0.01 * torch.randn(1, self.num_caps, self.input_shape[1],
                                                     self.dim_caps, self.input_shape[2]))
            # Prediction vectors (batch_size, num_caps_output, num_caps_input, dim_caps_output, 1)
            u_hat = torch.matmul(self.W, x)
            u_hat = u_hat.squeeze(-1)
            print("The shape of prediction vectors: ", u_hat.shape)

            # One routing iteration
            print("Test one routing iteration")
            # Initial logits (batch_size, num_caps_output, num_caps_input, 1)
            b = torch.zeros(*u_hat.size()[:3], 1)
            # Coupling coefficient
            c = F.softmax(b, dim=1)
            # Weighted sum (batch_size, num_caps_output, dim_caps_output)
            s = (c * u_hat).sum(dim=2)
            # Capsule vector output
            v = squash(s)
            print("The shape of capsule ouput: ", v.shape)
            # Update b
            b += torch.matmul(u_hat, v.unsqueeze(-1))
            print("Routing iteration completed")

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, x)
        u_hat = u_hat.squeeze(-1)
        # We do not want to create computational graph during the iteration
        # with torch.no_grad():
        b = torch.zeros(*u_hat.size()[:3], 1).to(x.device)
        # Routing iterations except for the last one
        for r in range(self.num_iter - 1):
            c = F.softmax(b, dim=1)
            s = (c * u_hat).sum(dim=2)
            v = squash(s)
            b = b + torch.matmul(u_hat, v.unsqueeze(-1))

        # Connect the last iteration to computational graph
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        # v = squash(s)
        v = s

        return v

class DynamicCapsules(nn.Module):
    def __init__(self, input_shape, num_channels_primary, dim_caps_primary,
                 num_classes, dim_caps_output, num_iter, device: torch.device,
                 kernel_size=9, dropout=False, coord=False, attention=False, stride=1):
        """
        Capsule networks with dynamic routing

        :param input_shape: Input image shape (batch_size, # channels, h, w)
        :param num_channels_primary: The number of channels of feature maps going in and out of Primary Capsules layer
        :param dim_caps_primary: The dimension of each capsule in the Primary Capsule layer
        :param num_classes: The number of classes of the dataset
        :param dim_caps_output: The dimension of each capsule in the Digit Capsules layer
        :param num_iter: The number of routing iteration in the Digit Capsule layer
        :param device: CPU or GPU
        :param kernel_size: The size of the kernel in Conv1 and Primary Capsule layer
        """
        super(DynamicCapsules, self).__init__()
        self.input_shape = input_shape
        self.num_channels_primary = num_channels_primary
        self.dim_caps_primary = dim_caps_primary
        self.num_classes = num_classes
        self.dim_caps_output = dim_caps_output
        self.num_iter = num_iter
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout
        if dropout:
            self.Dropout = nn.Dropout(p=0.25)
        self.coord = coord
        self.attention = attention
        self.relu = nn.ReLU(inplace=True)
        self.layer_dict = nn.ModuleDict()
        self.build_module()

    def build_module(self):
        with torch.no_grad():
            print("Building Dynamic Capsule Networks with input shape: ", self.input_shape)
            x = torch.zeros(self.input_shape)

            self.layer_dict['conv1'] = nn.Conv2d(self.input_shape[1], self.num_channels_primary,
                                                 self.kernel_size, stride=self.stride, bias=True)

            out = self.layer_dict['conv1'](x)
            print("shape after conv1: ", out.shape)
            out = self.relu(out)

            # Add coordinates to the feature maps
            if self.coord:
                out = coord_addition(out)
                print('The shape of feature maps after adding coordinates: ', out.shape)

            self.layer_dict['primarycaps'] = PrimaryCapsules(out.shape, self.num_channels_primary,
                                                             self.dim_caps_primary, self.kernel_size,
                                                             coord=self.coord, attention=self.attention)
            out = self.layer_dict['primarycaps'](out)
            # Only output 4 capsules because there are only 4 digits in the image
            self.layer_dict['digitcaps'] = RoutingCapsules(out.shape, 4, self.dim_caps_output,
                                                           self.num_iter, self.device)
            out = self.layer_dict['digitcaps'](out)  # (batch_size, num_caps_output = 4, dim_caps_output = 16)
            print("The shape of Digit Capsules layer output is: ", out.shape)

            # preds = torch.norm(out, dim=-1)
            # find top 4 capsules representing 4 digits appearing in the image
            # _, topk_length_idx = preds.topk(k=4)
            # _, max_length_idx = preds.max(dim=1)
            # y = torch.eye(self.num_classes)
            # y = y.index_select(dim=0, index=topk_length_idx).unsqueeze(2)
            # generate a mask
            # y = torch.zeros(self.num_classes)
            # y = y.index_fill_(0, topk_length_idx, 1)
            # y = y.unsqueeze(0).unsqueeze(2)

            # self.layer_dict['decoder'] = nn.Sequential(
            #         nn.Linear(self.dim_caps_output * 4, 512),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(512, 1024),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(1024, int(prod(self.input_shape[1:])) ),
            #         nn.Sigmoid()
            #     )
            # reconstructions = self.layer_dict['decoder'](out.view(out.size(0), -1))
            # reconstructions = reconstructions.view(self.input_shape)

            self.layer_dict['fcc'] = nn.Linear(16, 10)
            out = self.layer_dict['fcc'](out) # (batch_size, num_caps_output = 4, num_classes = 10)
            # print("The shape of outputs: ", out.shape,
            #       "The shape of reconstructed images: ", reconstructions.shape)

    def forward(self, x):
        out = self.layer_dict['conv1'](x)
        out = self.relu(out)
        if self.coord:
            out = coord_addition(out)
        out = self.layer_dict['primarycaps'](out)


        if self.dropout:
            dropout_mask = torch.ones(out.size()[:2]).to(out.device)
            dropout_mask = self.Dropout(dropout_mask)
            dropout_mask = dropout_mask.unsqueeze(-1)

            out = out * dropout_mask

        out = self.layer_dict['digitcaps'](out) # (batch_size, num_caps_output = 10, dim_caps_output = 16)
        # Compute the length of each capsule
        # preds = torch.norm(out, dim=-1) # (batch_size, num_caps_output = 10)
        # _, topk_length_idx = preds.topk(k=4)
        # Create a mask to zero out capsule vectors from the wrong digits
        # y = torch.zeros(self.num_classes)
        # y = y.index_fill_(0, topk_length_idx, 1)
        # y = y.unsqueeze(0).unsqueeze(2)

        # reconstructions = self.layer_dict['decoder'](out.view(out.size(0), -1))
        # reconstructions = reconstructions.view(x.shape)
        # out = self.layer_dict['fcc'](out)
        out = self.layer_dict['fcc'](out)
        # return out, reconstructions
        return out
