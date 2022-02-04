import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, loss_lambda=0.5):
        """
        Initialise margin loss for dgit existence
        :param loss_lambda: A parameter used for down-weighting the loss for absent digit classes
        """
        super(MarginLoss, self).__init__()
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        """
        :param inputs: A batch of capsule output vector lengths from the Digit Capsules layer (batch_size, num_capsules)
        :param labels: A batch of corresponding labels (batch_size, num_capsules)
        :return:
        """
        L_k = labels * F.relu(self.m_plus - inputs) ** 2 + self.loss_lambda * (1 - labels) * F.relu(
            inputs - self.m_minus) ** 2
        # Sum across all digital capsules and batch instances
        return L_k.sum()



class CapsuleLoss(nn.Module):
    def __init__(self, loss_lambda=0.5, recon_loss_scale=5e-4):
        """
        Initialise Capsule Loss
        :param loss_lambda: A parameter used for down-weighting the loss for absent digit classes
        :param recon_loss_scale: The coefficient of reconstruction loss
        """
        super(CapsuleLoss, self).__init__()
        self.margin_loss = MarginLoss(loss_lambda)
        self.reconstruction_loss = nn.MSELoss()
        self.recon_loss_scale = recon_loss_scale

    def forward(self, inputs, labels, images, reconstructions):
        """
        Combine margin loss and reconstruction loss.
        :param inputs: A batch of capsule output vector lengths from the Digit Capsules layer (batch_size, num_capsules)
        :param labels: A batch of corresponding labels (batch_size, num_capsules)
        :param images: A batch of orginal input images (batch.size, # channels, h, w)
        :param reconstructions: A batch of images reconstructed by Dynamics Capsules decoder (batch.size, # channels, h, w)
        :return: Loss combining margin loss and reconstruction loss
        """
        m_loss = self.margin_loss(inputs, labels)
        r_loss = self.reconstruction_loss(reconstructions, images)
        return m_loss + self.recon_loss_scale * r_loss



