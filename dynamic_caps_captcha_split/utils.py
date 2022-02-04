import torch


def coord_addition(features):
    b, c, h, w = features.shape
    h_offset = (torch.arange(h, dtype=torch.float32).to(features.device) + 0.50) / h
    w_offset = (torch.arange(w, dtype=torch.float32).to(features.device) + 0.50) / w
    h_layer = h_offset.view(1, 1, h, 1).repeat(b, 1, 1, w)
    w_layer = w_offset.view(1, 1, 1, w).repeat(b, 1, h, 1)

    # concatenate coordinates to the original feature maps
    return torch.cat((features, h_layer, w_layer), dim=1)

def coord_addition_cap(capsules):
    b, num_caps, h, w, _ = capsules.shape
    h_offset = (torch.arange(h, dtype=torch.float32).to(capsules.device) + 0.50) / h
    w_offset = (torch.arange(w, dtype=torch.float32).to(capsules.device) + 0.50) / w
    h_layer = h_offset.view(1, 1, h, 1, 1).repeat(b, num_caps, 1, w, 1)
    w_layer = w_offset.view(1, 1, 1, w, 1).repeat(b, num_caps, h, 1, 1)

    # concatenate coordinates to the original feature maps
    return torch.cat((capsules, h_layer, w_layer), dim=-1)