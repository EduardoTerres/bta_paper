import torch


def state_eval_transform(obs, device):
    """Drop-in replacement for the image `eval_transforms` callable (which normalizes
    by /255 and assumes (C, H, W)/(B, C, H, W) shapes) for vector observations.
    """
    return torch.as_tensor(obs, device=device).float()
