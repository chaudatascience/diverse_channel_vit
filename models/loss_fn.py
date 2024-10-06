from torch import nn, Tensor
import torch.nn.functional as F
import torch
from utils import pairwise_distance_v2


def proxy_loss(proxies, img_emb, gt_imgs, scale: float | nn.Parameter) -> Tensor:
    """
    proxies: (num_classes, dim)
    img_emb: (num_imgs, dim)
    gt_imgs: (num_imgs)
    """
    proxies_emb = scale * F.normalize(proxies, p=2, dim=-1)
    img_emb = scale * F.normalize(img_emb, p=2, dim=-1)

    img_dist = pairwise_distance_v2(proxies=proxies_emb, x=img_emb, squared=True)
    img_dist = img_dist * -1.0

    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    img_loss = cross_entropy(img_dist, gt_imgs)
    return img_loss


def ortho_proj_loss_fn_v2(features, labels, gamma_s, gamma_d, reverse_pos_pairs: bool, use_square: bool):
    """
    features: b, num_tokens, d
    labels: num_tokens
    gamma_s: default 1.0
    gamma_d: default 0.5
    reverse_pos_pairs: default False. If true, we want each token to be orthogonal to all other tokens, regarless of their channels.
    """
    device = features.device
    #  features are normalized
    features = F.normalize(features, p=2, dim=-1)

    labels = labels[None, :, None]  # extend dims

    mask = torch.eq(labels, labels.transpose(-2, -1)).bool().to(device)
    eye = torch.eye(mask.shape[-2], mask.shape[-1]).bool().to(device).unsqueeze(0)

    mask_pos = mask.masked_fill(eye, 0).float()
    mask_neg = (~mask).float()
    dot_prod = torch.matmul(features, features.transpose(-2, -1))

    mask_pos_sum = mask_pos.sum(dim=(-2, -1)) + 1e-6
    mask_neg_sum = mask_neg.sum(dim=(-2, -1)) + 1e-6

    pos_pairs_mean = (mask_pos * dot_prod).sum(dim=(-2, -1)) / mask_pos_sum
    neg_pairs_mean = (mask_neg * dot_prod).sum(dim=(-2, -1)) / mask_neg_sum

    if use_square:
        neg_pairs_mean = neg_pairs_mean**2

    if reverse_pos_pairs:
        if use_square:
            pos_pairs_mean = pos_pairs_mean**2
        loss = gamma_s * pos_pairs_mean + gamma_d * neg_pairs_mean
    else:
        loss = gamma_s * (1.0 - pos_pairs_mean) + gamma_d * neg_pairs_mean
    return loss.mean()
