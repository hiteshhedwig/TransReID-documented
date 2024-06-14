import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    # below is the way to find hard positive samples
    # the "hardest" positive sample is matching 548 sample with 548
    # >>> a = torch.tensor([548, 237, 381, 139])
    # >>> a.expand(4,4)
    # tensor([[548, 237, 381, 139],
    #         [548, 237, 381, 139],
    #         [548, 237, 381, 139],
    #         [548, 237, 381, 139]])
    # >>> a.expand(4,4).eq(a.expand(4,4).t())
    # tensor([[ True, False, False, False],
    #         [False,  True, False, False],
    #         [False, False,  True, False],
    #         [False, False, False,  True]])
    # 
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # identifying the hardest positive samples for each anchor
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]


    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

  # Hard Margin Loss:

  #     Use when strict separation between classes is essential, and the margin must be enforced clearly.
  #     Suitable for applications where strong class boundaries are required and slight violations are unacceptable.

  # Soft Margin Loss:

  #     Use when smooth gradients and stability are more critical, especially in complex models and deep neural networks.
  #     Suitable for scenarios where the data is noisy, and softer enforcement of margins can lead to better generalization.
    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            # The margin ranking loss encourages the distance between the anchor and the negative sample to be larger than the distance between the anchor and the positive sample by at least a margin.
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
            # Uses a hard margin
            # Ensures the negative distance is greater than the positive distance by at least the margin
        else:
            # The soft margin loss is a smooth approximation that penalizes the model if the positive distance is not less than the negative distance.
            self.ranking_loss = nn.SoftMarginLoss()
            # Encourages the negative distance to be greater than the positive distance using a smooth approximation.
            # No hard margin is used.

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        
        # torch.Size([16, 768])
        print("global_feat ", global_feat.size())
        # torch.Size([16, 16])
        print("labels ", labels)
        dist_mat = euclidean_dist(global_feat, global_feat)
        print("dist_mat ", dist_mat.size())
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        # torch.Size([16])
        print("dist_ap ", dist_ap.size())
        # torch.Size([16])
        print("dist_an ", dist_an.size())

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        # creates a new tensor y that is the same size as dist_an and fills it with ones
        # The tensor y is used as a target for the ranking_loss function, 
        # indicating that we want the positive distances to be smaller than the 
        # negative distances
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        print("y ", y.size())

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        print("loss ", loss)
        return loss, dist_ap, dist_an


