import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Adapted from here: https://github.com/adambielski/siamese-triplet/blob/master/losses.py 
                       (OnlineContrastiveLoss)
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchors, negatives, positives):

        anchors = anchors/anchors.norm(dim=-1, keepdim=True)
        negatives = negatives/negatives.norm(dim=-1, keepdim=True)
        positives = positives/positives.norm(dim=-1, keepdim=True)

        positive_loss = (anchors - positives).pow(2).sum(1)
        negative_loss = F.relu(self.margin - (anchors - negatives).pow(2).sum(1).sqrt()).pow(2)

        loss = 0.5*torch.cat([positive_loss, negative_loss], dim=0)
        
        return loss.mean()


class ContrastiveCosine(nn.Module):
    """
    Contrastive loss using cosine similarity
    """
    def __init__(self, margin=0.2):
        super(ContrastiveCosine, self).__init__()
        self.margin = margin
        
    def forward(self, anchors, negatives, positives):

        anchors = anchors/anchors.norm(dim=-1, keepdim=True)
        negatives = negatives/negatives.norm(dim=-1, keepdim=True)
        positives = positives/positives.norm(dim=-1, keepdim=True)

        positive_loss = F.cosine_embedding_loss(anchors, positives, torch.ones(anchors.shape[0]).to(anchors.device))
        negative_loss = F.cosine_embedding_loss(anchors, negatives, -torch.ones(anchors.shape[0]).to(anchors.device), margin=self.margin)

        loss = 0.5*(positive_loss + negative_loss)
        
        return loss


class ContrastiveLoss_with_noise(nn.Module):
    """
    Contrastive loss
    Adapted from here: https://github.com/adambielski/siamese-triplet/blob/master/losses.py 
                       (OnlineContrastiveLoss)
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss_with_noise, self).__init__()
        self.margin = margin
        
    def forward(self, anchors, negatives, positives, neg_noise):

        anchors = anchors/anchors.norm(dim=-1, keepdim=True)
        negatives = negatives/negatives.norm(dim=-1, keepdim=True)
        positives = positives/positives.norm(dim=-1, keepdim=True)
        neg_noise=  neg_noise/neg_noise.norm(dim=-1, keepdim=True)

        positive_loss = (neg_noise - positives).pow(2).sum(1)
        negative_loss = F.relu(self.margin - (anchors - negatives).pow(2).sum(1).sqrt()).pow(2)

        loss = 0.5*torch.cat([positive_loss, negative_loss], dim=0)
        
        return loss.mean()