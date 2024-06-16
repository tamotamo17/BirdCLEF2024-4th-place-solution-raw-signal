import torch
from torch import nn

class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth_eps=0, num_classes=182, use_rating=False, use_label_weight=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_rating = use_rating
        self.use_label_weight = use_label_weight
        self.smooth_eps = smooth_eps
        self.num_classes = num_classes

    def forward(self, preds, targets, rating):
        targets = targets.clamp(self.smooth_eps/self.num_classes, 1-self.smooth_eps)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss

        if self.use_rating:
            loss = loss.mean(1)
            loss = rating * loss
        elif self.use_label_weight:
            loss = loss.mean(0)
        loss = loss.mean()
        return loss

class BCELoss(nn.Module):
    def __init__(self, smooth_eps=0, num_classes=182, use_rating=False, use_label_weight=False):
        super().__init__()
        self.use_rating = use_rating
        self.use_label_weight = use_label_weight
        self.smooth_eps = smooth_eps
        self.num_classes = num_classes

    def forward(self, preds, targets, rating):
        targets = targets.clamp(self.smooth_eps/self.num_classes, 1-self.smooth_eps)
        loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        if self.use_rating:
            loss = loss.mean(1)
            loss = rating * loss
        elif self.use_label_weight:
            loss = loss.mean(0)
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None, use_rating=False, use_label_weight=False, smooth_eps=0):
        super().__init__()

        self.focal = BCEFocalLoss(use_rating=use_rating, use_label_weight=use_label_weight, smooth_eps=smooth_eps)

        self.weights = weights
    def forward(self, input, target, rating):
        input_ = input["logit"]
        target = target.float()
        rating = rating.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target, rating)
        aux_loss = self.focal(clipwise_output_with_max, target, rating)

        return self.weights[0] * loss + self.weights[1] * aux_loss

class BCE2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None, use_rating=False, use_label_weight=False, smooth_eps=0):
        super().__init__()

        self.bce = BCELoss(use_rating=use_rating, use_label_weight=use_label_weight, smooth_eps=smooth_eps)

        self.weights = weights
    def forward(self, input, target, rating):
        input_ = input["logit"]
        target = target.float()
        rating = rating.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.bce(input_, target, rating)
        aux_loss = self.bce(clipwise_output_with_max, target, rating)

        return self.weights[0] * loss + self.weights[1] * aux_loss
    
def cutmix_criterion(preds, new_targets, new_rating, smooth_eps=0):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    rating1, rating2, _ = new_rating[0], new_rating[1], new_rating[2]
    criterion = BCEFocalLoss(use_rating=False, use_label_weight=False, smooth_eps=smooth_eps)
    return lam * criterion(preds, targets1, rating1) + (1 - lam) * criterion(preds, targets2, rating2)

def mixup_criterion(preds, new_targets, new_rating, smooth_eps=0):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    rating1, rating2, _ = new_rating[0], new_rating[1], new_rating[2]
    criterion = BCEFocalLoss(use_rating=False, use_label_weight=False, smooth_eps=smooth_eps)
    return lam * criterion(preds, targets1, rating1) + (1 - lam) * criterion(preds, targets2, rating2)


def loss_fn(logits, targets, rating, smooth_eps=0):
    loss_fct = BCEFocalLoss(use_rating=False, use_label_weight=False, smooth_eps=smooth_eps)
    loss = loss_fct(logits, targets, rating)
    return loss