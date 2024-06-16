import numpy as np
from sklearn import metrics

def roc_auc_score_ignore_index(y_true, y_pred, index):
    y_true = np.delete(y_true, index, axis=1)
    y_pred = np.delete(y_pred, index, axis=1)
    return metrics.roc_auc_score(y_true, y_pred)

def calc_score(y_true, y_pred, indices_ignore):
    return roc_auc_score_ignore_index(np.array(y_true), np.array(y_pred), indices_ignore)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self, indices_ignore):
        self.reset()
        self.indices_ignore = indices_ignore

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_true_scored = []
        self.y_pred_scored = []

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        self.y_true.extend(y_true.tolist())
        self.y_pred.extend(y_pred.tolist())

    @property
    def avg(self):
        self.score = calc_score(np.array(self.y_true), np.array(self.y_pred), self.indices_ignore)

        return {
            "score" : self.score,
        }