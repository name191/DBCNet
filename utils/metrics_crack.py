import torch
import numpy as np

def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, thresh=0.5):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    # assert y_true.ndim == 2
    # assert y_pred.ndim == 1 or y_pred.ndim == 2
    #
    # if y_pred.ndim == 2:
    #     y_pred = y_pred.argmax(dim=1)
    y_pred_pos = y_pred >= thresh
    y_pred_neg = y_pred <= thresh
    y_true_pos = y_true >= thresh
    y_true_neg = y_true <= thresh
    tp = (y_pred_pos & y_true_pos).sum().to(torch.float32)
    tn = (y_pred_neg & y_true_neg).sum().to(torch.float32)
    fp = (y_pred_pos & y_true_neg).sum().to(torch.float32)
    fn = (y_pred_neg & y_true_pos).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # f1.requires_grad = is_training
    return acc.data.cpu().numpy(), precision.data.cpu().numpy(), recall.data.cpu().numpy(), f1.data.cpu().numpy()


def iou_score(output, target, thresh=0.5):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.contiguous().view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.contiguous().view(-1).data.cpu().numpy()
    output_ = output > thresh
    target_ = target > thresh
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    # intersection = (output * target).sum()
    # union = (output.sum() + target.sum() - intersection)
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum() + target_.sum() +smooth)
    return iou, dice


def miou(output, target, thresh=0.5):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = output.contiguous().view(-1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.contiguous().view(-1).data.cpu().numpy()
    
    # intersection = (output * target).sum()
    # union = (output.sum() + target.sum() - intersection)
    # iou = (intersection + smooth) / (union + smooth)
    output_ = output > thresh
    target_ = target > thresh
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    # intersection_r = ((1 - output) * (1 - target)).sum()
    # union_r = ((1 - output).sum() + (1 - target).sum() - intersection_r)
    # iou_reverse = (intersection_r + smooth) / (union_r + smooth)
    output_reverse = (1 - output) > (1 - thresh)
    target_reverse = (1 - target) > (1 - thresh)
    iou_reverse = (smooth + (output_reverse & target_reverse).sum()) / ((output_reverse | target_reverse).sum() + smooth)
    return (iou + iou_reverse) / 2


def dice_coef(output, target, thresh=0.5):
    smooth = 1e-5

    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    # intersection = (output * target).sum()
    intersection = (output > thresh) & (target > thresh).sum()

    return (2. * intersection + smooth) / \
        ((output > thresh).sum() + (target > thresh).sum() + smooth)


def multiclass_mean_iou(output, target):
    B, C, H, W = output.shape
    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    smooth = 1e-5
    ious = []
    for i in range(C):
        class_target = np.zeros((B, H, W))
        class_target = target == i
        output_ = output[:, i, :, :] > 0.5
        intersection = (output_ & class_target).sum()
        union = (output_ | class_target).sum()
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
    return sum(ious) / len(ious)


if __name__ == '__main__':
    output = torch.randn([2, 5, 128, 128])
    target = torch.randint(0, 6, (2, 128, 128))
    mIoU = multiclass_mean_iou(output, target)
    print(mIoU)
