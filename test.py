from torch import BoolTensor, IntTensor, Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Preds should be a list of elements, where each element is a dict
# containing 3 keys: boxes, scores, labels
mask_pred = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
preds = [
    {
        "scores": Tensor([0.536]),
        # The labels keyword should contain an [N,] tensor
        # with integers of the predicted classes
        "labels": IntTensor([0]),
        # The masks keyword should contain an [N,H,W] tensor,
        # where H and W are the image height and width, respectively,
        # with boolean masks. This is only required when iou_type is `segm`.
        "masks": BoolTensor([mask_pred]),
    }
]

# Target should be a list of elements, where each element is a dict
# containing 2 keys: boxes and labels (and masks, if iou_type is `segm`).
# Each keyword should be formatted similar to the preds argument.
# The number of elements in preds and target need to match
mask_tgt = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
]
target = [
    {

        "labels": IntTensor([0]),
        "masks": BoolTensor([mask_tgt]),
    }
]

if __name__ == "__main__":
    # Initialize metric
    metric = MeanAveragePrecision(iou_type="segm")

    # Update metric with predictions and respective ground truth
    metric.update(preds, target)

    # Compute the results
    result = metric.compute()
    print(result)