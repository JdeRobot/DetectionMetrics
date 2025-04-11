import torch

def match_predictions_to_ground_truth(pred, gt, iou_threshold=0.5):
    matched_gt = set()
    tp = 0  # True Positives
    fp = 0  # False Positives

    for pred_item in pred:
        pred_box = pred_item["box"]
        pred_label = pred_item["label"]
        match_found = False

        for i, gt_item in enumerate(gt):
            gt_box = gt_item["box"]
            gt_label = gt_item["label"]

            if i in matched_gt:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou >= iou_threshold and pred_label == gt_label:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break

        if not match_found:
            fp += 1

    fn = len(gt) - len(matched_gt)  # False Negatives

    return tp, fp, fn

def compute_detection_metrics(pred, gt, iou_threshold=0.5):
    tp, fp, fn = match_predictions_to_ground_truth(pred, gt, iou_threshold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }
def compute_iou(boxA, boxB):
    """Computes IoU between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp = 0
    fp = 0
    matched_gt = set()

    for pred_box in pred_boxes:
        match_found = False
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            fp += 1
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall
