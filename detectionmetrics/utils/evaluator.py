from detectionmetrics.utils.object_detection_metrics import compute_detection_metrics

class Evaluator:
    def __init__(self, model, dataset, iou_threshold=0.3):
        self.model = model
        self.dataset = dataset
        self.iou_threshold = iou_threshold

    def evaluate(self):
        total_tp, total_fp, total_fn = 0, 0, 0

        for i in range(len(self.dataset)):
            image, ground_truth = self.dataset[i]
            prediction = self.model.predict(image)

            print(f"\n=== Sample {i} ===")
            print("Predicted:", prediction)
            print("Ground Truth:", ground_truth)

            if not prediction or not ground_truth:
                continue

            # Convert prediction to expected format
            pred = [
                {'box': box, 'label': label, 'score': score}
                for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores'])
            ]

            # Convert ground truth to expected format
            gt = [
                {'box': box, 'label': label}
                for box, label in zip(ground_truth['boxes'], ground_truth['labels'])
            ]

            # Compute metrics (IoU prints happen inside this)
            metrics = compute_detection_metrics(pred, gt, self.iou_threshold)
            tp = metrics["true_positives"]
            fp = metrics["false_positives"]
            fn = metrics["false_negatives"]

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
