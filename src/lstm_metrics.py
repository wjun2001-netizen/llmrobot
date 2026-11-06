"""Binary metrics, including exact rank-based ROC-AUC with averaged ties."""
import math


def binary_metrics(labels, probabilities, threshold=.5):
    if not labels or len(labels) != len(probabilities) or not 0 <= threshold <= 1:
        raise ValueError("Need nonempty aligned labels/probabilities and a valid threshold")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("Binary integer labels required")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Invalid probabilities")
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    positives, negatives = sum(labels), len(labels) - sum(labels)
    auc = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, labels))
        rank_sum, index = 0., 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][0] == ordered[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2
            rank_sum += average_rank * sum(label for _, label in ordered[index:end])
            index = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {"accuracy": (tp + tn) / len(labels), "precision": precision, "recall": recall, "f1_score": f1,
            "roc_auc": auc, "roc_auc_unavailable_reason": None if auc is not None else "validation_has_single_class",
            "threshold": threshold, "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "validation_class_counts": {"0": negatives, "1": positives}, "zero_division_convention": 0}
