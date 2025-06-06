import numpy as np
import os
def calculate_anomaly_scores(folder_num):
    # Load prediction and ground truth data
    base_path = f'results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_{folder_num}'

    preds = np.load(os.path.join(base_path,'pred.npy'))
    trues= np.load(os.path.join(base_path,'true.npy'))

    # Check shape compatibility
    if preds.shape != trues.shape:
        raise ValueError(f"Shape mismatch: predictions {preds.shape} and ground truth {trues.shape}")
    
    # Calculate anomaly scores
    # For each timestamp, compute the squared difference summed across all features
    squared_errors = np.square(trues - preds)
    
    # Sum across the feature dimension (last dimension)
    # This gives us one score per timestamp
    anomaly_scores = np.sum(squared_errors, axis=-1)
    
    return anomaly_scores

def detect_anomalies(anomaly_scores, threshold=None, method='percentile', percentile=95):
    """
    Detect anomalies based on calculated anomaly scores.
    
    Parameters:
    -----------
    anomaly_scores : numpy.ndarray
        Array of anomaly scores
    threshold : float, optional
        Fixed threshold for anomaly detection. If None, threshold is determined by method.
    method : str, optional
        Method to determine threshold if not explicitly provided:
        - 'percentile': Use a percentile of the scores
        - 'mean_std': Use mean + n_std * standard deviation
    percentile : float, optional
        Percentile to use if method='percentile'
    n_std : float, optional
        Number of standard deviations to use if method='mean_std'
    
    Returns:
    --------
    numpy.ndarray
        Boolean array indicating anomalies (True) and normal points (False)
    float
        The threshold used for detection
    """
    if threshold is None:
        if method == 'percentile':
            threshold = np.percentile(anomaly_scores, percentile)
        elif method == 'mean_std':
            mean = np.mean(anomaly_scores)
            std = np.std(anomaly_scores)
            threshold = mean + 3 * std  # Using 3 standard deviations as default
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Detect anomalies
    anomalies = anomaly_scores > threshold
    
    return anomalies, threshold

def evaluate_detection(predicted_anomalies, true_labels):
    """
    Evaluate anomaly detection performance.
    
    Parameters:
    -----------
    predicted_anomalies : numpy.ndarray
        Boolean array of predicted anomalies
    true_labels : numpy.ndarray
        Boolean or binary array of true anomaly labels
    
    Returns:
    --------
    dict
        Dictionary with precision, recall, and F1 score
    """
    true_labels = true_labels.astype(bool)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum(predicted_anomalies & true_labels)
    fp = np.sum(predicted_anomalies & ~true_labels)
    fn = np.sum(~predicted_anomalies & true_labels)
    
    # Calculate precision, recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def find_optimal_threshold(anomaly_scores, true_labels, n_thresholds=1000, optimize='f1'):
    """
    Find the optimal threshold that maximizes a selected metric.

    Parameters:
    -----------
    anomaly_scores : numpy.ndarray
        Array of anomaly scores
    true_labels : numpy.ndarray
        Boolean or binary array of true anomaly labels
    optimize : str, optional
        Metric to optimize ('f1', 'precision', or 'recall')

    Returns:
    --------
    float
        Optimal threshold
    dict
        Evaluation metrics at optimal threshold
    """
    true_labels = true_labels.astype(bool)
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    print(f"mean anomaly score for anomalies {np.mean(anomaly_scores[true_labels==1])}")
    print(f"mean anomaly score for non-anomalies {np.mean(anomaly_scores[true_labels==0])}")
    thresholds = np.linspace(min_score, max_score, n_thresholds)

    # Sort scores and try each as a threshold
    best_metric = 0
    best_eval = {}

    for threshold in thresholds:
        predicted_anomalies = anomaly_scores > threshold
        eval_metrics = evaluate_detection(predicted_anomalies, true_labels)

        current_metric = eval_metrics[optimize]
        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = threshold
            best_eval = eval_metrics

    return best_threshold, best_eval

def main(folder_num):
    # Calculate anomaly scores
    anomaly_scores = calculate_anomaly_scores(folder_num)
    base_path = f'results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_{folder_num}'
    true_labels = np.load(os.path.join(base_path,'label.npy'))

    print(f"Calculated {len(anomaly_scores)} anomaly scores")
    print(f"Min score: {np.min(anomaly_scores)}, Max score: {np.max(anomaly_scores)}")
    
    # If labels are available, find optimal threshold and evaluate

    # Find optimal threshold
    threshold_f1, eval_f1 = find_optimal_threshold(anomaly_scores, true_labels, optimize='f1')
    # threshold_recall, eval_recall = find_optimal_threshold(anomaly_scores, true_labels, optimize='recall')

    print("\nOptimal threshold for F1-score:")
    print(f"Threshold: {threshold_f1}")
    print(f"F1-score: {eval_f1['f1']:.4f}")
    print(f"Precision: {eval_f1['precision']:.4f}")
    print(f"Recall: {eval_f1['recall']:.4f}")

    # print("\nOptimal threshold for Recall:")
    # print(f"Threshold: {threshold_recall}")
    # print(f"F1-score: {eval_recall['f1']:.4f}")
    # print(f"Precision: {eval_recall['precision']:.4f}")
    # print(f"Recall: {eval_recall['recall']:.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate anomaly scores from prediction results")
    parser.add_argument("folder_num", type=str, help="folder number (number of run)")
    parser.add_argument("--label_file", type=str, help="Path to labels file (label.npy)", default=None)
    
    args = parser.parse_args()
    main(args.folder_num)