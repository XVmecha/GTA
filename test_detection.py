import numpy as np
import json
import os
from sklearn.metrics import f1_score, recall_score, precision_score


def calculate_anomaly_scores(trues, preds):
    """
    Calculate anomaly scores based on the formula:
    yˆ(t) = ∑ᵢ₌₁ᴹ ||Y(t)ᵢ − Yˆ(t)ᵢ||₂²

    Parameters:
    -----------
    trues : numpy.ndarray
        Ground truth values with shape (N, dim)
    preds : numpy.ndarray
        Predicted values with shape (N, dim)

    Returns:
    --------
    numpy.ndarray
        Anomaly scores for each time point, shape (N,)
    """
    # Calculate the squared differences for each feature dimension
    squared_diff = np.square(trues - preds)

    # Sum across feature dimensions (axis=1) to get anomaly score at each time point
    anomaly_scores = np.sum(squared_diff, axis=1)

    return anomaly_scores


def find_optimal_threshold(anomaly_scores, true_labels, n_thresholds=100, save_path=None, experiment_name='experiment'):
    """
    Perform grid search to find optimal threshold for anomaly detection.

    Parameters:
    -----------
    anomaly_scores : numpy.ndarray
        Anomaly scores for each time point, shape (N,)
    true_labels : numpy.ndarray
        Ground truth labels (0 for normal, 1 for anomaly), shape (N,)
    n_thresholds : int
        Number of threshold values to try
    save_path : str
        Directory path to save results file
    experiment_name : str
        Name for the experiment (used in filename)

    Returns:
    --------
    dict
        Results containing best threshold, F1 score, recall, and precision
    """
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)

    thresholds = np.linspace(min_score, max_score, n_thresholds)

    best_f1 = 0
    best_recall = 0
    best_f1_threshold = None
    best_recall_threshold = None
    best_precision = 0
    best_f1_precision = 0
    best_recall_precision = 0

    results = {
        'thresholds': [],
        'f1_scores': [],
        'recall_scores': [],
        'precision_scores': []
    }

    for threshold in thresholds:
        # Apply threshold to get binary predictions
        predicted_labels = (anomaly_scores > threshold).astype(int)

        # Calculate metrics
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        # Store results
        results['thresholds'].append(float(threshold))
        results['f1_scores'].append(float(f1))
        results['recall_scores'].append(float(recall))
        results['precision_scores'].append(float(precision))

        # Track best F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = threshold
            best_f1_precision = precision

        # Track best recall
        if recall > best_recall:
            best_recall = recall
            best_recall_threshold = threshold
            best_recall_precision = precision

    # Create a dictionary with summary results
    summary = {
        'best_f1': float(best_f1),
        'best_f1_threshold': float(best_f1_threshold),
        'best_f1_precision': float(best_f1_precision),
        'best_recall': float(best_recall),
        'best_recall_threshold': float(best_recall_threshold),
        'best_recall_precision': float(best_recall_precision),
        'results': {
            'thresholds': results['thresholds'],
            'f1_scores': results['f1_scores'],
            'recall_scores': results['recall_scores'],
            'precision_scores': results['precision_scores']
        }
    }

    #save fp,fn for best recall threshold
    predicted_labels_best_recall = (anomaly_scores > best_recall_threshold).astype(int)
    fn_indices_best_recall = np.where((true_labels == 1) & (predicted_labels_best_recall == 0))[0]
    fp_indices_best_recall = np.where((true_labels == 0) & (predicted_labels_best_recall == 1))[0]

    predicted_labels_best_f1 = (anomaly_scores > best_f1_threshold).astype(int)
    fn_indices_best_f1 = np.where((true_labels == 1) & (predicted_labels_best_f1 == 0))[0]
    fp_indices_best_f1 = np.where((true_labels == 0) & (predicted_labels_best_f1== 1))[0]

    # Save results to file if save_path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        #save false negatives and false positive of best threshold
        with open(f'./results/{save_path}/{experiment_name}_fn_best_recall_timesteps.txt', 'w') as f:
            for idx in fn_indices_best_recall:
                f.write(f"{idx}\n")
        with open(f'./results/{save_path}/{experiment_name}fp_best_recall_timesteps.txt', 'w') as f:
            for idx in fp_indices_best_recall:
                f.write(f"{idx}\n")
        with open(f'./results/{save_path}/{experiment_name}_fn_best_f1_timesteps.txt', 'w') as f:
            for idx in fn_indices_best_f1:
                f.write(f"{idx}\n")
        with open(f'./results/{save_path}/{experiment_name}fp_best_f1_timesteps.txt', 'w') as f:
            for idx in fp_indices_best_f1:
                f.write(f"{idx}\n")
        # Save as JSON
        json_path = os.path.join(save_path, f"{experiment_name}_threshold_results.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)

        # Save a simple text summary
        txt_path = os.path.join(save_path, f"{experiment_name}_threshold_summary.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Best F1 Score: {best_f1:.4f}\n")
            f.write(f"Best F1 Threshold: {best_f1_threshold:.4f}\n")
            f.write(f"Precision at Best F1: {best_f1_precision:.4f}\n\n")
            f.write(f"Best Recall: {best_recall:.4f}\n")
            f.write(f"Best Recall Threshold: {best_recall_threshold:.4f}\n")
            f.write(f"Precision at Best Recall: {best_recall_precision:.4f}\n")

        print(f"Results saved to {json_path} and {txt_path}")

    return summary

if __name__ == "__main__":
    # Load your prediction and ground truth data
    preds = np.load('results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_new/pred.npy')  # Shape (N, dim)
    trues = np.load('results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_new/true.npy')  # Shape (N, dim)
    labels = np.load('results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_new/label.npy')  # Shape (N,)

    # Make sure preds and trues have the right shape
    if len(preds.shape) > 2:
        preds = preds[:, 0, :]
        trues = trues[:, 0, :]
        labels = labels[:, 0]

    # Calculate anomaly sscores
    anomaly_scores = calculate_anomaly_scores(trues, preds)

    # Find optimal thresholds and save results
    results = find_optimal_threshold(
        anomaly_scores,
        labels,
        n_thresholds=100,
        save_path='results/thresholds/',
        experiment_name='GTA_SWaT'
    )

    print(f"Best F1 score: {results['best_f1']:.4f} at threshold {results['best_f1_threshold']:.4f}")
    print(f"Best Recall: {results['best_recall']:.4f} at threshold {results['best_recall_threshold']:.4f}")