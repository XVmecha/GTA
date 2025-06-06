import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def evaluate_next_step_anomaly_detection(results_folder, model_name, dataset_name, plot=True):
    """
    Evaluate anomaly detection performance for only the immediate next time step

    Parameters
    ----------
    results_folder : str
        Path to the folder containing the results
    model_name : str
        Name of the model
    dataset_name : str
        Name of the dataset
    plot : bool
        Whether to generate and display plots

    Returns
    ----------
    dict
        Dictionary containing evaluation metrics
    """
    # Load saved results
    pred = np.load(f"{results_folder}/pred.npy")
    true = np.load(f"{results_folder}/true.npy")
    label = np.load(f"{results_folder}/label.npy")

    print(f"Loaded arrays - pred: {pred.shape}, true: {true.shape}, label: {label.shape}")

    # Slice to use only the immediate next time step (first time step in the prediction window)
    pred_next = pred[:, 0, :]  # Shape: (n_samples, n_features)
    true_next = true[:, 0, :]  # Shape: (n_samples, n_features)
    label_next = label[:, 0]  # Shape: (n_samples,)

    print(f"Sliced arrays - pred_next: {pred_next.shape}, true_next: {true_next.shape}, label_next: {label_next.shape}")

    # Calculate reconstruction error (anomaly score) for each sample
    # Mean squared error across all features
    reconstruction_error = np.mean(np.square(pred_next - true_next), axis=1)

    print(f"Reconstruction error shape: {reconstruction_error.shape}")
    print(f"Label distribution - Anomalies: {np.sum(label_next)}, Normal: {len(label_next) - np.sum(label_next)}")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(label_next, reconstruction_error)
    roc_auc = auc(fpr, tpr)

    # Find the optimal threshold (closest point to top-left corner)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    optimal_threshold = thresholds[optimal_idx]

    # Calculate predictions at optimal threshold
    y_pred = (reconstruction_error >= optimal_threshold).astype(int)

    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(label_next, reconstruction_error)
    pr_auc = average_precision_score(label_next, reconstruction_error)

    # Print metrics
    print(f"\nAUROC Score: {roc_auc:.4f}")
    print(f"Average Precision Score: {pr_auc:.4f}")

    # Print classification metrics at optimal threshold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

    accuracy = accuracy_score(label_next, y_pred)
    f1 = f1_score(label_next, y_pred)

    print(f"\nOptimal Threshold: {optimal_threshold:.6f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(label_next, y_pred)
    print(cm)

    print("\nClassification Report:")
    print(classification_report(label_next, y_pred))

    if plot:
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name} on {dataset_name} (Next Step Only)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{results_folder}/ROC_Curve_{model_name}_{dataset_name}_NextStep.png")
        plt.show()

        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name} on {dataset_name} (Next Step Only)')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(f"{results_folder}/PR_Curve_{model_name}_{dataset_name}_NextStep.png")
        plt.show()

        # Plot histogram of reconstruction errors
        plt.figure(figsize=(12, 6))
        plt.hist(reconstruction_error[label_next == 0], bins=50, alpha=0.5, label='Normal', color='blue')
        plt.hist(reconstruction_error[label_next == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Threshold: {optimal_threshold:.6f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Count')
        plt.title(f'Distribution of Reconstruction Errors - {model_name} on {dataset_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{results_folder}/Error_Distribution_{model_name}_{dataset_name}_NextStep.png")
        plt.show()

    # Return metrics
    metrics = {
        'auroc': roc_auc,
        'auprc': pr_auc,
        'accuracy': accuracy,
        'f1_score': f1,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }

    return metrics

if __name__ == "__main__":
    results_folder = '/home/berentzenaej/baselines/GTA/results/gta_SWaT_ftM_sl60_ll30_pl30_nl3_dm128_nh8_el3_dl2_df128_atprob_ebfixed_swat_test_1'
    metrics = evaluate_next_step_anomaly_detection(results_folder, 'GTA', 'SWaT')