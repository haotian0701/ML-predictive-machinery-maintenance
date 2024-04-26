from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def get_train_val_test_arrays(
    X: Dict,
    y: Dict,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[
    List[Any],  # X_train
    List[Any],  # X_val
    List[Any],  # X_test
    List[Any],  # y_train
    List[Any],  # y_val
    List[Any],  # y_test
]:
    """
    Purpose: Randomly split the data into training, validation, and test sets.
    :param X: Dict representing the features.
    :param y: Dict representing the labels
    :param test_size: Proportion of the dataset to include in the test split. Default is 0.2.
    :param val_size: Proportion of the dataset to include in the validation split. Default is 0.2.
    :param random_state: Random state for reproducibility. Default is 42.
    :return: Tuple containing the training, validation, and testing sets.
    """
    # ensure the keys are sorted so they are aligned in both dictionaries
    keys = sorted(set(X.keys()) & set(y.keys()))
    X_values = [
        X[k] for k in keys
    ]
    y_values = [
        y[k] for k in keys
    ]

    # convert to numpy arrays
    X_values = np.array(X_values)
    y_values = np.array(y_values)

    # split the data into training and the rest
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_values,
        y_values,
        test_size = (test_size + val_size),
        random_state = random_state,
        stratify = y_values
    )

    # split the rest into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size = (val_size / (test_size + val_size)),
        random_state = random_state,
        stratify = y_temp
    )

    return (X_train, X_val, X_test, y_train, y_val, y_test)


def get_tensordataset(
    X: torch.Tensor,
    y: torch.Tensor
) -> TensorDataset:
    """
    Purpose: Create a TensorDataset from feature and target tensors, 
        suitable for DataLoader ingestion.
    :param X: torch.Tensor representing the input features for the model.
    :param y: torch.Tensor representing the target outputs for the model.
    :return: TensorDataset containing feature-target tensors.
    """
    dataset = TensorDataset(
        X, y
    )
    return dataset


def get_dataloader(
    dataset: TensorDataset,
    batch_size: int = 64,
    shuffle: bool = True
) -> DataLoader:
    """
    Purpose: Creates a DataLoader to iterate over a given TensorDataset with
        specified batch size and shuffling.
    :param dataset: TensorDataset representing dataset to load into DataLoader.
    :param batch_size: int representing the number of samples per batch.
    :param shuffle: bool indicating whether to shuffle the data.
    :return: DataLoader that yields batches of data from the dataset.
    """
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle
    )
    return dataloader


def train_validate_test(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    device: torch.device = torch.device('cpu'),
    epochs: int = 100
) -> Tuple[
    int,         # best_val_auc_idx
    np.ndarray,  # best_val_y_true_list
    np.ndarray,  # best_val_y_hat_list
    np.ndarray,  # associated_test_y_true_list
    np.ndarray,  # associated_test_y_hat_list
    Dict         # train_val_test_log
]:
    """
    Purpose: Trains a model on training data, validates it on validation data,
        and tests it on test data, capturing performance metrics and tracking
        the best model based on validation performance.
    :param model: nn.Module representing the neural network model to train.
    :param train_loader: DataLoader representing the batched dataset for training.
    :param val_loader: DataLoader representing the batched dataset for validation.
    :param test_loader: DataLoader representing the batched dataset for testing.
    :param loss_fn: nn.modules.loss._Loss representing the loss function used for training.
    :param optimizer: optim.Optimizer representing the optimizer used for training.
    :param device: torch.device indicating the device on which the model is trained.
    :param epochs: int representing the number of training epochs.
    :return: Tuple containing the index of the best validation epoch, arrays of true and
        predicted labels from the best validation epoch, true and predicted labels from
        associated testing, and a log dictionary of training/validation/testing statistics.
    """
    train_val_test_log = dict()
    train_val_test_log['test'] = dict()

    # move model to device
    model = model.to(device)

    # instantiate train loss
    best_total_train_loss = float('inf')

    # define the best epoch to be the epoch with the best validation AUC
    best_val_auc = float('-inf')
    best_val_auc_idx = None
    best_model_state = None
    best_optim_state = None

    # define lists to keep track of best validation epoch (best AUC) and associated test true/preds
    best_val_y_true_list = []
    best_val_y_hat_list = []
    associated_test_y_true_list = []
    associated_test_y_hat_list = []

    # loop through epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_idx = epoch + 1

        train_val_test_log[epoch_idx] = {
            'train': dict(),
            'val': dict(),
            'secondval': dict(),
        }

        # 1. BEGIN TRAINING

        # set model to train mode
        model.train()

        # define best train loss, val AUC for this epoch
        total_train_loss = 0
        total_val_loss = 0
        total_test_loss = 0
        val_auc = 0


        total_secondval_loss = 0

        # loop through data batches
        for X_train_batch, y_train_batch in train_loader:

            # [batch_size, sequence_length, features] -> [sequence_length, batch_size, features]
            X_train_batch = X_train_batch.transpose(0, 1)

            # move tensors to device
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)

            # reset gradients to 0 before starting backpropagation
            optimizer.zero_grad()

            # compute model predictions for the batch
            y_train_hat = model(
                X_train_batch
            )

            # compute the loss for train batch
            train_batch_loss = loss_fn(
                y_train_hat,
                y_train_batch
            )

            # backpropagate the batch loss and compute gradients
            train_batch_loss.backward()

            # perform single optimization step by updating model parameters based on gradients
            optimizer.step()

            # accumulate epoch's total train loss
            total_train_loss += train_batch_loss.item()

        train_val_test_log[epoch_idx]['train']['loss'] = total_train_loss

        # update best train loss if necessary
        if total_train_loss < best_total_train_loss:
            best_total_train_loss = total_train_loss

        # ------------------------------------------------------------------------------------------
        # 2. BEGIN VALIDATION

        # switch model to evaluation mode
        model.eval()

        # keep track of evaluation metrics
        val_y_true_list = []
        val_y_hat_list = []

        # keep track of secondval metrics
        secondval_y_true_list = []
        secondval_y_hat_list = []

        # disable computing gradients (we are just evaluating)
        with torch.no_grad():

            # loop through validation batches
            for X_val_batch, y_val_batch in val_loader:

                # [batch_size, sequence_length, features] -> [sequence_length, batch_size, features]
                X_val_batch = X_val_batch.transpose(0, 1)

                # move tensors to device
                X_val_batch = X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)

                # get predictions from trained model
                y_val_hat = model(
                    X_val_batch
                )

                # append y and y_hat to lists
                val_y_true_list.append(y_val_batch.cpu())
                val_y_hat_list.append(y_val_hat.cpu())

                # compute the loss for val batch
                val_batch_loss = loss_fn(
                    y_val_hat,
                    y_val_batch
                )

                # total_val_loss
                total_val_loss += val_batch_loss.item()

            # flatten lists of batch outputs into single array
            val_y_true_list = torch.cat(val_y_true_list).numpy()
            val_y_hat_list = torch.cat(val_y_hat_list).numpy()

            # compute validation AUC
            val_auc = roc_auc_score(
                y_true = val_y_true_list,
                y_score = val_y_hat_list
            )

            # check if we have found a new best validation AUC
            if best_val_auc < val_auc:
                best_val_auc = val_auc
                best_val_auc_idx = epoch_idx

                # keep track of best val true and preds
                best_val_y_true_list = val_y_true_list
                best_val_y_hat_list = val_y_hat_list

                # keep track of model and optimizer
                best_model_state = model.state_dict()
                best_optim_state = optimizer.state_dict()

            train_val_test_log[epoch_idx]['val']['auc'] = val_auc
            train_val_test_log[epoch_idx]['val']['loss'] = total_val_loss

            # -------------------------------------------------------------------------------------
            # ~ BEGIN TEST ("secondval")

            # loop through secondval batches
            for X_secondval_batch, y_secondval_batch in test_loader:

                # [batch_size, sequence_length, features] -> [sequence_length, batch_size, features]
                X_secondval_batch = X_secondval_batch.transpose(0, 1)

                # move tensors to device
                X_secondval_batch = X_secondval_batch.to(device)
                y_secondval_batch = y_secondval_batch.to(device)

                # get predictions from trained model
                y_secondval_hat = model(
                    X_secondval_batch
                )

                # append y and y_hat to lists
                secondval_y_true_list.append(y_secondval_batch.cpu())
                secondval_y_hat_list.append(y_secondval_hat.cpu())

                # compute the loss for val batch
                secondval_batch_loss = loss_fn(
                    y_secondval_hat,
                    y_secondval_batch
                )

                # total_val_loss
                total_secondval_loss += secondval_batch_loss.item()

            # flatten lists of batch outputs into single array
            secondval_y_true_list = torch.cat(secondval_y_true_list).numpy()
            secondval_y_hat_list = torch.cat(secondval_y_hat_list).numpy()

            secondval_auc = roc_auc_score(
                y_true = secondval_y_true_list,
                y_score = secondval_y_hat_list
            )
            train_val_test_log[epoch_idx]['secondval']['auc'] = secondval_auc
            train_val_test_log[epoch_idx]['secondval']['loss'] = total_secondval_loss

    # ------------------------------------------------------------------------------------------
    # 3. BEGIN TESTING

    # switch model to evaluation mode
    model.eval()

    # disable computing gradients (we are just evaluating)
    with torch.no_grad():

        # load in model and optimizer from best val AUC epoch
        model.load_state_dict(
            best_model_state
        )
        optimizer.load_state_dict(
            best_optim_state
        )

        # loop through test batches
        for X_test_batch, y_test_batch in test_loader:

            # [batch_size, sequence_length, features] -> [sequence_length, batch_size, features]
            X_test_batch = X_test_batch.transpose(0, 1)

            # move tensors to device
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            # get predictions from trained model
            y_test_hat = model(
                X_test_batch
            )

            # convert tensors to 2D list and move to cpu
            associated_test_y_true_list.append(y_test_batch.cpu())
            associated_test_y_hat_list.append(y_test_hat.cpu())

            # compute the loss for val batch
            test_batch_loss = loss_fn(
                y_test_hat,
                y_test_batch
            )

            # total_val_loss
            total_test_loss += test_batch_loss.item()

        # flatten lists of batch outputs into single array
        associated_test_y_true_list = torch.cat(associated_test_y_true_list).numpy()
        associated_test_y_hat_list = torch.cat(associated_test_y_hat_list).numpy()

        train_val_test_log['test']['auc'] = roc_auc_score(
            y_true = associated_test_y_true_list,
            y_score = associated_test_y_hat_list
        )
        train_val_test_log['test']['loss'] = total_test_loss

    return (
        best_val_auc_idx,
        best_val_y_true_list, best_val_y_hat_list,
        associated_test_y_true_list, associated_test_y_hat_list,
        train_val_test_log
    )


def display_val_test_confusion_matrices(
    best_val_data_confusion_matrix: np.ndarray,
    associated_test_data_confusion_matrix: np.ndarray,
    val_auc: float,
    test_auc: float,
    target_metric_substring: str
) -> None:
    """
    Purpose: Displays validation and test confusion matrices side by side,
        useful for comparing model performance.
    :param best_val_data_confusion_matrix: np.ndarray representing the 
        confusion matrix from validation data.
    :param associated_test_data_confusion_matrix: np.ndarray representing 
        the confusion matrix from test data.
    :param val_auc: float representing the AUC score from the validation dataset.
    :param test_auc: float representing the AUC score from the test dataset.
    :param target_metric_substring: str representing the specific metric condition
        that these matrices correspond to.
    :return: None. Displays the confusion matrices using matplotlib.
    """
    # plotting both confusion matrices side by side
    _, axes = plt.subplots(
        nrows = 1,
        ncols = 2,
        figsize = (12, 6)
    )

    # val confusion matrix plot
    sns.heatmap(
        best_val_data_confusion_matrix,
        annot = True,
        fmt = "d",
        linewidths = 0.5,
        square = True,
        cmap = 'Blues',
        xticklabels = ['Negative', 'Positive'],
        yticklabels = ['Negative', 'Positive'],
        ax = axes[0]
    )
    axes[0].set_ylabel('Actual Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title(
        f'Val Data Confusion Matrix, {target_metric_substring} (AUC: {val_auc:.2f})',
        size = 10
    )

    # test confusion matrix plot
    sns.heatmap(
        associated_test_data_confusion_matrix,
        annot = True,
        fmt = "d",
        linewidths = 0.5,
        square = True,
        cmap = 'Blues',
        xticklabels = ['Negative', 'Positive'],
        yticklabels = ['Negative', 'Positive'],
        ax = axes[1]
    )
    axes[1].set_ylabel('Actual Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title(
        f'Test Data Confusion Matrix, {target_metric_substring} (AUC: {test_auc:.2f})',
        size = 10
    )

    # display plots
    plt.tight_layout()
    plt.show()

    return


def find_threshold(
    metric_values: np.ndarray,
    targets: List[float],
    thresholds_val: np.ndarray
) -> List[float]:
    """
    Purpose: Finds thresholds corresponding to the closest values in `metric_values` for each
        target in `targets`.
    :param metric_values (np.ndarray): Array of metric values (either TPR or FPR from ROC).
    :param targets (List[float]): List of target values for which to find the closest thresholds.
    :param thresholds_val (np.ndarray): Array of thresholds corresponding to the metric values.
    :return: List[float] containing thresholds closest to each target metric value.
    """
    closest_thresholds = [
        thresholds_val[np.argmin(np.abs(metric_values - target))] for target in targets
    ]
    return closest_thresholds


def evaluate_thresholds(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    target_tprs: List[float] = [0.8],
    target_fprs: List[float] = [0.05]
) -> None:
    """
    Purpose: Evaluates model performance by applying thresholds to validation
        and test data based on targeted TPR and FPR values, displaying results.
    :param y_true_val: np.ndarray representing true labels of validation dataset.
    :param y_pred_val: np.ndarray representing predicted probabilities from validation dataset.
    :param y_true_test: np.ndarray representing true labels of test dataset.
    :param y_pred_test: np.ndarray representing predicted probabilities from test dataset.
    :param target_tprs: List[float] representing targeted true positive rates to evaluate.
    :param target_fprs: List[float] representing targeted false positive rates to evaluate.
    :return: None.
    """
    # get ROC AUC for both val and test
    val_auc = roc_auc_score(
        y_true = y_true_val,
        y_score = y_pred_val
    )
    test_auc = roc_auc_score(
        y_true = y_true_test,
        y_score = y_pred_test
    )

    # compute ROC curve for validation set;
    # we will use val threshold on test, too
    fpr_val, tpr_val, thresholds_val = roc_curve(
        y_true = y_true_val,
        y_score = y_pred_val
    )

    # process each target TPR
    if target_tprs:
        thresholds_tpr = find_threshold(
            metric_values = tpr_val,
            targets = target_tprs,
            thresholds_val = thresholds_val
        )

        for i, threshold in enumerate(thresholds_tpr):
            # apply threshold to both validation and test data
            pred_val = [
                1 if x >= threshold else 0 for x in y_pred_val
            ]
            pred_test = [
                1 if x >= threshold else 0 for x in y_pred_test
            ]

            # calculate metrics for validation set
            val_recall = recall_score(
                y_true = y_true_val,
                y_pred = pred_val
            )
            val_accuracy = accuracy_score(
                y_true = y_true_val,
                y_pred = pred_val
            )
            val_conf_matrix = confusion_matrix(
                y_true = y_true_val,
                y_pred = pred_val
            )

            # calculate metrics for test set
            test_recall = recall_score(
                y_true = y_true_test,
                y_pred = pred_test
            )
            test_accuracy = accuracy_score(
                y_true = y_true_test,
                y_pred = pred_test
            )
            test_conf_matrix = confusion_matrix(
                y_true = y_true_test,
                y_pred = pred_test
            )

            # print results
            print(f"\t- TPR Target: {target_tprs[i]}")
            print(f"\t\t- Validation Accuracy: {val_accuracy}")
            print(f"\t\t- Validation Recall: {val_recall}")
            print(f"\t\t- Test Accuracy: {test_accuracy}")
            print(f"\t\t- Test Recall: {test_recall}")

            # print confusion matrices
            display_val_test_confusion_matrices(
                best_val_data_confusion_matrix = val_conf_matrix,
                associated_test_data_confusion_matrix = test_conf_matrix,
                val_auc = val_auc,
                test_auc = test_auc,
                target_metric_substring = f"TPR={target_tprs[i]}"
            )

    # process each target FPR
    if target_fprs:
        thresholds_fpr = find_threshold(
            metric_values = fpr_val,
            targets = target_fprs,
            thresholds_val = thresholds_val
        )

        for i, threshold in enumerate(thresholds_fpr):

            # apply threshold to both validation and test data
            pred_val = [
                1 if x >= threshold else 0 for x in y_pred_val
            ]
            pred_test = [
                1 if x >= threshold else 0 for x in y_pred_test
            ]

            # calculate metrics for validation and test sets
            val_accuracy = accuracy_score(
                y_true = y_true_val,
                y_pred = pred_val
            )
            val_conf_matrix = confusion_matrix(
                y_true = y_true_val,
                y_pred = pred_val
            )
            val_specificity = val_conf_matrix[0, 0] / (val_conf_matrix[0, 0] + val_conf_matrix[0, 1])

            test_accuracy = accuracy_score(
                y_true = y_true_test,
                y_pred = pred_test
            )
            test_conf_matrix = confusion_matrix(
                y_true = y_true_test,
                y_pred = pred_test
            )
            test_specificity = test_conf_matrix[0, 0] / (test_conf_matrix[0, 0] + test_conf_matrix[0, 1])

            # print results for validation and test sets
            print(f"\t- FPR Target: {target_fprs[i]}")
            print(f"\t\t- Validation Accuracy: {val_accuracy}")
            print(f"\t\t- Validation Specificity: {val_specificity}")
            print(f"\t\t- Test Accuracy: {test_accuracy}")
            print(f"\t\t- Test Specificity: {test_specificity}")

            # print confusion matrices
            display_val_test_confusion_matrices(
                best_val_data_confusion_matrix = val_conf_matrix,
                associated_test_data_confusion_matrix = test_conf_matrix,
                val_auc = val_auc,
                test_auc = test_auc,
                target_metric_substring = f"FPR={target_fprs[i]}"
            )

    return
