# =============================================================================
# utils/metrics.py — Model evaluation metrics
# NOTE: Full implementation will be added in stage 06 (06_model_training.py)
# =============================================================================


def classification_report_df(y_true, y_pred):
    """Return a tidy DataFrame version of sklearn's classification_report.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels from the model.

    Returns
    -------
    pd.DataFrame
        Rows = classes + macro/weighted averages.
        Columns = precision, recall, f1-score, support.
    """
    pass


def roc_auc_score_multiclass(y_true, y_proba):
    """Compute the macro-averaged one-vs-rest ROC-AUC for multiclass problems.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True integer class labels.
    y_proba : array-like of shape (n_samples, n_classes)
        Predicted class probabilities from the model.

    Returns
    -------
    float
        Macro-averaged ROC-AUC score.
    """
    pass
