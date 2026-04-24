"""
utils/models/classical.py
=========================
Scikit-learn and XGBoost model factories.
CONFIG dict is passed in from the calling script to keep hyperparameters centralised.
"""

from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    _XGBOOST_OK = True
except ImportError:
    _XGBOOST_OK = False


def get_random_forest(config: dict, class_weight=None) -> RandomForestClassifier:
    """
    Return a configured RandomForestClassifier.

    config : CONFIG["random_forest"] dict from the calling script.
    class_weight : optional override; if None, uses config value ("balanced").
    """
    params = config["random_forest"].copy()
    if class_weight is not None:
        params["class_weight"] = class_weight
    return RandomForestClassifier(**params)


def get_xgboost(config: dict) -> "XGBClassifier":
    """
    Return a configured XGBClassifier.
    Sample weights are applied at fit time, not here.
    """
    if not _XGBOOST_OK:
        raise ImportError(
            "xgboost is not installed. Run: pip install xgboost"
        )
    params = config["xgboost"].copy()
    return XGBClassifier(**params)
