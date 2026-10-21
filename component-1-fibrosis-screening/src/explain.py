"""
explain.py  --  SHAP reasons for whichever model was chosen.

Every candidate model is a Pipeline: (fill missing values [+ scale]) -> classifier.
Tree models use SHAP's TreeExplainer, logistic regression uses LinearExplainer.
"""
import numpy as np
import shap


def shap_values_for(pipe, X_background, X):
    """Returns one row of SHAP values per patient in X (positive = raises risk)."""
    pre, clf = pipe[:-1], pipe[-1]
    Xt, bg = pre.transform(X), pre.transform(X_background)
    if hasattr(clf, "coef_"):                       # logistic regression
        sv = shap.LinearExplainer(clf, bg).shap_values(Xt)
    else:                                           # gradient boosting / random forest
        sv = shap.TreeExplainer(clf).shap_values(Xt)
    sv = np.asarray(sv)
    if sv.ndim == 3:                                # random forest: one slice per class
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[1]
    return sv


def shap_base_value(pipe, X_background):
    """The model's average output (log-odds) that every patient's SHAP values start from."""
    pre, clf = pipe[:-1], pipe[-1]
    bg = pre.transform(X_background)
    if hasattr(clf, "coef_"):
        ev = shap.LinearExplainer(clf, bg).expected_value
    else:
        ev = shap.TreeExplainer(clf).expected_value
    return float(np.atleast_1d(ev)[-1])             # last entry = "has fibrosis" class
