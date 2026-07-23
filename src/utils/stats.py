import numpy as np
from scipy import stats

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test to assess statistical significance between two classifiers.
    Returns statistic and p-value.
    """
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))

    if b + c == 0:
        return 0.0, 1.0

    stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p_val = stats.chi2.sf(stat, df=1)
    return float(stat), float(p_val)

def calculate_confidence_interval(metric_values, confidence=0.95):
    """Calculate bootstrapped 95% confidence intervals."""
    mean_val = np.mean(metric_values)
    sem = stats.sem(metric_values)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(metric_values) - 1)
    return float(mean_val), float(mean_val - h), float(mean_val + h)
