import numpy as np
def infer_target_type(y):
    """'count' si entero-like y sin negativos; en otro caso 'continuous'."""
    y = np.asarray(y).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 'continuous'
    all_int_like = np.allclose(y, np.round(y))
    has_neg = (y < 0).any()
    return 'count' if (all_int_like and not has_neg) else 'continuous'