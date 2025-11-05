import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, average_precision_score, brier_score_loss)
from sklearn.base import clone

class TwoPartHurdleWrapper:
    """
    Parte 1: modelo de presencia (probabilidad de ser >0)
    Parte 2: modelo de monto condicional dado que es >0
    """
    def __init__(self, nonzero_label=True, regressor=None):
        self.scaler_X = StandardScaler()
        self.nonzero_label = nonzero_label

        self._base_log = LogisticRegression(max_iter=2000, class_weight='balanced')
        self._base_reg = (regressor if regressor is not None
                          else HistGradientBoostingRegressor(
                                max_depth=None, learning_rate=0.05, random_state=42))

        self.clf = None
        self.reg = None
        self._p_const = None
        self._single_class = None

        # para que tu pipeline pueda hacer model_obj.model = ...
        self.model = None
        self.default_ventana = 3
        self.default_flag_ventana = True

    def _y_to_bin(self, y):
        return (y != 0).astype(int) if self.nonzero_label else (y > 0).astype(int)

    def train(self, X_train, y_train, **kwargs):
        self.clf = CalibratedClassifierCV(clone(self._base_log), method='isotonic', cv=3)
        self.reg = clone(self._base_reg)
        self._p_const = None
        self._single_class = None

        y = np.asarray(y_train, float).ravel()
        y_bin = self._y_to_bin(y)

        Xs = self.scaler_X.fit_transform(X_train)

        classes = np.unique(y_bin)
        if classes.size < 2:
            # todo 0 o todo 1
            self._single_class = int(classes[0])
            self._p_const = 1.0 if self._single_class == 1 else 0.0

            if self._single_class == 1:
                self.reg.fit(Xs, y)
                self.model = self.reg
            else:
                self.reg = None
                self.model = None
            return self

        # caso normal
        self.clf.fit(Xs, y_bin)
        mask_reg = (y_bin == 1)
        if np.any(mask_reg):
            self.reg.fit(Xs[mask_reg], y[mask_reg])
            self.model = self.reg
        else:
            self.reg = None
            self.model = None
        return self

    def evaluate(self, model=None, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            raise ValueError("X_test e y_test son requeridos.")

        Xs = self.scaler_X.transform(X_test)
        y = np.asarray(y_test, float).ravel()

        # proba de presencia
        if self._p_const is not None:
            p = np.full(Xs.shape[0], float(self._p_const))
        else:
            p = self.clf.predict_proba(Xs)[:, 1]
        p = np.clip(p, 0.0, 1.0)

        # monto condicional
        if self.reg is None:
            y_pos = np.zeros_like(p)
        else:
            y_pos = self.reg.predict(Xs)

        y_hat = p * y_pos

        mse = mean_squared_error(y, y_hat)
        mae = mean_absolute_error(y, y_hat)
        r2  = r2_score(y, y_hat)

        y_bin_test = self._y_to_bin(y)
        if np.unique(y_bin_test).size == 2:
            roc   = roc_auc_score(y_bin_test, p)
            pr    = average_precision_score(y_bin_test, p)
            brier = brier_score_loss(y_bin_test, p)
        else:
            roc = pr = brier = np.nan

        return {
            'y_true': y,
            'y_pred': y_hat,
            'MSE': float(mse),
            'RMSE': float(np.sqrt(mse)),
            'MAE': float(mae),
            'R2': float(r2),
            'ROC_AUC_presence': None if np.isnan(roc) else float(roc),
            'PR_AUC_presence': None if np.isnan(pr) else float(pr),
            'Brier_presence': None if np.isnan(brier) else float(brier),
        }

    def predecir_futuro(self, modelo=None, historial_inicial=None,
                        meses_a_predecir=12, ventana=None, flag_ventana=None,
                        p_threshold=0.35, hurdle_rule=True, alpha=1.0):
        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana
        if historial_inicial is None:
            raise ValueError("historial_inicial es requerido.")

        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []
        for _ in range(meses_a_predecir):
            x = (np.array(hist[-ventana:], float).reshape(1, -1)
                 if flag_ventana else np.array([hist[-1]], float).reshape(1, -1))
            x = self.scaler_X.transform(x)

            # proba
            if self._p_const is not None:
                p = float(self._p_const)
            else:
                p = float(self.clf.predict_proba(x)[:, 1][0])
            p = float(np.clip(p, 0.0, 1.0))

            # monto condicional
            if self.reg is None:
                y_pos = 0.0
            else:
                y_pos = float(self.reg.predict(x)[0])

            if hurdle_rule:
                yhat = 0.0 if p < p_threshold else y_pos
            else:
                yhat = (p ** alpha) * y_pos

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)
