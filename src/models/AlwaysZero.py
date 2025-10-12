import numpy as np

class AlwaysZeroWrapper:
    def __init__(self):
        self.default_ventana = 3
        self.default_flag_ventana = True

    def train(self, X_train, y_train, **kwargs):
        # Nada que entrenar
        self.model="ZERO"
        return self

    def evaluate(self, model, X_test, y_test):
        y = np.asarray(y_test, float).ravel()
        yhat = np.zeros_like(y)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y, yhat)
        mae = mean_absolute_error(y, yhat)
        try:
            r2 = r2_score(y, yhat)
        except Exception:
            r2 = float('nan')
        return {"y_true": y, "y_pred": yhat, "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
                "MAE": float(mae), "R2": float(r2)}

    def predecir_futuro(self, modelo=None, historial_inicial=None,
                        meses_a_predecir=12, ventana=None, flag_ventana=None, **_):
        return np.zeros(meses_a_predecir, float)
