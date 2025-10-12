import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler

class ZeroInflatedPoissonWrapper:
    def __init__(self):
        self.model = None            # ZeroInflatedPoisson (spec)
        self.result = None           # ZeroInflatedPoissonResults (fitted)
        self.scaler_X = StandardScaler()
        # Defaults para forecast
        self.default_ventana = 3
        self.default_flag_ventana = True

    def _check_target_is_count(self, y):
        y = np.asarray(y)
        if np.any(y < 0):
            raise ValueError("ZeroInflatedPoisson requiere target >= 0 (cuentas). Hay negativos.")
        # Forzar a entero (Poisson modela conteos)
        if not np.issubdtype(y.dtype, np.integer):
            if np.allclose(y, np.round(y)):
                y = np.round(y).astype(int)
            else:
                raise ValueError("ZeroInflatedPoisson requiere enteros; el target no parece conteo.")
        return y.astype(int)

    def train(self, X_train, y_train, **kwargs):
        # Escalar y validar target
        Xs = self.scaler_X.fit_transform(X_train)
        y  = self._check_target_is_count(y_train)

        # Agregar constante (para mean y para inflación)
        Xc = sm.add_constant(Xs, has_constant='add')
        # Especificación
        self.model = sm.ZeroInflatedPoisson(y, Xc, exog_infl=Xc, inflation='logit')
        # Ajuste
        self.result = self.model.fit(disp=0)
        return self.result  # compat con tu pipeline

    def evaluate(self, model=None, X_test=None, y_test=None):
        model = model or self.result
        if model is None:
            raise AttributeError("No hay modelo entrenado disponible (self.result).")

        Xs = self.scaler_X.transform(X_test)
        Xc = sm.add_constant(Xs, has_constant='add')

        y_pred = model.predict(exog=Xc, exog_infl=Xc)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=0.0)

        y = np.asarray(y_test, dtype=float)
        mse   = mean_squared_error(y, y_pred)
        r2    = r2_score(y, y_pred)
        rmse  = float(np.sqrt(mse))
        mae   = mean_absolute_error(y, y_pred)
        medae = median_absolute_error(y, y_pred)

        return {
            'y_true': y,
            'y_pred': y_pred,
            'MSE': float(mse),
            'R2': float(r2),
            'RMSE': rmse,
            'MAE': float(mae),
            'MEDAE': float(medae),
        }

    def predecir_futuro(self, modelo=None, historial_inicial=None,
                        meses_a_predecir=12, ventana=None, flag_ventana=None):
        """
        Predicción autoregresiva usando el modelo ZIP ajustado.
        Si 'modelo' es None, usa self.result.
        """
        modelo = modelo or self.result
        if modelo is None:
            raise AttributeError("No hay modelo entrenado disponible (self.result).")

        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana

        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada = np.array(hist[-ventana:], dtype=float).reshape(1, -1)
            else:
                entrada = np.array([hist[-1]], dtype=float).reshape(1, -1)

            entrada = self.scaler_X.transform(entrada)
            entrada = sm.add_constant(entrada, has_constant='add')

            yhat = float(modelo.predict(exog=entrada, exog_infl=entrada)[0])
            # Poisson → no-negativo
            yhat = max(yhat, 0.0)

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)
