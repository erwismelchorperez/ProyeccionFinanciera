import statsmodels.api as sm
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler


class ZeroInflatedPoissonWrapper:
    """
    Wrapper para ZeroInflatedPoisson que:
      - puede entrenar con X,y (como ya lo tenías)
      - o puede entrenar directo desde una serie mensual con índices de fecha
        usando lags y split por fecha (train_from_series)
      - mantiene la firma evaluate(...) y predecir_futuro(...)
    """
    def __init__(self):
        self.model = None             # spec
        self.result = None            # fitted
        self.scaler_X = StandardScaler()

        # defaults para forecast autoregresivo
        self.default_ventana = 3
        self.default_flag_ventana = True

        # para saber con qué serie se entrenó
        self.train_start = None
        self.train_end = None
        self.colname = None

    # ---------------------------------------------------------
    # helpers internos
    # ---------------------------------------------------------
    def _check_target_is_count(self, y):
        y = np.asarray(y)
        if np.any(y < 0):
            raise ValueError("ZeroInflatedPoisson requiere target >= 0 (cuentas). Hay negativos.")

        # forzar a entero
        if not np.issubdtype(y.dtype, np.integer):
            # si parece entero, lo redondeamos
            if np.allclose(y, np.round(y)):
                y = np.round(y).astype(int)
            else:
                raise ValueError("ZeroInflatedPoisson requiere enteros; el target no parece conteo.")
        return y.astype(int)

    def _build_lags_from_series(self, serie: pd.Series, n_lags: int = 3):
        """
        serie: Serie 1D con índice datetime y valores >=0
        Devuelve X, y donde X son los lags y y es el valor actual.
        """
        serie = serie.sort_index()
        vals = serie.values.astype(float)
        X_list, y_list = [], []

        for i in range(n_lags, len(vals)):
            X_list.append(vals[i - n_lags:i])
            y_list.append(vals[i])

        X = np.asarray(X_list, float)
        y = np.asarray(y_list, float)
        # recortar el índice también, por si quieres
        idx = serie.index[n_lags:]
        return X, y, idx

    def _train_test_split_by_date(self,
                                  idx: pd.DatetimeIndex,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  train_start: dt.datetime,
                                  train_end: dt.datetime):
        """
        Recibe índice datetime que corresponde a X,y y parte en train/test según fechas.
        """
        # máscara de train
        mask_train = (idx >= pd.to_datetime(train_start)) & (idx <= pd.to_datetime(train_end))
        X_train = X[mask_train]
        y_train = y[mask_train]

        # lo que queda después de train_end es test
        mask_test = (idx > pd.to_datetime(train_end))
        X_test = X[mask_test]
        y_test = y[mask_test]
        idx_test = idx[mask_test]

        return X_train, y_train, X_test, y_test, idx_test

    # ---------------------------------------------------------
    # 1) train clásico (el tuyo)
    # ---------------------------------------------------------
    def train(self, X_train, y_train, **kwargs):
        # Escalar y validar target
        Xs = self.scaler_X.fit_transform(X_train)
        y  = self._check_target_is_count(y_train)

        # Agregar constante (para mean y para inflación)
        Xc = sm.add_constant(Xs, has_constant='add')

        self.model = sm.ZeroInflatedPoisson(y, Xc, exog_infl=Xc, inflation='logit')
        self.result = self.model.fit(disp=0)
        return self.result  # compat con tu pipeline

    # ---------------------------------------------------------
    # 2) train_from_series: versión para tu loop nuevo
    # ---------------------------------------------------------
    def train_from_series(self,
                          serie: pd.Series | pd.DataFrame,
                          n_lags: int = 3,
                          train_start: dt.datetime = dt.datetime(2013, 1, 1),
                          train_end: dt.datetime = dt.datetime(2024, 1, 1),
                          colname: str | None = None):
        """
        serie: mensual (o diaria pero de conteos), UNA sola columna/cuenta.
               Debe ser >=0 porque es Poisson.
        Hace:
          - si es DF, toma la primera col
          - crea lags
          - parte por fecha
          - entrena ZIP
        Devuelve el mismo dict que usas en model_scores.
        """
        # normalizar a Serie
        if isinstance(serie, pd.DataFrame):
            if serie.shape[1] != 1:
                raise ValueError("train_from_series espera una sola columna para ZIP.")
            serie = serie.iloc[:, 0]

        # guardar nombre
        if colname is None:
            colname = serie.name or "cuenta"
        self.colname = colname

        # construir lags
        X_all, y_all, idx_all = self._build_lags_from_series(serie, n_lags=n_lags)

        # split por fecha
        X_train, y_train, X_test, y_test, idx_test = self._train_test_split_by_date(
            idx_all, X_all, y_all,
            train_start=train_start,
            train_end=train_end
        )

        # entrena con el método clásico
        self.train(X_train, y_train)

        # evaluar si hay test
        if len(X_test) > 0:
            eval_res = self.evaluate(self.result, X_test, y_test)
            # guardamos el índice de test para graficar luego
            eval_res["index_test"] = idx_test
        else:
            eval_res = {
                "y_true": np.array([]),
                "y_pred": np.array([]),
                "MSE": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "MAE": np.nan,
                "MEDAE": np.nan,
                "index_test": idx_test
            }

        # guardar fechas por si las quieres en el wrapper
        self.train_start = train_start
        self.train_end = train_end

        return eval_res

    # ---------------------------------------------------------
    # 3) evaluate (igual que tenías)
    # ---------------------------------------------------------
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
        r2    = r2_score(y, y_pred) if len(np.unique(y)) > 1 else float("nan")
        rmse  = float(np.sqrt(mse))
        mae   = mean_absolute_error(y, y_pred)
        medae = median_absolute_error(y, y_pred)

        return {
            'y_true': y,
            'y_pred': y_pred,
            'MSE': float(mse),
            'R2': float(r2) if not np.isnan(r2) else r2,
            'RMSE': rmse,
            'MAE': float(mae),
            'MEDAE': float(medae),
        }

    # ---------------------------------------------------------
    # 4) predecir_futuro (igual que tenías)
    # ---------------------------------------------------------
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
