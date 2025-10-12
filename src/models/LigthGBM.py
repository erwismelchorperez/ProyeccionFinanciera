import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LightGBM_TweedieWrapper:
    def __init__(self, tweedie_variance_power=1.5, **params):
        base = dict(
            objective='tweedie',
            tweedie_variance_power=tweedie_variance_power,
            learning_rate=0.03,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_data_in_leaf=12,
            min_gain_to_split=0.0,
            n_estimators=400,
            random_state=42,
        )
        base.update(params)
        self.model = lgb.LGBMRegressor(**base)
        self.scaler_X = StandardScaler()
        # Para evitar el warning de feature names
        self.feature_names_ = None
        self._use_df_ = False
        # Defaults para forecast
        self.default_ventana = 3
        self.default_flag_ventana = True

    def _wrap(self, Xs):
        if self._use_df_:
            return pd.DataFrame(Xs, columns=self.feature_names_)
        return Xs

    def train(self, X_train, y_train, **kwargs):
        y = np.asarray(y_train, float).ravel()

        # 1) Tweedie no admite negativos
        if np.any(y < 0):
            raise ValueError("LightGBM Tweedie no acepta valores negativos en el target.")

        # 2) Casos degenerados (fatal en LightGBM): todo cero
        if np.all(y == 0):
            raise ValueError("LightGBM Tweedie: sum(y)==0 → omito entrenamiento.")

        Xs = self.scaler_X.fit_transform(X_train)
        # recuerda si entrenaste con DF
        if isinstance(X_train, pd.DataFrame):
            self._use_df_ = True
            self.feature_names_ = list(X_train.columns)
        else:
            self._use_df_ = False
            self.feature_names_ = None

        X_in = self._wrap(Xs)

        eval_set = kwargs.get('eval_set')
        fit_params = {}
        if eval_set is not None:
            Xv, yv = eval_set
            Xv = self.scaler_X.transform(Xv)
            Xv_in = self._wrap(Xv)
            # solo validación para early stopping
            fit_params = dict(eval_set=[(Xv_in, yv)], early_stopping_rounds=100, verbose=False)

        self.model.fit(X_in, y, **fit_params)
        # self.model queda entrenado y persistirá al hacer joblib.dump(wrapper, ...)
        return self.model  # por compatibilidad con tu pipeline

    def evaluate(self, model=None, X_test=None, y_test=None):
        model = model or self.model
        if model is None:
            raise AttributeError("No hay modelo entrenado disponible (self.model).")

        Xs = self.scaler_X.transform(X_test)
        X_in = self._wrap(Xs)
        y = np.asarray(y_test, dtype=float)
        yhat = model.predict(X_in)

        mse = mean_squared_error(y, yhat)
        return {
            'y_true': y,
            'y_pred': yhat,
            'MSE': float(mse),
            'RMSE': float(np.sqrt(mse)),
            'MAE': float(mean_absolute_error(y, yhat)),
            'R2' : float(r2_score(y, yhat))
        }

    def predecir_futuro(self, modelo=None, historial_inicial=None,
                        meses_a_predecir=12, ventana=None, flag_ventana=None):
        modelo = modelo or self.model
        if modelo is None or not hasattr(modelo, "predict"):
            raise AttributeError("No hay modelo entrenado disponible (self.model).")

        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana

        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []
        for _ in range(meses_a_predecir):
            if flag_ventana:
                x = np.array(hist[-ventana:], float).reshape(1, -1)
            else:
                x = np.array([hist[-1]], float).reshape(1, -1)
            x = self.scaler_X.transform(x)
            x_in = self._wrap(x)
            yhat = float(modelo.predict(x_in)[0])
            preds.append(yhat)
            hist.append(yhat)
        return np.array(preds, float)
