from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error
)
import numpy as np
import pyswarms as ps   # ← para la clase PSO

class HyperparameterLinear:
    def __init__(self):
        # scalers propios del wrapper
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        # por si quieres usar ventana en forecast
        self.default_ventana = 3
        self.default_flag_ventana = True

    def train(self, X_train, y_train):
        """
        X_train, y_train: ya vienen de hacer_lags_y_split(...)
        Aquí SOLO escalamos y entrenamos un LinearRegression sencillo.
        (Si luego quieres GridSearch, lo puedes volver a meter.)
        """
        # 1) escalar
        Xs = self.scaler_X.fit_transform(X_train)
        ys = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # 2) modelo lineal
        lr = LinearRegression()
        lr.fit(Xs, ys)

        self.model = lr
        return self.model

    def evaluate(self, model, X_test, y_test):
        # aplicar el mismo escalador de X
        Xs = self.scaler_X.transform(X_test)
        y_pred_scaled = model.predict(Xs)

        # desescalar predicción
        y_pred = self.scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        # métricas contra y_test (y_test está en escala original)
        mse = mean_squared_error(y_test, y_pred)
        r2  = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        return {
            "y_true": np.asarray(y_test),
            "y_pred": y_pred,
            "MSE": float(mse),
            "R2": float(r2),
            "RMSE": rmse,
            "MAE": float(mae),
            "MEDAE": float(medae),
        }

    def predecir_futuro(self,
                        modelo=None,
                        historial_inicial=None,
                        meses_a_predecir=12,
                        ventana=None,
                        flag_ventana=None,
                        ultimas_vars=None):
        """
        historial_inicial: últimos valores de la SERIE (escala original)
        ultimas_vars: np.array de las últimas variables exógenas usadas en train.
                      Puede ser None si entrenaste solo con la serie.
        """
        if modelo is None:
            if self.model is None:
                raise ValueError("No hay modelo entrenado en el wrapper.")
            modelo = self.model

        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana

        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []

        # cuántas columnas esperaba el scaler al entrenar
        n_feats_entreno = self.scaler_X.n_features_in_

        for _ in range(meses_a_predecir):
            # 1) parte de la serie
            if flag_ventana:
                entrada_serie = np.array(hist[-ventana:], float).reshape(1, -1)
            else:
                entrada_serie = np.array([hist[-1]], float).reshape(1, -1)

            # 2) si en train hubo más columnas, las agregamos
            if entrada_serie.shape[1] < n_feats_entreno:
                faltan = n_feats_entreno - entrada_serie.shape[1]

                if ultimas_vars is None:
                    extras = np.zeros((1, faltan), float)
                else:
                    extras = np.array(ultimas_vars, float).reshape(1, -1)
                    # recorta o rellena
                    if extras.shape[1] > faltan:
                        extras = extras[:, :faltan]
                    elif extras.shape[1] < faltan:
                        extras = np.concatenate(
                            [extras, np.zeros((1, faltan - extras.shape[1]))],
                            axis=1
                        )
                entrada = np.concatenate([entrada_serie, extras], axis=1)
            else:
                entrada = entrada_serie

            # 3) escalar y predecir
            entrada_s = self.scaler_X.transform(entrada)
            yhat_s = modelo.predict(entrada_s)[0]
            yhat = self.scaler_y.inverse_transform([[yhat_s]])[0][0]

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)
    

class HyperparameterLinear_PSO:
    """
    Igual interfaz que tu clase Linear "bonita", pero por dentro
    usa PSO para elegir (fit_intercept, positive).
    """
    def __init__(self):
        # escaladores para X e y (como tenías)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # se llenan en train()
        self.X_train_scaled = None
        self.y_train_scaled = None
        self.best_model = None
        self.model = None
        self.best_params_ = None   # para parecerse al otro
        self.train_time = None     # opcional

    # ---------- función objetivo para PSO ----------
    def _objective_function(self, particles):
        """
        particles: array (n_particles, 2)
          col 0 -> fit_intercept (0/1)
          col 1 -> positive (0/1)
        Devuelve el MSE sobre el train escalado.
        """
        losses = []
        for p in particles:
            fit_intercept = bool(round(p[0]))
            positive      = bool(round(p[1]))
            model = LinearRegression(
                fit_intercept=fit_intercept,
                positive=positive
            )
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)
        return np.array(losses)

    # ---------- train ----------
    def train(self, X_train, y_train, iters=50, swarm_size=20):
        """
        X_train, y_train vienen ya con tus lags hechos.
        Aquí escalamos, corremos PSO y entrenamos el mejor modelo.
        """
        # 1) escalar como ya lo hacías
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        # 2) límites de las 2 variables: 0 o 1
        bounds = (np.array([0, 0]), np.array([1, 1]))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=2,
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            bounds=bounds
        )

        # 3) optimizar
        best_cost, best_position = optimizer.optimize(
            self._objective_function,
            iters=iters,
            verbose=False
        )

        fit_intercept = bool(round(best_position[0]))
        positive      = bool(round(best_position[1]))

        # 4) entrenar modelo final con esos hiperparámetros
        self.best_model = LinearRegression(
            fit_intercept=fit_intercept,
            positive=positive
        )
        self.best_model.fit(self.X_train_scaled, self.y_train_scaled)

        # para que tu pipeline lo encuentre
        self.model = self.best_model
        self.best_params_ = {
            "fit_intercept": fit_intercept,
            "positive": positive,
            "pso_cost": best_cost,
        }
        return self.best_model

    # ---------- evaluate ----------
    def evaluate(self, model, X_test, y_test):
        """
        Igual firma que tus otros modelos.
        """
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        mse   = mean_squared_error(y_test, y_pred)
        rmse  = float(np.sqrt(mse))
        mae   = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2    = r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else float("nan")

        return {
            "y_true": y_test,
            "y_pred": y_pred,
            "MSE": float(mse),
            "RMSE": rmse,
            "MAE": float(mae),
            "MEDAE": float(medae),
            "R2": float(r2) if not np.isnan(r2) else r2,
            "best_params_": self.best_params_,
        }

    # ---------- predecir_futuro ----------
    def predecir_futuro(self, modelo=None, historial_inicial=None,
                         meses_a_predecir=12, ventana=3, flag_ventana=True):
        """
        Misma interfaz que tus otros wrappers.
        Usa el MISMO scaler_X / scaler_y que se aprendió en train().
        """
        if modelo is None:
            modelo = self.model
        if modelo is None:
            raise AttributeError("No hay modelo entrenado en self.model.")

        hist = list(np.ravel(historial_inicial).astype(float))
        preds = []

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada = np.array(hist[-ventana:], dtype=float).reshape(1, -1)
            else:
                entrada = np.array([hist[-1]], dtype=float).reshape(1, -1)

            entrada_scaled = self.scaler_X.transform(entrada)
            yhat_scaled = modelo.predict(entrada_scaled)[0]
            yhat = self.scaler_y.inverse_transform([[yhat_scaled]])[0][0]

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)