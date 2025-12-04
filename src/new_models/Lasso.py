import numpy as np
import pyswarms as ps
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error
)
from sklearn.model_selection import TimeSeriesSplit


class HyperparameterLasso:
    """
    Lasso “normal” con GridSearch, misma interfaz que tus otros wrappers.
    """
    def __init__(self):
        # puedes ajustar estos al vuelo
        self.param_grid = {
            "alpha": [0.001, 0.01, 0.1, 1.0]
        }
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.grid_search = None
        self.best_model = None
        self.model = None
        self.best_params_ = None
        self.default_ventana=3
        self.default_flag_ventana=True

    def train(self, X_train, y_train):
        # escalar
        Xs = self.scaler_X.fit_transform(X_train)
        ys = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # usar CV de series si quieres ser homogéneo
        cv = TimeSeriesSplit(n_splits=5)

        grid = GridSearchCV(
            Lasso(max_iter=10000),
            self.param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(Xs, ys)

        self.grid_search = grid
        self.best_model = grid.best_estimator_
        self.model = self.best_model
        self.best_params_ = grid.best_params_
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        Xs = self.scaler_X.transform(X_test)
        y_pred_s = model.predict(Xs)
        y_pred = self.scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else float("nan")

        return {
            "y_true": np.asarray(y_test),
            "y_pred": y_pred,
            "MSE": float(mse),
            "RMSE": rmse,
            "MAE": float(mae),
            "MEDAE": float(medae),
            "R2": float(r2) if not np.isnan(r2) else r2,
            "best_params_": self.best_params_,
        }

    def predecir_futuro(self,
                        modelo=None,
                        historial_inicial=None,
                        meses_a_predecir=12,
                        ventana=None,
                        flag_ventana=None,
                        ultimas_vars=None):
        """
        Igual que el Linear que ya ajustaste:
        - usa los últimos 'ventana' de la serie
        - si el modelo fue entrenado con más columnas (variables), las pegamos al final
        - usa los mismos scalers
        """
        if modelo is None:
            modelo = self.model
        if modelo is None:
            raise ValueError("No hay modelo entrenado en el wrapper.")

        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana

        # historial de la serie en escala original
        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []

        # cuántas features tenía X en train
        n_feats_entreno = self.scaler_X.n_features_in_

        for _ in range(meses_a_predecir):
            # 1) parte autoregresiva
            if flag_ventana:
                entrada_serie = np.array(hist[-ventana:], float).reshape(1, -1)
            else:
                entrada_serie = np.array([hist[-1]], float).reshape(1, -1)

            # 2) ¿entrenamos con más columnas?
            if entrada_serie.shape[1] < n_feats_entreno:
                faltan = n_feats_entreno - entrada_serie.shape[1]

                if ultimas_vars is None:
                    extras = np.zeros((1, faltan), float)
                else:
                    extras = np.array(ultimas_vars, float).reshape(1, -1)
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


class HyperparameterLasso_PSO:
    """
    Igual interfaz que las demás *_PSO.
    Busca el mejor alpha en un rango continuo con PSO.
    """
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X_train_scaled = None
        self.y_train_scaled = None
        self.best_model = None
        self.model = None
        self.best_params_ = None
        self.default_ventana = 3
        self.default_flag_ventana = True

    # función objetivo para PSO
    def _objective_function(self, alpha_array):
        """
        alpha_array: shape (n_particles, 1)
        """
        losses = []
        for row in alpha_array:
            alpha_val = float(row[0])
            # asegurar que alpha no sea 0
            alpha_val = max(alpha_val, 1e-5)

            model = Lasso(alpha=alpha_val, max_iter=10000)
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)
        return np.array(losses, float)

    def train(self, X_train, y_train, iters=50, swarm_size=20):
        # escalar
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        # alpha en [1e-5, 10]
        bounds = (np.array([1e-5]), np.array([10.0]))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=1,
            options={"c1": 0.5, "c2": 0.3, "w": 0.9},
            bounds=bounds
        )

        best_cost, best_pos = optimizer.optimize(
            self._objective_function,
            iters=iters,
            verbose=False
        )

        best_alpha = float(best_pos[0])
        best_alpha = max(best_alpha, 1e-5)

        # entrenar modelo final
        self.best_model = Lasso(alpha=best_alpha, max_iter=10000)
        self.best_model.fit(self.X_train_scaled, self.y_train_scaled)

        self.model = self.best_model
        self.best_params_ = {
            "alpha": best_alpha,
            "pso_cost": float(best_cost),
        }
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        Xs = self.scaler_X.transform(X_test)
        y_pred_s = model.predict(Xs)
        y_pred = self.scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else float("nan")

        return {
            "y_true": np.asarray(y_test),
            "y_pred": y_pred,
            "MSE": float(mse),
            "RMSE": rmse,
            "MAE": float(mae),
            "MEDAE": float(medae),
            "R2": float(r2) if not np.isnan(r2) else r2,
            "best_params_": self.best_params_,
        }

    def predecir_futuro(self,
                        modelo=None,
                        historial_inicial=None,
                        meses_a_predecir=12,
                        ventana=None,
                        flag_ventana=None,
                        ultimas_vars=None):
        if modelo is None:
            modelo = self.model
        if modelo is None:
            raise ValueError("No hay modelo entrenado en el wrapper.")

        ventana = self.default_ventana if ventana is None else ventana
        flag_ventana = self.default_flag_ventana if flag_ventana is None else flag_ventana

        hist = list(map(float, np.ravel(historial_inicial)))
        preds = []

        n_feats_entreno = self.scaler_X.n_features_in_

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada_serie = np.array(hist[-ventana:], float).reshape(1, -1)
            else:
                entrada_serie = np.array([hist[-1]], float).reshape(1, -1)

            if entrada_serie.shape[1] < n_feats_entreno:
                faltan = n_feats_entreno - entrada_serie.shape[1]

                if ultimas_vars is None:
                    extras = np.zeros((1, faltan), float)
                else:
                    extras = np.array(ultimas_vars, float).reshape(1, -1)
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

            entrada_s = self.scaler_X.transform(entrada)
            yhat_s = modelo.predict(entrada_s)[0]
            yhat = self.scaler_y.inverse_transform([[yhat_s]])[0][0]

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)
