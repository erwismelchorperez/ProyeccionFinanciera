from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps


class HyperparameterRidge:
    def __init__(self):
        self.param_grid = {
            "alpha": [0.1, 1.0, 10.0]
        }
        self.model = None
        self.best_model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train(self, X_train, y_train):
        # escalar como en Linear
        Xs = self.scaler_X.fit_transform(X_train)
        ys = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        grid = GridSearchCV(
            Ridge(),
            self.param_grid,
            cv=10,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(Xs, ys)
        self.best_model = grid.best_estimator_
        self.model = self.best_model
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        Xs = self.scaler_X.transform(X_test)
        y_pred_s = model.predict(Xs)
        y_pred = self.scaler_y.inverse_transform(
            y_pred_s.reshape(-1, 1)
        ).ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        if len(np.unique(y_test)) > 1:
            r2 = r2_score(y_test, y_pred)
        else:
            r2 = float("nan")

        return {
            "y_true": y_test,
            "y_pred": y_pred,
            "MSE": float(mse),
            "R2": float(r2) if r2 == r2 else r2,  # deja NaN si toca
            "RMSE": rmse,
            "MAE": mae,
            "MEDAE": medae,
        }

    def predecir_futuro(
        self,
        modelo=None,
        historial_inicial=None,
        meses_a_predecir=12,
        ventana=3,
        flag_ventana=True,
    ):
        # usar el modelo interno si no lo pasan
        if modelo is None:
            modelo = self.model
        if modelo is None:
            raise AttributeError("No hay modelo entrenado disponible en self.model.")

        hist = list(historial_inicial)
        preds = []

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada = np.array(hist[-ventana:]).reshape(1, -1)
            else:
                entrada = np.array([hist[-1]]).reshape(1, -1)

            entrada_s = self.scaler_X.transform(entrada)
            yhat_s = modelo.predict(entrada_s)[0]
            yhat = self.scaler_y.inverse_transform([[yhat_s]])[0][0]

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)

class HyperparameterRidge_PSO:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.X_train_scaled = None
        self.y_train_scaled = None
        self.best_model = None
        self.model = None
        self.best_params_ = None

    def _objective_function(self, particles):
        losses = []
        for p in particles:
            # alpha en un rango razonable
            alpha = float(np.clip(p[0], 1e-4, 100.0))
            model = Ridge(alpha=alpha)
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)
        return np.array(losses)

    def train(self, X_train, y_train, iters=50, swarm_size=20):
        # escalar
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        # búsqueda de alpha en [0.0001, 100]
        bounds = (np.array([1e-4]), np.array([100.0]))

        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=1,
            options={"c1": 0.5, "c2": 0.3, "w": 0.9},
            bounds=bounds,
        )

        best_cost, best_position = optimizer.optimize(
            self._objective_function,
            iters=iters,
            verbose=False,
        )

        alpha = float(best_position[0])
        self.best_model = Ridge(alpha=alpha)
        self.best_model.fit(self.X_train_scaled, self.y_train_scaled)

        self.model = self.best_model
        self.best_params_ = {
            "alpha": alpha,
            "pso_cost": best_cost,
        }
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        Xs = self.scaler_X.transform(X_test)
        y_pred_s = model.predict(Xs)
        y_pred = self.scaler_y.inverse_transform(
            y_pred_s.reshape(-1, 1)
        ).ravel()

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        if len(np.unique(y_test)) > 1:
            r2 = r2_score(y_test, y_pred)
        else:
            r2 = float("nan")

        return {
            "y_true": y_test,
            "y_pred": y_pred,
            "MSE": float(mse),
            "R2": float(r2) if r2 == r2 else r2,
            "RMSE": rmse,
            "MAE": mae,
            "MEDAE": medae,
            "best_params_": self.best_params_,
        }

    def predecir_futuro(
        self,
        modelo=None,
        historial_inicial=None,
        meses_a_predecir=12,
        ventana=3,
        flag_ventana=True,
    ):
        if modelo is None:
            modelo = self.model
        if modelo is None:
            raise AttributeError("No hay modelo entrenado en self.model.")

        hist = list(historial_inicial)
        preds = []

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada = np.array(hist[-ventana:]).reshape(1, -1)
            else:
                entrada = np.array([hist[-1]]).reshape(1, -1)

            entrada_s = self.scaler_X.transform(entrada)
            yhat_s = modelo.predict(entrada_s)[0]
            yhat = self.scaler_y.inverse_transform([[yhat_s]])[0][0]

            preds.append(yhat)
            hist.append(yhat)

        return np.array(preds, float)
