from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps
class HyperparameterLasso:
    def __init__(self):
        self.param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1.0]
        }
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def train(self, X_train, y_train):
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        grid = GridSearchCV(Lasso(max_iter=10000), self.param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        self.best_model = grid.best_estimator_
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        X_test = self.scaler_X.transform(X_test)
        y_pred = model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        return {'y_true': y_test, 'y_pred': y_pred, 'MSE': mse, 'R2': r2, 'RMSE':rmse,'MAE':mae, 'MEDAE':medae}
    def predecir_futuro(self, modelo, historial_inicial, meses_a_predecir=12, ventana=3, flag_ventana = True):
        """
        Predice valores futuros de forma autoregresiva.
        
        modelo: modelo ya entrenado con .predict()
        historial_inicial: array con los últimos 'ventana' valores conocidos
        meses_a_predecir: número de predicciones a generar
        ventana: tamaño de la ventana temporal
        """
        historial = list(historial_inicial)  # convertir a lista para manejar crecimiento
        predicciones = []
        
        for _ in range(meses_a_predecir):
            if flag_ventana:
                # Usa los últimos 'ventana' valores como entrada
                entrada = np.array(historial[-ventana:]).reshape(1, -1)
            else:
                # Usa solo el último valor como entrada
                entrada = np.array([historial[-1]]).reshape(1, -1)

            entrada = self.scaler_X.transform(entrada)
            prediccion = modelo.predict(entrada)[0]
            prediccion = self.scaler_y.inverse_transform([[prediccion]])[0][0]
            predicciones.append(prediccion)
            historial.append(prediccion)
        
        return np.array(predicciones)

class HyperparameterLasso_PSO:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _objective_function(self, alpha_array):
        losses = []
        for alpha in alpha_array:
            alpha_value = alpha[0]
            model = Lasso(alpha=alpha_value, max_iter=10000)
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)
        return np.array(losses)

    def train(self, X_train, y_train, iters=50, swarm_size=20):
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Definimos los límites del espacio de búsqueda: alpha ∈ [0.0001, 10]
        bounds = (np.array([0.0001]), np.array([10.0]))

        # Creamos el optimizador
        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=1,
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
            bounds=bounds
        )

        best_cost, best_alpha = optimizer.optimize(self._objective_function, iters=iters)

        # Entrenar el mejor modelo encontrado
        self.best_model = Lasso(alpha=best_alpha[0], max_iter=10000)
        self.best_model.fit(self.X_train_scaled, self.y_train_scaled)
        return self.best_model

    def evaluate(self, model, X_test, y_test):
        X_test_scaled = self.scaler_X.transform(X_test)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        return {
            'y_true': y_test,
            'y_pred': y_pred,
            'MSE': mse,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MEDAE': medae
        }

    def predecir_futuro(self, modelo, historial_inicial, meses_a_predecir=12, ventana=3, flag_ventana = True):
        """
        Predice valores futuros de forma autoregresiva.
        
        modelo: modelo ya entrenado con .predict()
        historial_inicial: array con los últimos 'ventana' valores conocidos
        meses_a_predecir: número de predicciones a generar
        ventana: tamaño de la ventana temporal
        """
        historial = list(historial_inicial)  # convertir a lista para manejar crecimiento
        predicciones = []
        
        for _ in range(meses_a_predecir):
            if flag_ventana:
                # Usa los últimos 'ventana' valores como entrada
                entrada = np.array(historial[-ventana:]).reshape(1, -1)
            else:
                # Usa solo el último valor como entrada
                entrada = np.array([historial[-1]]).reshape(1, -1)

            entrada = self.scaler_X.transform(entrada)
            prediccion = modelo.predict(entrada)[0]
            prediccion = self.scaler_y.inverse_transform([[prediccion]])[0][0]
            predicciones.append(prediccion)
            historial.append(prediccion)
        
        return np.array(predicciones)