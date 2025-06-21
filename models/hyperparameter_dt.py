from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pyswarms as ps
class HyperparameterDT:
    def __init__(self):
        # Definir el modelo base
        self.model = DecisionTreeRegressor(random_state=42)
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Parámetros para GridSearchCV (puedes ajustar estos)
        self.param_grid = {
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'auto', 'sqrt', 'log2']
        }

        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring='neg_mean_squared_error',
            cv=10,
            n_jobs=-1,
            verbose=1
        )
    def train(self, X_train, y_train):
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.grid_search.fit(X_train, y_train)
        return self.grid_search.best_estimator_
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

class HyperparameterDT_PSO:
    def __init__(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.best_model = None

    def _objective_function(self, params):
        """
        PSO pasa un array 2D: cada fila es un conjunto de parámetros a evaluar.
        Por eso iteramos en batch.
        params es shape (n_particles, dimensions)
        """
        n_particles = params.shape[0]
        losses = []
        for i in range(n_particles):
            max_depth = int(params[i][0])
            min_samples_split = int(params[i][1])
            min_samples_leaf = int(params[i][2])

            # Evitar valores inválidos (menores que 1)
            max_depth = max(1, max_depth)
            min_samples_split = max(2, min_samples_split)
            min_samples_leaf = max(1, min_samples_leaf)

            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(self.X_train_scaled, self.y_train_scaled)
            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)
        return np.array(losses)

    def train(self, X_train, y_train, iters=30, swarm_size=20):
        self.X_train_scaled = self.scaler_X.fit_transform(X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Límites para los parámetros: [max_depth, min_samples_split, min_samples_leaf]
        # max_depth entre 1 y 30
        # min_samples_split entre 2 y 20
        # min_samples_leaf entre 1 y 10
        bounds = (np.array([1, 2, 1]), np.array([30, 20, 10]))

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # parámetros PSO, puedes ajustarlos

        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=3,
            options=options,
            bounds=bounds
        )

        best_cost, best_pos = optimizer.optimize(self._objective_function, iters)

        print("\n🧠 Mejores hiperparámetros encontrados por PSO (pyswarms):")
        print(f"  - max_depth = {int(best_pos[0])}")
        print(f"  - min_samples_split = {int(best_pos[1])}")
        print(f"  - min_samples_leaf = {int(best_pos[2])}")

        self.best_model = DecisionTreeRegressor(
            max_depth=int(best_pos[0]),
            min_samples_split=int(best_pos[1]),
            min_samples_leaf=int(best_pos[2]),
            random_state=42
        )
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