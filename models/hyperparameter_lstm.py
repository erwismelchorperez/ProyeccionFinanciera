import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pyswarms as ps
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import KFold
class HyperparameterLSTM:
    def __init__(self, timesteps, features):
        self.timesteps = timesteps
        self.features = features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.best_model = None

    def build_model(self, units=50, learning_rate=0.001):
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(self.timesteps, self.features)))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def train(self, X_train, y_train, param_grid=None, epochs=20, batch_size=32):
        # Escalar
        n_samples, t, f = X_train.shape
        X_train_2d = X_train.reshape(n_samples, t * f)
        X_train_scaled = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, t, f)

        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # DEFINIR modelo como una función lambda sin parámetros requeridos
        def model_builder(units=50, learning_rate=0.001):
            model = Sequential()
            model.add(LSTM(units=units, activation='relu', input_shape=(self.timesteps, self.features)))
            model.add(Dense(1))
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
            return model

        keras_reg = KerasRegressor(
            model=model_builder,  # correcta firma
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        if param_grid is None:
            param_grid = {
                "model__units": [50, 100],
                "model__learning_rate": [0.001, 0.01]
            }

        grid = GridSearchCV(
            estimator=keras_reg,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=KFold(n_splits=3),  # fuerza división válida
            n_jobs=-1,
            verbose=1,
            error_score='raise'  # para que lance el error real si falla
        )

        grid.fit(X_train_scaled, y_train_scaled)
        self.best_model = grid.best_estimator_
        return self.best_model
    def evaluate(self, model, X_test, y_test):
        n_samples, t, f = X_test.shape
        X_test_2d = X_test.reshape(n_samples, t * f)
        X_test_scaled = self.scaler_X.transform(X_test_2d)
        X_test_scaled = X_test_scaled.reshape(n_samples, t, f)

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        return {'y_true': y_test, 'y_pred': y_pred, 'MSE': mse, 'R2': r2, 'RMSE':rmse,'MAE':mae, 'MEDAE':medae}

    def predecir_futuro(self, model, historial_inicial, meses_a_predecir=12, ventana=3):
        """
        Predicción autoregresiva para datos 3D (samples, timesteps, features),
        pero la ventana y historial_inicial son 2D (ventana, features).
        """
        historial = list(historial_inicial)  # lista de ventanas con features
        predicciones = []

        for _ in range(meses_a_predecir):
            entrada = np.array(historial[-ventana:])  # (ventana, features)
            entrada_3d = entrada.reshape(1, ventana, self.features)

            # Escalar
            entrada_2d = entrada_3d.reshape(1, ventana * self.features)
            entrada_scaled_2d = self.scaler_X.transform(entrada_2d)
            entrada_scaled_3d = entrada_scaled_2d.reshape(1, ventana, self.features)

            pred_scaled = model.predict(entrada_scaled_3d)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]

            predicciones.append(pred)

            # Agregar nuevo valor para siguiente iteración (solo 1 feature, adapta si es multifeature)
            # Aquí asumimos 1 feature, si tienes más adapta para agregar vector completo
            historial.append([pred])

        return np.array(predicciones)


class HyperparameterLSTM_PSO:
    def __init__(self, timesteps, features):
        self.timesteps = timesteps
        self.features = features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.best_model = None

    def build_model(self, units=50, learning_rate=0.001):
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(self.timesteps, self.features)))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def _objective_function(self, params):
        n_particles = params.shape[0]
        losses = []
        for i in range(n_particles):
            units = int(params[i][0])
            learning_rate = params[i][1]

            units = max(1, units)
            learning_rate = max(1e-5, learning_rate)

            model = self.build_model(units=units, learning_rate=learning_rate)

            # Escalar datos de entrenamiento ya guardados
            model.fit(self.X_train_scaled, self.y_train_scaled, epochs=10, batch_size=32, verbose=0)

            y_pred = model.predict(self.X_train_scaled)
            loss = mean_squared_error(self.y_train_scaled, y_pred)
            losses.append(loss)

        return np.array(losses)

    def train(self, X_train, y_train, iters=30, swarm_size=20):
        n_samples, t, f = X_train.shape
        X_train_2d = X_train.reshape(n_samples, t * f)
        self.X_train_scaled = self.scaler_X.fit_transform(X_train_2d)
        self.X_train_scaled = self.X_train_scaled.reshape(n_samples, t, f)

        self.y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Parámetros PSO: [units, learning_rate]
        bounds = (np.array([10, 1e-4]), np.array([200, 0.1]))

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        optimizer = ps.single.GlobalBestPSO(
            n_particles=swarm_size,
            dimensions=2,
            options=options,
            bounds=bounds
        )

        best_cost, best_pos = optimizer.optimize(self._objective_function, iters)

        print("\n🧠 Mejores hiperparámetros encontrados por PSO (pyswarms):")
        print(f"  - units = {int(best_pos[0])}")
        print(f"  - learning_rate = {best_pos[1]:.6f}")

        self.best_model = self.build_model(units=int(best_pos[0]), learning_rate=best_pos[1])
        self.best_model.fit(self.X_train_scaled, self.y_train_scaled, epochs=20, batch_size=32, verbose=0)

        return self.best_model

    def evaluate(self, model, X_test, y_test):
        n_samples, t, f = X_test.shape
        X_test_2d = X_test.reshape(n_samples, t * f)
        X_test_scaled = self.scaler_X.transform(X_test_2d)
        X_test_scaled = X_test_scaled.reshape(n_samples, t, f)

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)
        return {'y_true': y_test, 'y_pred': y_pred, 'MSE': mse, 'R2': r2, 'RMSE':rmse,'MAE':mae, 'MEDAE':medae}

    def predecir_futuro(self, model, historial_inicial, meses_a_predecir=12, ventana=3):
        historial = list(historial_inicial)
        predicciones = []

        for _ in range(meses_a_predecir):
            entrada = np.array(historial[-ventana:])
            entrada_3d = entrada.reshape(1, ventana, self.features)

            entrada_2d = entrada_3d.reshape(1, ventana * self.features)
            entrada_scaled_2d = self.scaler_X.transform(entrada_2d)
            entrada_scaled_3d = entrada_scaled_2d.reshape(1, ventana, self.features)

            pred_scaled = model.predict(entrada_scaled_3d)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]

            predicciones.append(pred)
            historial.append([pred])

        return np.array(predicciones)
