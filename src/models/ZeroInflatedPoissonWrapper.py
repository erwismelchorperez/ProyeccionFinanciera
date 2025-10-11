import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.preprocessing import StandardScaler


class ZeroInflatedPoissonWrapper:
    def __init__(self):
        self.model = None
        self.result = None
        self.scaler_X = StandardScaler()

    def train(self, X_train, y_train, **kwargs):
        # Escalar X_train
        X_train = self.scaler_X.fit_transform(X_train)
        y_train = np.asarray(y_train).astype(int)

        X_train = sm.add_constant(X_train, has_constant='add')

        self.model = sm.ZeroInflatedPoisson(y_train, X_train, exog_infl=X_train, inflation='logit')
        self.result = self.model.fit(disp=0)
        return self.result


    def evaluate(self, model, X_test, y_test):
        # Aplicar el mismo escalador
        X_test = self.scaler_X.transform(X_test)
        X_test = sm.add_constant(X_test, has_constant='add')

        y_pred = model.predict(exog=X_test, exog_infl=X_test)

        # Evitar NaN/Inf
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=0.0)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        return {
            'y_true': np.asarray(y_test),
            'y_pred': y_pred,
            'MSE': mse,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MEDAE': medae
        }

    def predecir_futuro(self, modelo, historial_inicial, meses_a_predecir=12, ventana=3, flag_ventana=True):
        """
        Predice valores futuros de forma autoregresiva usando ZeroInflatedPoisson.
        
        modelo: modelo entrenado (self.result)
        historial_inicial: últimos 'ventana' valores conocidos
        meses_a_predecir: número de pasos a predecir
        ventana: tamaño de la ventana temporal
        """
        historial = list(map(float, historial_inicial))  

        predicciones = []

        for _ in range(meses_a_predecir):
            if flag_ventana:
                entrada = np.array(historial[-ventana:], dtype=float).reshape(1, -1)
            else:
                entrada = np.array([historial[-1]], dtype=float).reshape(1, -1)

            # Escalar entrada
            entrada = self.scaler_X.transform(entrada)

            #  Agregar constante
            entrada = sm.add_constant(entrada, has_constant='add')

            # Predecir
            prediccion = modelo.predict(exog=entrada, exog_infl=entrada)[0]

            # Guardar predicción y actualizar historial
            predicciones.append(prediccion)
            historial.append(prediccion)

        return np.array(predicciones)
