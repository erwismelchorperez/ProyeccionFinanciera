from data_loader import FinancialDataLoader
from preprocessor import FinancialPreprocessor
from visualizer import FinancialVisualizer

from models.hyperparameter_dt import HyperparameterDT, HyperparameterDT_PSO
from models.hyperparameter_mlp import HyperparameterMLP, HyperparameterMLP_PSO
from models.hyperparameter_lasso import HyperparameterLasso, HyperparameterLasso_PSO
from models.hyperparameter_linear import HyperparameterLinear, HyperparameterLinear_PSO
#from models.hyperparameter_lstm import HyperparameterLSTM, HyperparameterLSTM_PSO
from models.hyperparameter_rf import HyperparameterRandomForest, HyperparameterRandomForest_PSO
from models.hyperparameter_ridge import HyperparameterRidge, HyperparameterRidge_PSO
#from models.hyperparameter_svr import HyperparameterSVR
from models.hyperparameter_xgb import HyperparameterXGBoost, HyperparameterXGBoost_PSO

import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split
# Diccionario para traducir meses de español a inglés abreviado
meses_map = {
    'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
    'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
}
# Ejecutar modelos
models = {
    "DT": HyperparameterDT(),
    "DTPSO": HyperparameterDT_PSO(),
    "MLP": HyperparameterMLP(),
    "MLPSO": HyperparameterMLP_PSO(),
    "Lasso": HyperparameterLasso(),
    "LassoPSO": HyperparameterLasso_PSO(),
    "Linear": HyperparameterLinear(),
    "LinearPSO": HyperparameterLinear_PSO(),
    #"LSTM": HyperparameterLSTM(timesteps=3, features=1),
    #"LSTMPSO": HyperparameterLSTM_PSO(timesteps=3, features=1),
    "RF": HyperparameterRandomForest(),
    "RFPSO": HyperparameterRandomForest_PSO(),
    "Ridge": HyperparameterRidge(),
    "RidgePSO": HyperparameterRidge_PSO(),
    #"SVR": HyperparameterSVR(),# este por el momento no se que hace se tarda mucho en el entrenamiento
    "XGBoost": HyperparameterXGBoost(),
    "XGBoostPSO": HyperparameterXGBoost_PSO()
}
financialdata = FinancialDataLoader('./dataset/Estados_FinancierosGaby.csv')
#financialdata = FinancialDataLoader('./dataset/Estados_Financieros.csv')
# Cargar datos
financialdata.load_data()
financialdata.ProcesarDataset()
df = financialdata.getDataset()
financialdata.SepararDatos()
Entrenamiento = financialdata.getEntrenamiento()
Pruebas = financialdata.getPruebas()
Validation = financialdata.getValidation()

##########################################################
cuenta_objetivo = 'Disponibilidades'#, Disponibilidades, CAJA, Cartera de crdito vigente ->archivo Gaby
#cuenta_objetivo = 'DISPONIBILIDADES'#, , CAJA
Entrenamiento = Entrenamiento[['FECHA', cuenta_objetivo]]
Pruebas = Pruebas[['FECHA', cuenta_objetivo]]
Validation = Validation[['FECHA', cuenta_objetivo]]
print(Entrenamiento[cuenta_objetivo])
fechaPrueba = Pruebas['FECHA']

Entrenamiento[cuenta_objetivo] = pd.to_numeric(Entrenamiento[cuenta_objetivo], errors='coerce')
Entrenamiento[cuenta_objetivo] = Entrenamiento[cuenta_objetivo].round(2)
Pruebas[cuenta_objetivo] = pd.to_numeric(Pruebas[cuenta_objetivo], errors='coerce')
Pruebas[cuenta_objetivo] = Pruebas[cuenta_objetivo].round(2)
Validation[cuenta_objetivo] = pd.to_numeric(Validation[cuenta_objetivo], errors='coerce')
Validation[cuenta_objetivo] = Validation[cuenta_objetivo].round(2)

# Definimos la ventana de meses para usar como input
flag_ventana = False  # <- CAMBIA AQUÍ según lo necesites
ventana = 3          # tamaño de la ventana

print(Entrenamiento)
usar_datos_3d = any('LSTM' in name for name in models.keys())
if flag_ventana:
    # Entrenamiento (2019-2021)
    serie_train = Entrenamiento[cuenta_objetivo].values
    X_train, y_train = financialdata.crear_dataset_supervisado(serie_train, ventana, reshape_3d=usar_datos_3d)

    # Para probar, usamos los 3 últimos valores de entrenamiento + primeros de test
    serie_completa = np.concatenate([Entrenamiento[cuenta_objetivo].values[-ventana:],
                                    Pruebas[cuenta_objetivo].values])
    X_test, y_test = financialdata.crear_dataset_supervisado(serie_completa, ventana, reshape_3d=usar_datos_3d)

    # datos para la validación
    serie_validation = Validation[cuenta_objetivo].values
    #serie_validation = financialdata.convertir_a_float_si_es_str(serie_validation, decimales=2, flag = flag_ventana)
    print("Serie Validation\n",serie_validation)
else:
    X_train = Entrenamiento[cuenta_objetivo]
    y_train = Entrenamiento[cuenta_objetivo]
    X_test = Pruebas[cuenta_objetivo]
    y_test = Pruebas[cuenta_objetivo]
    serie_validation = Validation[cuenta_objetivo].values
    print(serie_validation)
    #serie_validation = financialdata.convertir_a_float_si_es_str(serie_validation, decimales=2, flag = flag_ventana)
print("Xtrain               \n",X_train)
#X_train = financialdata.convertir_a_float_si_es_str(X_train, decimales=2, flag = flag_ventana)
#y_train = financialdata.convertir_a_float_si_es_str(y_train, decimales=2, flag = flag_ventana)
#X_test = financialdata.convertir_a_float_si_es_str(X_test, decimales=2, flag = flag_ventana)
#y_test = financialdata.convertir_a_float_si_es_str(y_test, decimales=2, flag = flag_ventana)

fechas_test = Pruebas['FECHA'].reset_index(drop=True)
if not flag_ventana:
    X_train = X_train.values.reshape(-1, 1)
    y_train = y_train.values.reshape(-1, 1)
    X_test = X_test.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)


model_scores = {}
viz = FinancialVisualizer()
rows = []
for name, model_obj in models.items():
    print(f"Entrenando {name}...")
    if 'PSO' in name:
        best_model = model_obj.train(X_train, y_train, iters=100, swarm_size=200)
        results = model_obj.evaluate(best_model, X_test, y_test)
    else:
        best_model = model_obj.train(X_train, y_train)
        results = model_obj.evaluate(best_model, X_test, y_test)

    model_scores[name] = results

    if not flag_ventana:
        y_test = y_test.reshape(-1).astype(float)
        model_scores[name]['y_true'] = model_scores[name]['y_true'].reshape(-1).astype(float)
    print("Real         2024:       ",y_test)
    print("Predicho     2024:       ",results['y_pred'])
    viz.plot_predictions(fechas_test, y_test, results['y_pred'], title=f"{name} - Real vs Predicho", save_path=f"plots/{name}_pred_vs_real.png")

    #############
    # Últimos 3 valores reales de diciembre, noviembre, octubre 2022
    if flag_ventana:
        historial_inicial = Pruebas[cuenta_objetivo].values[-3:]
    else:
        historial_inicial = Pruebas[cuenta_objetivo][-3:]
    #historial_inicial = financialdata.convertir_a_float_si_es_str(historial_inicial, decimales=2, flag = flag_ventana)

    # Asegurar formato 2D para LSTM
    if 'LSTM' in name and flag_ventana:
        historial_inicial = historial_inicial.reshape(ventana, 1)

    meses_a_predecir = 3
    if len(serie_validation) < meses_a_predecir:
        serie_validation = np.pad(serie_validation, (0, meses_a_predecir - len(serie_validation)), mode='constant', constant_values=0)
    # Predecimos 12 meses del año 2025
    pred_2023 = model_obj.predecir_futuro(best_model, historial_inicial, meses_a_predecir=meses_a_predecir, flag_ventana = flag_ventana)
    # Crear fechas para 2025
    fechas_2023 = pd.date_range(start='2025-01-01', periods=meses_a_predecir, freq='M').strftime('%b-%y')
    # Graficar
    #viz.plot_predictions(fechas_2023, [np.nan]*meses_a_predecir, pred_2023, title="Predicción 2023", save_path=f"plots/{name}_prediccion_2023.png")# esto es de manera general
    print("Validation           ",serie_validation, "       ", type(serie_validation))
    print("Prediction           ",pred_2023, "       ", type(pred_2023))
    rows.append(financialdata.PredichoRealDiferencia(name, serie_validation, pred_2023))
    # vamos a crear una nueva función para construir
    viz.plot_predictions(fechas_2023, serie_validation, pred_2023, title="Predicción 2025", save_path=f"plots/{name}_prediccion_2025.png")# esto es de la fecha del 2025, prediciendo los 3 primeros meses


print(model_scores)
viz.plot_multiple_predictions(fechas_test, y_test, model_scores, title="Modelos - Real vs Predicho", save_path="plots/comparacion_modelos.png")
viz.plot_model_errors(model_scores, save_path="plots/errores_comparados.png")
viz.plot_model_errors_boxplot(model_scores, save_path="plots/boxplot_errores_comparados.png")
########################
df = pd.DataFrame(rows)
df.to_csv("./plots/validacionmodelo.csv", index=False)

# Métricas finales
print("Resumen de métricas:")
for name, score in model_scores.items():
    print(f"{name}: MSE={score['MSE']:.2f}, R²={score['R2']:.3f}")
