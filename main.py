from src.data_loader import FinancialDataLoader #prepara datos, SI/NO
from src.preprocessor import FinancialPreprocessor
from src.visualizer import FinancialVisualizer

from src.models.hyperparameter_dt import HyperparameterDT, HyperparameterDT_PSO
from src.models.hyperparameter_mlp import HyperparameterMLP, HyperparameterMLP_PSO
from src.models.hyperparameter_lasso import HyperparameterLasso, HyperparameterLasso_PSO
from src.models.hyperparameter_linear import HyperparameterLinear, HyperparameterLinear_PSO
#from models.hyperparameter_lstm import HyperparameterLSTM, HyperparameterLSTM_PSO
from src.models.hyperparameter_rf import HyperparameterRandomForest, HyperparameterRandomForest_PSO
from src.models.hyperparameter_ridge import HyperparameterRidge, HyperparameterRidge_PSO
#from models.hyperparameter_svr import HyperparameterSVR
from src.models.hyperparameter_xgb import HyperparameterXGBoost, HyperparameterXGBoost_PSO
from src.models.ZeroInflatedPoissonWrapper import ZeroInflatedPoissonWrapper
from src.models.AlwaysZero import AlwaysZeroWrapper
from src.models.LigthGBM import LightGBM_TweedieWrapper
from src.models.TwoPart import TwoPartHurdleWrapper
from src.storage import crear_carpeta_cuenta, crear_carpeta_institucion, guardar_modelo #creamos carpeta y guardamos modelos en carpetas
from src.insertar_modelos import obtener_mapeo_codigos, insertar_modelo, get_connection
from src.zerosNozeros import infer_target_type
from src.choosemodels import choose_models
from src.all_zero import all_zero_short_circuit
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import psycopg
from sklearn.model_selection import train_test_split



def main(institucion: int, sucursal:int, templateid:int):
    #creamos la nueva carpeta institucion #
    # ruta raíz de salida para esta institucion
    suc_matriz,suc_dir,plots_dir=crear_carpeta_institucion(institucion,sucursal)

    #consulta
    codigo_to_id=obtener_mapeo_codigos(templateid) #codigos que pertenecen al templateid

    # Diccionario para traducir meses de español a inglés abreviado
    meses_map = {
        'ene': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'may': 'May', 'jun': 'Jun',
        'jul': 'Jul', 'ago': 'Aug', 'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dic': 'Dec'
    }
    # Ejecutar modelos
    models = {
        #"DT": HyperparameterDT(),
        #"DTPSO": HyperparameterDT_PSO(),
        #"MLP": HyperparameterMLP(),
        #"MLPSO": HyperparameterMLP_PSO(),
        "ZeroInflatedPoisson": ZeroInflatedPoissonWrapper(),
        "Lightgbm": LightGBM_TweedieWrapper(),
        "TwoPart": TwoPartHurdleWrapper(nonzero_label=True),
        "Lasso": HyperparameterLasso(),
        "LassoPSO": HyperparameterLasso_PSO(),
        "Linear": HyperparameterLinear(),
        "LinearPSO": HyperparameterLinear_PSO(),
        #"RF": HyperparameterRandomForest(),
        #"RFPSO": HyperparameterRandomForest_PSO(),
        "Ridge": HyperparameterRidge(),
        "RidgePSO": HyperparameterRidge_PSO(),
        #"XGBoost": HyperparameterXGBoost(),
        #"XGBoostPSO": HyperparameterXGBoost_PSO()

        # de aqui para abajo falta implementar
        #"SVR": HyperparameterSVR(),# este por el momento no se que hace se tarda mucho en el entrenamiento
        #"LSTM": HyperparameterLSTM(timesteps=3, features=1),
        #"LSTMPSO": HyperparameterLSTM_PSO(timesteps=3, features=1)
    }
    financialdata = FinancialDataLoader('./dataset/Estados_FinancierosGaby_proyeccion.csv')
    #financialdata = FinancialDataLoader('./dataset/Estados_FinancierosGaby.csv')
    #financialdata = FinancialDataLoader('./dataset/Estados_Financieros.csv')
    # Cargar datos
    financialdata.load_data()
    financialdata.ProcesarDataset() #FILTRA DATASET, TRASPUESTA, LIMPIA FILAS, FECHA MODO COLUMNA
    #financialdata.conservaSOLOdatosNUMERICOS();
    financialdata.reemplazaGuionPorCERO()
    df = financialdata.getDataset()
    financialdata.SepararDatos() #PREPARA DATASET ENTRENAMIENTO PRUEBA, VALIDACIÓN
    Entrenamiento = financialdata.getEntrenamiento()
    Pruebas = financialdata.getPruebas()
    Validation = financialdata.getValidation()

    ##########################################################
    #cuenta_objetivo = 'Inversiones en valores'#, Disponibilidades, CAJA, Cartera de crdito vigente ->archivo Gaby
    #cuenta_objetivo = 'DISPONIBILIDADES'#, , CAJA
    #Procesar cada cuenta objetivo
    #print([col for col in df.columns if col!='FECHA'])
    #target = "106010101" #102
    #cuentas_objetivo = [col for col in df.columns if col != "FECHA"]
    #if target in cuentas_objetivo:
    #    cuentas_objetivo = [target] + [c for c in cuentas_objetivo if c != target]
    
    cuentas_objetivo = [col for col in df.columns if col != 'FECHA']  #selecciona las cuentas objetivo con SI ya previamente procesadas}
    for cuenta_objetivo in cuentas_objetivo:
        print("CUENTA OBJETIVO--------------------")
        print(cuenta_objetivo)
        Entrenamiento_filtrado= Entrenamiento[['FECHA', cuenta_objetivo]]
        Pruebas_filtrado = Pruebas[['FECHA', cuenta_objetivo]]
        Validation_filtrado = Validation[['FECHA', cuenta_objetivo]]
        fechaPrueba_filtrado = Pruebas['FECHA']
        # Definimos la ventana de meses para usar como input
        flag_ventana = True  # <- CAMBIA AQUÍ según lo necesites
        ventana = 3          # tamaño de la ventana

        (X_train, y_train, X_trainFinal, y_trainFinal, X_test, y_test, serie_validation) = financialdata.ProcesarDatosEntrenamientoPruebasValidaction(cuenta_objetivo, flag_ventana, ventana, any('LSTM' in name for name in models.keys()))
        #print("Xtrain               \n",X_train)
        #X_train = financialdata.convertir_a_float_si_es_str(X_train, decimales=2, flag = flag_ventana)
        #y_train = financialdata.convertir_a_float_si_es_str(y_train, decimales=2, flag = flag_ventana)
        #X_test = financialdata.convertir_a_float_si_es_str(X_test, decimales=2, flag = flag_ventana)
        #y_test = financialdata.convertir_a_float_si_es_str(y_test, decimales=2, flag = flag_ventana)

        fechas_test = Pruebas_filtrado['FECHA'].reset_index(drop=True) #Pruebas_filtrado
        if not flag_ventana:
            X_train = X_train.values.reshape(-1, 1)
            y_train = y_train.values.reshape(-1, 1)
            X_trainFinal = X_trainFinal.values.reshape(-1, 1)
            y_trainFinal = y_trainFinal.values.reshape(-1, 1)
            X_test = X_test.values.reshape(-1, 1)
            y_test = y_test.values.reshape(-1, 1)


            # ---- NUEVO: calcular proporción de ceros ----
        zero_ratio = (y_train == 0).sum() / len(y_train)
        print(cuenta_objetivo)
        print(f"Proporción de ceros en {cuenta_objetivo}: {zero_ratio:.2%}")
        # --- Dentro de tu loop por cuenta ---
        zr = float((y_train == 0).mean())
        has_neg = bool((y_train < 0).any())
        target_type = infer_target_type(y_train)

        print(f"[{cuenta_objetivo}] tipo={target_type} | ceros={zr:.1%} | negativos={has_neg}")


        # Selección simple y robusta:
        if all_zero_short_circuit(y_train):
            print(f"[{cuenta_objetivo}] TODO CERO en train → usar AlwaysZero y saltar el resto.")
            modelos_a_entrenar = {"AlwaysZero": AlwaysZeroWrapper()}
        else:
            modelos_a_entrenar = choose_models(models, y_train, cuenta_objetivo)

        #modelos_a_entrenar = choose_models(models, y_train, cuenta_objetivo)
        # Si tiene más del 70% de ceros -> usar SOLO ZeroInflatedPoisson
        #if zero_ratio > 0.7:
        #    modelos_a_entrenar = {"ZeroInflatedPoisson": models["ZeroInflatedPoisson"]}
        #else:
        #    modelos_a_entrenar = models  # usa todos los modelos definidos

        # ---- Entrenamiento ----
        
        model_scores = {}
        viz = FinancialVisualizer()
        rows = []
        tempmodels = {}
        fitted_wrappers = {}
        for name, model_obj in modelos_a_entrenar.items():
            try:
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

                """
                    Aqui vamos a reentrenar los modelos, para ellos vamos a utilizar los modelos hasta el año 2024
                """
                print(best_model)
                if 'PSO' in name:
                    best_model = model_obj.train(X_trainFinal, y_trainFinal, iters=100, swarm_size=200)
                else:
                    best_model = model_obj.train(X_trainFinal, y_trainFinal)
                print(best_model)

                #############
                # Últimos 3 valores reales de diciembre, noviembre, octubre 2022
                if flag_ventana:
                    #historial_inicial = Pruebas[cuenta_objetivo].values[-3:]
                    historial_inicial = Pruebas_filtrado[cuenta_objetivo].values[-3:]
                else:
                    #historial_inicial = Pruebas[cuenta_objetivo][-3:]
                    historial_inicial = Pruebas_filtrado[cuenta_objetivo][-3:]
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
                fechas_2023 = pd.date_range(start='2025-01-01', periods=meses_a_predecir, freq='ME').strftime('%b-%y')
                # Graficar
                #viz.plot_predictions(fechas_2023, [np.nan]*meses_a_predecir, pred_2023, title="Predicción 2023", save_path=f"plots/{name}_prediccion_2023.png")# esto es de manera general
                print("Validation           ",serie_validation, "       ", type(serie_validation))
                print("Prediction           ",pred_2023, "       ", type(pred_2023))
                rows.append(financialdata.PredichoRealDiferencia(name, serie_validation, pred_2023))
                # vamos a crear una nueva función para construir
                viz.plot_predictions(fechas_2023, serie_validation, pred_2023, title="Predicción 2025", save_path=f"plots/{name}_prediccion_2025_"+str(flag_ventana)+".png",show=False)# esto es de la fecha del 2025, prediciendo los 3 primeros meses
                # >>> MUY IMPORTANTE: fija el modelo interno en el wrapper
                # (Para todos tus wrappers ya migrados a Opción B: TwoPart, LightGBM, LinearPSO, etc.)
                if hasattr(model_obj, "model"):
                    model_obj.model = best_model
                tempmodels[name] = best_model
                fitted_wrappers[name] = model_obj      # <-- wrapper entrenado (EL QUE DEBES SERIALIZAR)
                # (Opcional) guarda la “ventana” usada dentro del wrapper
                if hasattr(model_obj, "default_ventana"):
                    model_obj.default_ventana = ventana
                if hasattr(model_obj, "default_flag_ventana"):
                    model_obj.default_flag_ventana = flag_ventana
            except ValueError as e:
                print(f"Saltando {name}: {e}")
                continue


        # apartir de aqui vamos a guardar los primeros 3 modelos ordenados de menor a mayor
        print(tempmodels)
        print(model_scores)
        # Crear la carpeta (incluye subcarpetas si no existen)
        #crear subcarpeta para cada cuenta dentro de la carpeta de la institucion
        cuenta_dir_matriz=crear_carpeta_cuenta(suc_matriz,cuenta_objetivo)
        if sucursal!=0:
            cuenta_dir=crear_carpeta_cuenta(suc_dir,cuenta_objetivo)

        # Ordenar por R2 de mayor a menor
        sorted_by_r2 = dict(sorted(model_scores.items(), key=lambda x: x[1]['R2'], reverse=True))
        print(sorted_by_r2)


        rank=0
        # Mostrar resultados ordenados
        top_k = 3
        # Toma hasta TOP_K, pero no asume que siempre hay 3
        seleccion = list(sorted_by_r2.items())[:min(top_k, len(sorted_by_r2))]
        i=0;
        for rank, (name, metrics) in enumerate(seleccion, start=1):
        #for i, (model, metrics) in enumerate(sorted_by_r2.items()):
            if i >= 3:
                break
            print(f"{name}: R2 = {metrics['R2']:.4f}")
            #nombre_modelo=f"{model}_{templateid}_{cuenta_objetivo}.pkl"
            nombre_modelo=f"{name}_{templateid}_{cuenta_objetivo}.joblib"
            obj = fitted_wrappers.get(name, tempmodels.get(name))
            if obj is None:
                print(f"  [WARN] {name} no tiene wrapper ni modelo interno disponible; se omite.")
                continue

            #joblib.dump(wrapper_entrenado, nombre_modelo, compress=3)
            # Guardar cada modelo
            # ruta completa del archivo en matriz
            ruta_modelo_matriz = os.path.join(cuenta_dir_matriz, nombre_modelo)
            #guardar_modelo(tempmodels[model],os.path.join(cuenta_dir_matriz,nombre_modelo)) #matriz
            joblib.dump(obj, ruta_modelo_matriz, compress=3)
            if sucursal!=0:
                ruta_sucursal = os.path.join(cuenta_dir, nombre_modelo)
                #guardar_modelo(templateid[model],os.path.join(cuenta_dir,nombre_modelo)); #sucursal particula
                joblib.dump(obj, ruta_sucursal, compress=3)
            # Dentro del loop donde guardas modelos
            cuentaid = codigo_to_id.get(cuenta_objetivo)
            nombre_modelo_bd=f"modelo{rank}_{templateid}_{cuenta_objetivo}"
            if cuentaid:
                insertar_modelo(
                    cuentaid=cuentaid,
                    modelo=nombre_modelo_bd,
                    ubicacion=ruta_modelo_matriz
                )
            i=i+1;

        #viz.plot_multiple_predictions(fechas_test, y_test, model_scores, title="Modelos - Real vs Predicho", save_path="plots/comparacion_" + cuenta_objetivo + "_modelos_"+str(flag_ventana)+".png")
        
        viz.plot_multiple_predictions(
            fechas_test, y_test, model_scores,
            title="Modelos - Real vs Predicho",
            save_path=os.path.join(plots_dir, f"comparacion_{cuenta_objetivo}_modelos_template_{templateid}_{flag_ventana}.png"),
            show=False
        )
        #viz.plot_model_errors(model_scores, save_path="plots/errores_" + cuenta_objetivo + "_comparados_"+str(flag_ventana)+".png")
        viz.plot_model_errors(
            model_scores,
            save_path=os.path.join(plots_dir, f"errores_{cuenta_objetivo}_comparados_template_{templateid}_{flag_ventana}.png")
        )
        #viz.plot_model_errors_boxplot(model_scores, save_path="plots/boxplot_" + cuenta_objetivo + "_errores_comparados_"+str(flag_ventana)+".png")
        viz.plot_model_errors_boxplot(
            model_scores,
            save_path=os.path.join(plots_dir, f"boxplot_template_{templateid}_{cuenta_objetivo}_errores_comparados_{flag_ventana}.png")
        )

        ########################
        df = pd.DataFrame(rows)
        #df.to_csv("./plots/validacionmodelo_" + cuenta_objetivo + "_"+str(flag_ventana)+".csv", index=False)
        df.to_csv(os.path.join(plots_dir, f"validacionmodelo_template_{templateid}_{cuenta_objetivo}_{flag_ventana}.csv"), index=False)
        # Métricas finales
        print("Resumen de métricas:")
        for name, score in model_scores.items():
            print(f"{name}: MSE={score['MSE']:.2f}, R²={score['R2']:.3f}")
            
    
if __name__ == "__main__":
    # Validar que se pase un argumento entero
    if len(sys.argv) != 4:
        print("Este programa requiere tres enteros (institucionid, sucursalid, templateid): <enteros> <enteros> <enteros>")
        sys.exit(1)

    try:
        institucion = int(sys.argv[1])
        sucursal = int(sys.argv[2])
        templateid = int(sys.argv[3])
    except ValueError:
        print("El parámetro debe ser un número entero.")
        sys.exit(1)
    root_dir = f"./instituciones"
    os.makedirs(root_dir, exist_ok=True)  # crea ./instituciones/ si no existe
    # Llamar a la función principal
    main(institucion,sucursal,templateid)