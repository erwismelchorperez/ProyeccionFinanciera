import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
import time
import os
import matplotlib.pyplot as plt
from src.storage import crear_carpeta_cuenta,crear_carpeta_institucion,guardar_modelo
from src.insertar_modelos import obtener_mapeo_codigos, insertar_modelo, get_connection
#from src.db_mapeos import obtener_mapeo_codigos

def run(institucion: int,sucursal:int,templateid:int):
    suc_matriz,suc_dir,plots_dir=crear_carpeta_institucion(institucion,sucursal)
    #consulta
    print("INICIO")
    codigo_to_id=obtener_mapeo_codigos(templateid) #codigos que pertenecen al templateid
    print("→ mapeo listo, total códigos:", len(codigo_to_id))
    print("---------------------------------------")
    from src.data_loader_ import Loader
    #from src.utils import read,aumentar_columna_por_mes,putTest_cuenta,all_zero,tiene_negativos,ceros_iniciales,choose_models,aumentar_columna_por_mes_saltando_ceros_iniciales,splitsTrainTest,splitsTrainTest_from_series,plot_resultados_modelos,plot_modelos_alineados,plot_serie_completa_con_model_scores,normalizar_resultado_para_export
    from src.utils import (
        read,
        aumentar_columna_por_mes,
        putTest_cuenta,
        all_zero,
        tiene_negativos,
        ceros_iniciales,
        choose_models,
        aumentar_columna_por_mes_saltando_ceros_iniciales,
        splitsTrainTest,
        splitsTrainTest_from_series,
        plot_resultados_modelos,
        plot_modelos_alineados,
        plot_serie_completa_con_model_scores,
        normalizar_resultado_para_export,
    )
    from sklearn.metrics import mean_squared_error
    from src.new_models.LSTM import LSTMWrapper
    from src.new_models.TCN import TCNWrapper
    from src.new_models.Linearmodel import HyperparameterLinear,HyperparameterLinear_PSO
    from src.new_models.AlwaysZero import AlwaysZeroWrapper
    from src.new_models.Ridge import HyperparameterRidge,HyperparameterRidge_PSO
    from src.new_models.ZeroInflatedPoissonWrapper import ZeroInflatedPoissonWrapper
    from src.new_models.Lasso import HyperparameterLasso,HyperparameterLasso_PSO
    from src.new_models.MLP import MLPSeriesWrapper
    from src.new_models.TwoPart import TwoPartHurdleWrapper
    from src.new_models.LightGBM import LightGBM_TweedieSeriesWrapper
    from src.resumen import exportar_predicciones_y_resumen,exportar_predicciones_y_resumen_solo_mejor
    models = {
    #"ZeroInflatedPoisson": ZeroInflatedPoissonWrapper(),
    "Lightgbm": LightGBM_TweedieSeriesWrapper(),
    "TwoPart": TwoPartHurdleWrapper(nonzero_label=True),
    "Lasso": HyperparameterLasso(),
    "LassoPSO": HyperparameterLasso_PSO(),
    "Linear": HyperparameterLinear(),
    "LinearPSO": HyperparameterLinear_PSO(),
    "Ridge": HyperparameterRidge(),
    "RidgePSO": HyperparameterRidge_PSO(),
    "LSTM": LSTMWrapper(),
    "TCN": TCNWrapper(),     #
    "MLP": MLPSeriesWrapper()
    }
    proyeccionFinanciera=Loader("./dataset/dataset_con_proyeccion.csv")
    proyeccionFinanciera.load_data()
    #cuentas=[col for col in self.dataset_aumentado.columns if col!='date']
    dataset=proyeccionFinanciera.getDataset()
    df_num = dataset.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    # rangos fijos que quieres usar
    TRAIN_START = pd.Timestamp("2013-01-01")
    TRAIN_END   = pd.Timestamp("2024-01-01")

    # predecir desde 2024-02 hasta 2025-03 → son 15 meses
    TEST_START = pd.Timestamp("2024-02-01")
    TEST_END   = pd.Timestamp("2025-03-01")
    FUTURE_H=15
    df_mensual = df_num.copy()
    if not isinstance(df_mensual.index, pd.DatetimeIndex):
        df_mensual.index = pd.to_datetime(df_mensual.index)
    df_mensual = df_mensual.asfreq("MS").sort_index()  # mensual (Month Start)
    forecast_end = dt.datetime(2025, 3, 1)  # predecir hasta 2025-03 (inclusive)

    # todas las cuentas:
    cols = df_mensual.columns.astype(str).tolist()
    # Si solo una para probar:
    # cols = ['101']
    iter=0;
    s_map = {}
    splits = {}
    train_ratio = 0.8
    output_xlsx ="predicciones_V3.xlsx"
    for col in cols:
        # 1) inicializa los contenedores PARA ESTA CUENTA
        model_scores     = {}   # métricas de cada modelo
        fitted_wrappers  = {}   # el wrapper entrenado
        tempmodels       = {}   # modelo crudo (sklearn/keras)
        #col = str(101)#col = str(106010102) #108 #103
        col = str(col)
        if col not in df_mensual.columns.astype(str).tolist():
            print(f"'{col}' no existe, salto.")
            continue

        # 2) serie mensual de esa cuenta
        serie_mensual = df_mensual[col].astype(float)
        real_cuenta = df_mensual[[col]].rename(columns={col: "real"})
        # 3) validaciones sobre la serie ORIGINAL (mensual)
        all_zero_val   = all_zero(serie_mensual)
        has_neg    = tiene_negativos(serie_mensual)
        lead_count = ceros_iniciales(serie_mensual)
        zero_ratio = (serie_mensual == 0).mean()
        

        print(f"\nCuenta {col}:")
        print(f"  - todo_cero?            {all_zero_val}")
        print(f"  - tiene_negativos?      {has_neg}")
        print(f"  - proporción_0_inicial: {lead_count:.2%}")

        # 4) todo cero → no aumento
        if all_zero_val:
            print(f"[{col}] toda la serie es 0 → no aumento, usar AlwaysZero o saltar.")

            # usa el DF (real_cuenta) para sacar el tramo
            real_tramo_df = real_cuenta.loc[TEST_START:TEST_END].copy()
            if real_tramo_df.empty:
                # si no hay ese rango, toma últimos 15 meses del DF
                real_tramo_df = real_cuenta.iloc[-15:].copy()

            # ahora sí, esto es un DF con col "real"
            y_true = real_tramo_df["real"].values.astype(float)
            y_pred = np.zeros_like(y_true, dtype=float)
            idx    = real_tramo_df.index

            mse  = float(np.mean((y_true - y_pred) ** 2))
            rmse = float(np.sqrt(mse))

            model_scores["AlwaysZero"] = {
                "y_true": y_true,
                "y_pred": y_pred,
                "MSE": mse,
                "RMSE": rmse,
                "cuenta": col,
                "modelo": "AlwaysZero",
                "idx": idx,
            }

            exportar_predicciones_y_resumen_solo_mejor(
                cuenta=col,
                real_full=real_cuenta,            # DF con columna 'real'
                model_scores=model_scores,
                output_path=output_xlsx,
                meses_objetivo=15,
                real_desde="2024-02-01",
            )
            # pasar a la siguiente cuenta
            continue

        # a) tiene ceros al inicio → aumentar solo después de esos ceros
        if lead_count > 0.6:
            serie_diaria = aumentar_columna_por_mes_saltando_ceros_iniciales(
                df_mensual, col=col,
                n_dias_agregar=21,
                incluir_original=True,
                solo_impares=True
            )
        else:
            # b) caso normal → aumentar todo
            serie_diaria = aumentar_columna_por_mes(
                df_mensual, col=col,
                n_dias_agregar=21,
                incluir_original=True,
                solo_impares=True
            )

        # esta es la que usarán los modelos “densos”
        x_diaria = serie_diaria.to_frame(name=col)
        print(x_diaria)
        
        modelos_a_entrenar = choose_models(
            models,
            all_zero_val=all_zero_val,
            has_neg=has_neg,
            lead_zero_ratio=lead_count,
            zero_ratio=zero_ratio,
            lead_thresh=0.60,
            high_zero_thresh=0.50
        )
        print(modelos_a_entrenar)
        
        # 5) entrenar modelo por modelo
        # ---------------------------------------------------------------------
        for name, model_obj in modelos_a_entrenar.items():
            try:
                print(f"\n=== Entrenando {name} para cuenta {col} ===")

                # ------------------------------------------------------------
                # CASO A: modelo “de serie” (TCNWrapper o el que tú hiciste)
                # ------------------------------------------------------------
                if hasattr(model_obj, "train_from_series"):
                    # le pasamos la serie DIARIA porque ya hiciste el aumento
                    t0 = time.perf_counter()
                    results = model_obj.train_from_series(
                        x_diaria,
                        train_start=TRAIN_START,
                        train_end=TRAIN_END
                    )
                    t1 = time.perf_counter()
                    train_time = t1 - t0

                    # results ya trae y_true / y_pred / MSE / RMSE 
                    results = {
                        **results,
                        "train_time": train_time,
                        "cuenta": col,
                        "modelo": name,
                    }
                    norm = normalizar_resultado_para_export(
                        results,
                        real_cuenta,
                        meses_objetivo=15
                    )
                    results["norm"] = norm

                    model_scores[name] = results

                    #model_scores[name] = results

                    # guarda el wrapper ya entrenado
                    fitted_wrappers[name] = model_obj

                    # para serializar el modelo keras dentro:
                    if hasattr(model_obj, "model"):
                        tempmodels[name] = model_obj.model

                    # predicción futura 15 meses (si la clase la tiene)
                    if hasattr(model_obj, "predecir_futuro"):
                        try:
                            fut = model_obj.predecir_futuro(
                                meses_a_predecir=FUTURE_H,
                                ventana=getattr(model_obj, "default_ventana", 3),
                                flag_ventana=getattr(model_obj, "default_flag_ventana", True),
                            )
                            results["future_pred"] = fut
                        except Exception as e:
                            print(f"{name}: no pude predecir futuro: {e}")

                # ------------------------------------------------------------
                # CASO B: modelo clásico con lags
                # ------------------------------------------------------------
                else:
                    # 1) lags + split por FECHA fijo 2013-01 → 2024-01
                    X_train, y_train, X_test, y_test, train_s, test_s = splitsTrainTest_from_series(
                        serie_mensual,
                        n_lags=3,
                        train_start=TRAIN_START,
                        train_end=TRAIN_END
                    )

                    if len(X_train) == 0:
                        print(f"{name}: sin datos suficientes para entrenar → salto.")
                        continue

                    # 2) entrenar
                    t0 = time.perf_counter()
                    if "PSO" in name:
                        best_model = model_obj.train(X_train, y_train, iters=100, swarm_size=200)
                    else:
                        best_model = model_obj.train(X_train, y_train)
                    t1 = time.perf_counter()
                    train_time = t1 - t0

                    # 3) evaluar en el tramo > 2024-01 que te quedó
                    if len(X_test) > 0:
                        results = model_obj.evaluate(best_model, X_test, y_test)
                    else:
                        # no hay test
                        results = {
                            "y_true": y_test,
                            "y_pred": np.array([]),
                            "MSE": np.nan,
                            "RMSE": np.nan,
                        }

                    # 4) guardar resultados + tiempos + metadatos
                    results["train_time"] = train_time
                    results["cuenta"]     = col
                    results["modelo"]     = name
                    norm = normalizar_resultado_para_export(
                        results,
                        real_cuenta,
                        meses_objetivo=15
                    )
                    results["norm"] = norm

                    model_scores[name] = results
                    #model_scores[name]    = results

                    # 5) guardar modelo
                    if hasattr(model_obj, "model"):
                        model_obj.model = best_model
                    fitted_wrappers[name] = model_obj
                    tempmodels[name]      = best_model

                    # 6) predicción futura 15 meses (2024-02 → 2025-03)
                    if hasattr(model_obj, "predecir_futuro"):
                        # último historial lo tomamos del último valor conocido (enero 2024)
                        historial_inicial = serie_mensual.loc[:TRAIN_END].values[-3:]
                        try:
                            fut = model_obj.predecir_futuro(
                                best_model,
                                historial_inicial=historial_inicial,
                                meses_a_predecir=FUTURE_H,
                                ventana=3,
                                flag_ventana=True
                            )
                            results["future_pred"] = fut
                        except Exception as e:
                            print(f"{name}: no pude predecir futuro: {e}")
                
            except ValueError as e:
                print(f"Saltando {name} en {col}: {e}")
                continue
            except Exception as e:
                print(f"Error inesperado en {name} / {col}: {e}")
                continue
        print(tempmodels)
        print(model_scores)
        if len(model_scores) > 0:
            plot_resultados_modelos(
                model_scores,
                titulo=f"Cuenta {col} — real vs modelos"
            )
        if len(model_scores) > 0:
            # real mensual completa de esa cuenta
            real_full = df_mensual[[col]].rename(columns={col: "Adj Close"})
            exportar_predicciones_y_resumen_solo_mejor(
                cuenta=col,
                real_full=real_cuenta,
                model_scores=model_scores,
                output_path=output_xlsx,
            )

            # si tienes el index del tramo de test (del TCN o del split)
            # por ejemplo si entrenaste TCNWrapper y lo guardaste:
            # test_index = tcn_wrapper.test_df.index
            # si no, puedes pasar None
            plot_serie_completa_con_model_scores(
                real_full=real_full,
                model_scores=model_scores,
                test_index=None,   # o el que sí tengas
                titulo=f"Cuenta {col} — real vs modelos",
                plots_dir=plots_dir,                  # ← tu ruta existente
                filename=f"cuenta_{col}.png",         # opcional
            )
        cuenta_dir_matriz=crear_carpeta_cuenta(suc_matriz,col)
        if sucursal!=0:
            cuenta_dir=crear_carpeta_cuenta(suc_dir,col)

        ranked = sorted(
            model_scores.items(),
            key=lambda kv: (
                kv[1].get("RMSE", float("inf")),
                kv[1].get("MAE",  float("inf")),
            )
        )

        top_k = 3
        seleccion = ranked[:top_k]

        for rank, (name, metrics) in enumerate(seleccion, start=1):
            r2 = metrics.get("R2")  # puede ser None
            if r2 is not None and not np.isnan(r2):
                print(f"#{rank} {name}: R2 = {r2:.4f}")
            else:
                print(f"#{rank} {name}: R2 = N/A")

            nombre_modelo = f"{name}_{templateid}_{col}.joblib"

            obj = fitted_wrappers.get(name) or tempmodels.get(name)
            if obj is None:
                print(f"  [WARN] {name} no tiene wrapper/modelo disponible; se omite.")
                continue

            ruta_modelo_matriz = os.path.join(cuenta_dir_matriz, nombre_modelo)
            joblib.dump(obj, ruta_modelo_matriz, compress=3)

            if sucursal != 0:
                ruta_sucursal = os.path.join(cuenta_dir, nombre_modelo)
                joblib.dump(obj, ruta_sucursal, compress=3)
            
            cuentaid = codigo_to_id.get(col)
            nombre_modelo_bd=f"modelo{rank}_{templateid}_{col}"
            #print(col)
            #print(cuentaid)
            if cuentaid:
                insertar_modelo(
                    cuentaid=cuentaid,
                    sucursalid=sucursal,
                    modelo=nombre_modelo_bd,
                    ubicacion=ruta_modelo_matriz
                )
            else:
                print("NO SE INSERTO EN BD")
        break

if __name__=="__main__":
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
    run(institucion,sucursal,templateid)
