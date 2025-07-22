import pandas as pd
import numpy as np
class FinancialDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
        self.dataset = any
        self.datasetfiltrado = any
        self.Entrenamiento = any
        self.EntrenamientoFinal = any
        self.Pruebas = any
        self.Validation = any

    def load_data(self):
        self.dataset = pd.read_csv(self.filepath)
    def ProcesarDataset(self):
        # Filtrar el dataframe con las cuentas con proyeccion 'SI' con las que vamos a entrenar los predictores de regresión
        self.datasetfiltrado = self.dataset[self.dataset['NIVEL'] == 2]# 2 para a nivel de disponibilidades, 3 para caja
        #self.datasetfiltrado = self.dataset[self.dataset['proyeccion'].str.strip().str.upper() == 'SI']
        # Mostrar cuántas filas cumplen con proyección == "SI"
        #print(f"Filas con proyección 'SI': {len(self.datasetfiltrado)}")

        # 2. Quitar la columna 'NIVEL', 'proyeccion', 'Codigo', ya que no son necesarias
        self.datasetfiltrado = self.datasetfiltrado.drop(columns=['NIVEL'])
        if 'proyeccion' in self.datasetfiltrado.columns:
            self.datasetfiltrado = self.datasetfiltrado.drop(columns=['proyeccion'])
        if 'Codigo' in self.datasetfiltrado.columns:
            self.datasetfiltrado = self.datasetfiltrado.drop(columns=['Codigo'])

        # 3. Transponer: queremos que las fechas sean el índice (filas)
        self.datasetfiltrado = self.datasetfiltrado.set_index('CUENTAS').T
        #self.datasetfiltrado = self.datasetfiltrado.set_index('BALANCE GENERAL').T #nuevo dataset 

        # 4. Opcional: limpiar nombres de columnas si hay espacios
        self.datasetfiltrado.columns = self.datasetfiltrado.columns.str.strip()

        # 5. Resetear el índice para que las fechas estén como columna
        self.datasetfiltrado = self.datasetfiltrado.reset_index().rename(columns={'index': 'FECHA'})
        #print(self.datasetfiltrado)
        self.FormatearFecha()
    def filtrarCuentasConDatosNumericos(self):
        '''Elimina las columnas (cuentas) que no tengan al menos un valor numerico válido (ignora '-') y se queda con las que tienen al menos un valor numerico'''
        # Convertir '-' en NaN
        self.datasetfiltrado.replace('-', np.nan, inplace=True)
        # Convertir todo lo que se pueda a numérico
        for col in self.datasetfiltrado.columns:
            if col != 'FECHA':
                #convierte cada valor de cada columna a valores numericos, pd.to_numeric()
                self.datasetfiltrado[col] = pd.to_numeric(self.datasetfiltrado[col], errors='coerce') #coerce fuerza a cada valor no numerico a convertirlo a NaN
        # Eliminar columnas con todos los valores NaN (sin datos válidos)
        columnas_validas = ['FECHA'] + [
            col for col in self.datasetfiltrado.columns
            if col != 'FECHA' and self.datasetfiltrado[col].notna().any()
        ]
        # Filtrar solo las columnas con al menos un valor numérico válido
        self.datasetfiltrado = self.datasetfiltrado[columnas_validas]
    def FormatearFecha(self):
        self.datasetfiltrado['FECHA'] = self.datasetfiltrado['FECHA'].apply(lambda x: self.formatearColumns(x))
        #print(self.datasetfiltrado)
    def formatearColumns(self, col):
        return col.replace('31-', '').replace('30-', '').replace('29-', '').replace('28-', '').replace('-', '')
    def SepararDatos(self):
        print("Fecha            ",self.datasetfiltrado['FECHA'].max())
        # Extraer todos los años únicos disponibles
        self.datasetfiltrado['AÑO'] = self.datasetfiltrado['FECHA'].str.extract(r'(\d{2})$')
        self.datasetfiltrado['AÑO'] = '20' + self.datasetfiltrado['AÑO']  # ejemplo: '19' -> '2019'
        self.datasetfiltrado['AÑO'] = self.datasetfiltrado['AÑO'].astype(int)

        # Detectar el último año automáticamente
        ultimo_año = self.datasetfiltrado['AÑO'].max()

        # Separar conjuntos
        self.Entrenamiento = self.datasetfiltrado[self.datasetfiltrado['AÑO'] < (ultimo_año - 1)]
        self.EntrenamientoFinal = self.datasetfiltrado[self.datasetfiltrado['AÑO'] < (ultimo_año)]
        self.Pruebas = self.datasetfiltrado[self.datasetfiltrado['AÑO'] == (ultimo_año - 1)]
        self.Validation = self.datasetfiltrado[self.datasetfiltrado['AÑO'] == ultimo_año]

        # (Opcional) eliminar columna auxiliar 'AÑO'
        self.datasetfiltrado.drop(columns=['AÑO'], inplace=True)
        """
        self.Entrenamiento = self.datasetfiltrado[self.datasetfiltrado['FECHA'].str.contains('19|20|21')]
        self.Pruebas = self.datasetfiltrado[self.datasetfiltrado['FECHA'].str.contains('22')]
        """
    def crear_dataset_supervisado(self, serie, ventana, reshape_3d=False):
        X, y = [], []
        for i in range(len(serie) - ventana):
            X.append(serie[i:i+ventana])
            y.append(serie[i+ventana])
        X = np.array(X)
        y = np.array(y)
        if reshape_3d:
            X = X.reshape((X.shape[0], X.shape[1], 1))  # 3D para LSTM
        return X, y
    def convertir_a_float_si_es_str(self, array, decimales = 2, flag= True):
        if flag:
            if np.issubdtype(array.dtype, np.str_) or isinstance(array[0][0], str):
                array = array.astype(float)
            else:
                array = array.astype(float)
        else:
            if np.issubdtype(array.dtype, np.str_):
                array = array.astype(float)
            else:
                array = array.astype(float)
        return np.round(array, decimales)
    def PredichoRealDiferencia(self, model, validacion, prediccion):
        diferencia = np.zeros(len(validacion))
        fila = {}
        fila['modelo'] = model
        for i in range(len(validacion)):
            diferencia[i] = validacion[i] - prediccion[i]
            fila['MR'+str(i)] = validacion[i]
            fila['MP'+str(i)] = prediccion[i]
            fila['diff'+str(i)] = diferencia[i]
        return fila
    def ProcesarDatosEntrenamientoPruebasValidaction(self, cuenta_objetivo, flag_ventana, ventana, usar_datos_3d):
        #para evitar sobreercribir en cada llamada de una cuenta objetivo nueva
        entrenamiento = self.Entrenamiento.copy()
        entrenamientoFinal = self.EntrenamientoFinal.copy()
        pruebas = self.Pruebas.copy()
        validation = self.Validation.copy()

        (X_train, y_train, X_trainFinal, y_trainFinal, X_test, y_test, serie_validation) = (None, None, None, None, None, None, None)
        entrenamiento[cuenta_objetivo] = pd.to_numeric(entrenamiento[cuenta_objetivo], errors='coerce')
        entrenamiento[cuenta_objetivo] = entrenamiento[cuenta_objetivo].round(2)
        entrenamientoFinal[cuenta_objetivo] = pd.to_numeric(entrenamientoFinal[cuenta_objetivo], errors='coerce')
        entrenamientoFinal[cuenta_objetivo] = entrenamientoFinal[cuenta_objetivo].round(2)
        pruebas[cuenta_objetivo] = pd.to_numeric(pruebas[cuenta_objetivo], errors='coerce')
        pruebas[cuenta_objetivo] = pruebas[cuenta_objetivo].round(2)
        validation[cuenta_objetivo] = pd.to_numeric(validation[cuenta_objetivo], errors='coerce')
        validation[cuenta_objetivo] = validation[cuenta_objetivo].round(2)
        print(self.Entrenamiento)
        usar_datos_3d = usar_datos_3d
        if flag_ventana:
            # Entrenamiento (2019-2021)
            serie_train = entrenamiento[cuenta_objetivo].values
            X_train, y_train = self.crear_dataset_supervisado(serie_train, ventana, reshape_3d=usar_datos_3d)
            serie_trainfinal = entrenamientoFinal[cuenta_objetivo].values
            X_trainFinal, y_trainFinal = self.crear_dataset_supervisado(serie_train, ventana, reshape_3d=usar_datos_3d)

            # Para probar, usamos los 3 últimos valores de entrenamiento + primeros de test
            serie_completa = np.concatenate([entrenamiento[cuenta_objetivo].values[-ventana:],
                                            pruebas[cuenta_objetivo].values])
            X_test, y_test = self.crear_dataset_supervisado(serie_completa, ventana, reshape_3d=usar_datos_3d)

            # datos para la validación
            serie_validation = self.Validation[cuenta_objetivo].values
            #serie_validation = self.convertir_a_float_si_es_str(serie_validation, decimales=2, flag = flag_ventana)
            print("Serie Validation\n",serie_validation)
        else:
            X_train = entrenamiento[cuenta_objetivo]
            y_train = entrenamiento[cuenta_objetivo]
            X_trainFinal = entrenamientoFinal[cuenta_objetivo]
            y_trainFinal = entrenamientoFinal[cuenta_objetivo]
            X_test = pruebas[cuenta_objetivo]
            y_test = pruebas[cuenta_objetivo]
            serie_validation = validation[cuenta_objetivo].values
            print(serie_validation)
            #serie_validation = self..convertir_a_float_si_es_str(serie_validation, decimales=2, flag = flag_ventana)
        return X_train, y_train, X_trainFinal, y_trainFinal, X_test, y_test, serie_validation
    # get/set
    def getDataset(self):
        return self.datasetfiltrado
    def getEntrenamiento(self):
        return self.Entrenamiento
    def getPruebas(self):
            return self.Pruebas
    def getValidation(self):
            return self.Validation
