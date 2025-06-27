import pandas as pd
import numpy as np
class FinancialDataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.meses = ['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']
        self.dataset = any
        self.datasetfiltrado = any
        self.Entrenamiento = any
        self.Pruebas = any
        self.Validation = any

    def load_data(self):
        self.dataset = pd.read_csv(self.filepath)
    def ProcesarDataset(self):
        # Filtrar el dataframe con las cuentas con las que vamos a entrenar los predictores de regresión
        self.datasetfiltrado = self.dataset[self.dataset['NIVEL'] == 2]# 2 para a nivel de disponibilidades, 3 para caja
        # 2. Quitar la columna 'NIVEL' ya que ya no es necesaria
        self.datasetfiltrado = self.datasetfiltrado.drop(columns=['NIVEL'])

        # 3. Transponer: queremos que las fechas sean el índice (filas)
        self.datasetfiltrado = self.datasetfiltrado.set_index('CUENTAS').T

        # 4. Opcional: limpiar nombres de columnas si hay espacios
        self.datasetfiltrado.columns = self.datasetfiltrado.columns.str.strip()

        # 5. Resetear el índice para que las fechas estén como columna
        self.datasetfiltrado = self.datasetfiltrado.reset_index().rename(columns={'index': 'FECHA'})
        #print(self.datasetfiltrado)
        self.FormatearFecha()
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

    # get/set
    def getDataset(self):
        return self.datasetfiltrado
    def getEntrenamiento(self):
        return self.Entrenamiento
    def getPruebas(self):
            return self.Pruebas
    def getValidation(self):
            return self.Validation
