"""
import tensorflow as tf, sklearn, scikeras
print("TF:", tf.__version__,
        "SKL:", sklearn.__version__,
        "Scikeras:", scikeras.__version__)
"""
import numpy as np
def PredichoRealDiferencia(model, validacion, prediccion):
    diferencia = np.zeros(len(validacion))
    fila = {}
    fila['modelo'] = model
    for i in range(len(validacion)):
        diferencia[i] = validacion[i] - prediccion[i]
        fila['MR'+str(i)] = validacion[i]
        fila['MP'+str(i)] = prediccion[i]
        fila['diff'+str(i)] = diferencia[i]


validacion = np.array([24108153, 19799516,22218738])
prediccion = np.array([19284923.,19284923.,19284923.])
model = 'DT'
PredichoRealDiferencia(model,validacion, prediccion)
