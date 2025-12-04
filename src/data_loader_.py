from src.utils import read,aumentarDatoporMes,plot_cuenta_mensual_vs_diario,leer_variables
import matplotlib.pyplot as plt
import pandas as pd

class Loader:
    def __init__(self, filepath, variables_path=None):
        self.filepath=filepath
        self.filepath_variables=variables_path;
    def load_data(self):
        self.dataset=read(self.filepath);
        print(self.dataset.head())
        print(self.dataset.tail())
        self.dataset_aumentado=aumentarDatoporMes(self.dataset,21)
        # Ejemplo:
        #cuenta_obj = "30205"  # o 30205, etc.
        #plot_cuenta_mensual_vs_diario(self.dataset, self.dataset_aumentado, cuenta_obj)
    def load_data_variables(self):
        self.dataset_variables=leer_variables(self.filepath_variables)
    def select_variables(self,idx_cols: list[int]) ->pd.DataFrame:
        return self.dataset_variables.iloc[:, idx_cols].copy()
    def getDataset(self):
        return self.dataset
    def getDatasetVariables(self):
        return self.dataset_variables
    def getDatasetAumentado(self):
        return self.dataset_aumentado
        