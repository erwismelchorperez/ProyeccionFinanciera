from src.utils import read,aumentarDatoporMes,plot_cuenta_mensual_vs_diario
import matplotlib.pyplot as plt


class Loader:
    def __init__(self,filepath):
        self.filepath=filepath;
    def load_data(self):
        self.dataset=read(self.filepath);
        print(self.dataset.head())
        print(self.dataset.tail())
        self.dataset_aumentado=aumentarDatoporMes(self.dataset,21)
        # Ejemplo:
        #cuenta_obj = "30205"  # o 30205, etc.
        #plot_cuenta_mensual_vs_diario(self.dataset, self.dataset_aumentado, cuenta_obj)
    def getDataset(self):
        return self.dataset
    def getDatasetAumentado(self):
        return self.dataset_aumentado
        