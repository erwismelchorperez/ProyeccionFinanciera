import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class FinancialVisualizer:
    @staticmethod
    def plot_predictions(fechas, y_true, y_pred, title='Predicción vs Real', save_path=None):
        # Convertir a arrays si son Series
        if isinstance(y_true, (pd.Series, list)): y_true = np.array(y_true)
        if isinstance(y_pred, (pd.Series, list)): y_pred = np.array(y_pred)

        plt.figure(figsize=(14, 6))
        plt.plot(fechas, y_pred, label='Predicho', marker='x', linestyle='--', color='orange')
        plt.plot(fechas, y_true, label='Real', marker='o', color='blue', linewidth=2)
        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    @staticmethod
    def plot_multiple_predictions(fechas_test, y_test, results_dict, title="Comparación de modelos", save_path=None):
        """
        results_dict: diccionario con estructura {'NombreModelo': {'y_pred': predicciones, ...}, ...}
        """
        plt.figure(figsize=(14, 6))
        plt.plot(fechas_test, y_test, label="Real", color="black", linewidth=2)

        # Colores distintos para cada modelo
        colors = plt.cm.get_cmap('tab10', len(results_dict))

        for idx, (model_name, result) in enumerate(results_dict.items()):
            plt.plot(fechas_test, result['y_pred'], label=model_name, linestyle='--', color=colors(idx))

        plt.title(title)
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"📈 Gráfico guardado en {save_path}")
        plt.show()
    @staticmethod
    def plot_model_errors(model_scores, save_path=None):
        names = list(model_scores.keys())
        mses = [score['MSE'] for score in model_scores.values()]

        plt.figure(figsize=(8, 5))
        sns.barplot(x=names, y=mses, palette="Set2")
        plt.title("Comparación de errores (MSE)")
        plt.ylabel("MSE")
        plt.grid(True, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    @staticmethod
    def plot_model_errors_boxplot(model_scores, save_path=None):
        # Preparar DataFrame para seaborn
        data = []
        for name, scores in model_scores.items():
            errors = scores['y_true'] - scores['y_pred']
            for err in errors:
                data.append({'Modelo': name, 'Error': err})
        df_errors = pd.DataFrame(data)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Modelo', y='Error', data=df_errors, palette="Set3")
        plt.title("Distribución de errores (Residuos) por modelo")
        plt.grid(True, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
