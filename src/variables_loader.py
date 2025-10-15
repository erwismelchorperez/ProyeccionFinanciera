# src/extra_loader.py
import pandas as pd
from .loader_datasets import BaseLoader
from .utils import parse_ddmmyyyy, format_mes_yy, to_numeric, drop_cols, to_numeric_except

class ExtraLoader(BaseLoader):
    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1) quitar columnas que no te sirven en este archivo
        #df = drop_cols(df, ["col_inutil_1", "col_inutil_2"])

        # 2) formatear fecha a "ene13"
        dt = parse_ddmmyyyy(df["fecha"])
        df["fecha"] = format_mes_yy(dt)

        # 3) convertir numéricos
        #df = to_numeric(df, ["tipo_cambio_dolar", "cetes_28dias", "bonos_3anios"], fillna=0)
        df=to_numeric_except(df,['fecha'],fillna=0.0)

        return df

    def validate(self, df: pd.DataFrame) -> None:
        if df["fecha"].isna().any():
            raise ValueError("ExtraLoader: hay fechas inválidas")
