import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

MESES_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}

def parse_ddmmyyyy(s):
    return pd.to_datetime(s, dayfirst=True, errors='coerce')

def format_mes_yy(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.month.map(MESES_ES) + dt_series.dt.year.mod(100).astype(str).str.zfill(2)

def to_numeric(df, cols, fillna=0.0):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(fillna)
    return df

def to_numeric_except(df, exclude=None, fillna=0.0, inplace=True, convert_datetimes=False):
    exclude = set(exclude or [])
    cols = [c for c in df.columns if c not in exclude]

    # opcionalmente evita tocar datetimes
    if not convert_datetimes:
        cols = [c for c in cols if not is_datetime64_any_dtype(df[c])]

    # convierte solo las que NO son numéricas (evita trabajo innecesario)
    cols_to_convert = [c for c in cols if not is_numeric_dtype(df[c])]

    converted = df[cols_to_convert].apply(pd.to_numeric, errors="coerce")
    if fillna is not None:
        converted = converted.fillna(fillna)

    if inplace:
        df[cols_to_convert] = converted
        return df
    else:
        out = df.copy()
        out[cols_to_convert] = converted
        print(df)
        return out


def drop_cols(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")
