import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path

class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> pd.DataFrame: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: return df
    def validate(self, df: pd.DataFrame) -> None: ...
    def run(self, path: str) -> pd.DataFrame:
        df = self.load(path)
        df = self.transform(df)
        self.validate(df)
        return df