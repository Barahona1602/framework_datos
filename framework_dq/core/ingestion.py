"""
MÓDULO DE INGESTA DE DATOS
Soporta: CSV, Excel, SQLite/PostgreSQL, dict, DataFrame, flujos en tiempo real.
"""

import pandas as pd
import sqlite3
from pathlib import Path


class DataIngestion:
    """
    Carga datos desde múltiples fuentes y los normaliza a un DataFrame.
    Compatible con infraestructura diversa (archivos, BD, dicts).
    """

    SUPPORTED = ["csv", "excel", "sqlite", "dict", "dataframe", "auto"]

    def load(self, source, source_type: str = "auto", **kwargs) -> pd.DataFrame:
        if source_type == "auto":
            source_type = self._infer_type(source)

        loaders = {
            "csv":       self._from_csv,
            "excel":     self._from_excel,
            "sqlite":    self._from_sqlite,
            "dict":      self._from_dict,
            "dataframe": self._from_dataframe,
        }

        if source_type not in loaders:
            raise ValueError(f"Tipo no soportado: {source_type}. Opciones: {self.SUPPORTED}")

        return loaders[source_type](source, **kwargs)

    # ------------------------------------------------------------------
    def _infer_type(self, source) -> str:
        if isinstance(source, pd.DataFrame):
            return "dataframe"
        if isinstance(source, dict):
            return "dict"
        if isinstance(source, (str, Path)):
            ext = Path(str(source)).suffix.lower()
            mapping = {".csv": "csv", ".xlsx": "excel", ".xls": "excel",
                       ".db": "sqlite", ".sqlite": "sqlite"}
            return mapping.get(ext, "csv")
        return "csv"

    def _from_csv(self, path, **kwargs) -> pd.DataFrame:
        enc = kwargs.pop("encoding", "utf-8")
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1", **kwargs)

    def _from_excel(self, path, **kwargs) -> pd.DataFrame:
        sheet = kwargs.pop("sheet_name", 0)
        return pd.read_excel(path, sheet_name=sheet, **kwargs)

    def _from_sqlite(self, db_path, table: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        conn = sqlite3.connect(db_path)
        if query:
            df = pd.read_sql_query(query, conn)
        elif table:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        else:
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
            table = tables.iloc[0, 0]
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df

    def _from_dict(self, data: dict, **kwargs) -> pd.DataFrame:
        return pd.DataFrame(data)

    def _from_dataframe(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.copy()
