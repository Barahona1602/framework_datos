"""
MÓDULO DE PREPROCESAMIENTO AUTOMÁTICO
- Identifica tipos: numérica, categórica, fecha
- Estandariza formatos y escala variables
- Aplica reglas configurables (imputación, encoding)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class AutoPreprocessor:
    """
    Preprocesamiento inteligente que detecta automáticamente el tipo
    de cada variable y aplica las transformaciones adecuadas.
    """

    def __init__(self, date_threshold: float = 0.8, cat_threshold: int = 50):
        self.date_threshold = date_threshold  # fracción para detectar fechas
        self.cat_threshold  = cat_threshold   # máx. valores únicos para categórica
        self.scaler          = StandardScaler()
        self.label_encoders  = {}
        self.column_types    = {}
        self.numeric_cols    = []
        self.categorical_cols = []
        self.date_cols       = []
        self.dropped_cols    = []

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame):
        df = df.copy()
        self._detect_types(df)
        df = self._handle_dates(df)
        df = self._handle_missing(df)
        df = self._encode_categoricals(df)
        df = self._scale_numerics(df)

        report = {
            "summary": (f"{len(self.numeric_cols)} numéricas | "
                        f"{len(self.categorical_cols)} categóricas | "
                        f"{len(self.date_cols)} fechas | "
                        f"{len(self.dropped_cols)} descartadas"),
            "numeric_cols":      self.numeric_cols,
            "categorical_cols":  self.categorical_cols,
            "date_cols":         self.date_cols,
            "dropped_cols":      self.dropped_cols,
            "column_types":      self.column_types,
        }
        return df, report

    # ------------------------------------------------------------------
    def _detect_types(self, df: pd.DataFrame):
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()

            # Intento de fecha
            if dtype == object:
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if parsed.notna().mean() >= self.date_threshold:
                    self.date_cols.append(col)
                    self.column_types[col] = "date"
                    continue

            if pd.api.types.is_numeric_dtype(dtype):
                self.numeric_cols.append(col)
                self.column_types[col] = "numeric"
            elif dtype == object or pd.api.types.is_categorical_dtype(dtype):
                if n_unique <= self.cat_threshold:
                    self.categorical_cols.append(col)
                    self.column_types[col] = "categorical"
                else:
                    # Texto libre — se descarta del modelo (demasiada cardinalidad)
                    self.dropped_cols.append(col)
                    self.column_types[col] = "text_dropped"
            else:
                self.dropped_cols.append(col)
                self.column_types[col] = "unknown_dropped"

    def _handle_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_cols:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            df[col + "_year"]  = parsed.dt.year
            df[col + "_month"] = parsed.dt.month
            df[col + "_day"]   = parsed.dt.day
            df[col + "_dow"]   = parsed.dt.dayofweek
            self.numeric_cols.extend([col + "_year", col + "_month",
                                       col + "_day",   col + "_dow"])
            df.drop(columns=[col], inplace=True)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "MISSING")
        if self.dropped_cols:
            existing = [c for c in self.dropped_cols if c in df.columns]
            df.drop(columns=existing, inplace=True)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        return df

    def _scale_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [c for c in self.numeric_cols if c in df.columns]
        if existing:
            df[existing] = self.scaler.fit_transform(df[existing])
        return df

    def get_feature_names(self, df: pd.DataFrame):
        return [c for c in df.columns]
