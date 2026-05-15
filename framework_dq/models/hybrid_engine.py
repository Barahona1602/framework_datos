"""
MOTOR HÍBRIDO DE DETECCIÓN DE ANOMALÍAS
Combina tres técnicas no supervisadas:
  1. Autoencoder (TensorFlow/Keras) — error de reconstrucción
  2. One-Class SVM (scikit-learn)   — frontera de normalidad
  3. Isolation Forest (scikit-learn) — aislamiento de outliers

La puntuación final es un ensemble ponderado de los tres modelos.
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")

# TensorFlow opcional — se usa si está disponible
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class HybridAnomalyEngine:
    """
    Motor híbrido que ensambla tres detectores no supervisados.
    Cada modelo produce un anomaly score; el ensemble los combina
    mediante ponderación configurable.
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        # Pesos del ensemble (deben sumar 1.0)
        self.weights = cfg.get("weights", {
            "autoencoder":     0.40,
            "isolation_forest": 0.35,
            "one_class_svm":   0.25,
        })
        self.contamination  = cfg.get("contamination", 0.05)
        self.ae_epochs      = cfg.get("ae_epochs", 50)
        self.ae_batch_size  = cfg.get("ae_batch_size", 32)
        self.threshold_pct  = cfg.get("threshold_percentile", 95)

        self._ae_model    = None
        self._iforest     = None
        self._ocsvm       = None
        self._threshold   = None
        self.feature_names = None

    # ------------------------------------------------------------------
    # Entrenamiento + detección (fit + predict en un solo paso)
    # ------------------------------------------------------------------
    def fit_detect(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.values.astype(np.float32)
        self.feature_names = list(df.columns)

        scores_ae   = self._fit_autoencoder(X)
        scores_if   = self._fit_isolation_forest(X)
        scores_svm  = self._fit_ocsvm(X)

        # Normalizar cada score a [0, 1]
        scores_ae  = self._normalize(scores_ae)
        scores_if  = self._normalize(scores_if)
        scores_svm = self._normalize(scores_svm)

        # Ensemble ponderado
        ensemble_score = (
            self.weights["autoencoder"]      * scores_ae  +
            self.weights["isolation_forest"] * scores_if  +
            self.weights["one_class_svm"]    * scores_svm
        )

        # Umbral adaptativo
        self._threshold = np.percentile(ensemble_score, self.threshold_pct)
        labels = (ensemble_score > self._threshold).astype(int)

        results = df.copy()
        results["score_autoencoder"]     = scores_ae
        results["score_isolation_forest"] = scores_if
        results["score_ocsvm"]           = scores_svm
        results["anomaly_score"]         = ensemble_score
        results["anomaly_label"]         = labels
        return results

    # ------------------------------------------------------------------
    # Autoencoder
    # ------------------------------------------------------------------
    def _fit_autoencoder(self, X: np.ndarray) -> np.ndarray:
        if not TF_AVAILABLE:
            # Fallback: PCA-based reconstruction error
            return self._pca_reconstruction_error(X)

        n_features = X.shape[1]
        enc_dim    = max(2, n_features // 2)
        bottleneck = max(2, n_features // 4)

        inp  = Input(shape=(n_features,))
        enc  = Dense(enc_dim,    activation="relu")(inp)
        enc  = Dropout(0.1)(enc)
        btn  = Dense(bottleneck, activation="relu")(enc)
        dec  = Dense(enc_dim,    activation="relu")(btn)
        out  = Dense(n_features, activation="linear")(dec)

        self._ae_model = Model(inp, out)
        self._ae_model.compile(optimizer="adam", loss="mse")

        cb = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
        self._ae_model.fit(
            X, X,
            epochs     = self.ae_epochs,
            batch_size = self.ae_batch_size,
            validation_split = 0.1,
            callbacks  = [cb],
            verbose    = 0
        )

        X_hat  = self._ae_model.predict(X, verbose=0)
        errors = np.mean((X - X_hat) ** 2, axis=1)
        return errors

    def _pca_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Fallback cuando TensorFlow no está disponible."""
        from sklearn.decomposition import PCA
        n_comp = max(1, min(X.shape[1] // 2, X.shape[0] - 1))
        pca = PCA(n_components=n_comp)
        X_r = pca.inverse_transform(pca.fit_transform(X))
        return np.mean((X - X_r) ** 2, axis=1)

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    def _fit_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        self._iforest = IsolationForest(
            contamination = self.contamination,
            random_state  = 42,
            n_estimators  = 200
        )
        self._iforest.fit(X)
        # score_samples devuelve valores negativos: más negativo = más anómalo
        raw = -self._iforest.score_samples(X)
        return raw

    # ------------------------------------------------------------------
    # One-Class SVM
    # ------------------------------------------------------------------
    def _fit_ocsvm(self, X: np.ndarray) -> np.ndarray:
        # Submuestreo para SVM (costoso en datasets grandes)
        n = min(5000, X.shape[0])
        idx = np.random.choice(X.shape[0], n, replace=False)
        X_sample = X[idx]

        self._ocsvm = OneClassSVM(
            kernel = "rbf",
            nu     = self.contamination,
            gamma  = "scale"
        )
        self._ocsvm.fit(X_sample)
        raw = -self._ocsvm.score_samples(X)
        return raw

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def get_threshold(self):
        return self._threshold

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Inferencia en nuevos datos (modelos ya entrenados)."""
        X = df.values.astype(np.float32)
        scores_ae  = self._normalize(-self._ae_model.predict(X, verbose=0).mean(axis=1)) if self._ae_model else self._normalize(self._pca_reconstruction_error(X))
        scores_if  = self._normalize(-self._iforest.score_samples(X))
        scores_svm = self._normalize(-self._ocsvm.score_samples(X))
        ensemble   = (self.weights["autoencoder"] * scores_ae +
                      self.weights["isolation_forest"] * scores_if +
                      self.weights["one_class_svm"] * scores_svm)
        return (ensemble > self._threshold).astype(int)
