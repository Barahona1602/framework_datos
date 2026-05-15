"""
MÓDULO DE EXPLICABILIDAD
Explica por qué cada registro fue marcado como anomalía.
Usa desviación por feature respecto a la distribución normal.
Compatible con SHAP cuando está disponible.
"""

import numpy as np
import pandas as pd


class ExplainabilityModule:
    """
    Genera explicaciones para cada anomalía detectada.
    Indica cuál variable contribuyó más a la puntuación anómala.
    """

    def explain(self, processed_df: pd.DataFrame, results: pd.DataFrame, engine) -> dict:
        anomalies   = results[results["anomaly_label"] == 1]
        normal_data = results[results["anomaly_label"] == 0]

        feature_cols = [c for c in processed_df.columns]
        explanations = []

        # Estadísticas de referencia (datos normales)
        if len(normal_data) == 0:
            ref_mean = processed_df.mean()
            ref_std  = processed_df.std().replace(0, 1e-6)
        else:
            ref_mean = processed_df.loc[normal_data.index, feature_cols].mean()
            ref_std  = processed_df.loc[normal_data.index, feature_cols].std().replace(0, 1e-6)

        for idx in anomalies.index:
            row    = processed_df.loc[idx, feature_cols]
            z_scores = ((row - ref_mean) / ref_std).abs()
            top_features = z_scores.nlargest(3)

            explanations.append({
                "index":        idx,
                "anomaly_score": results.loc[idx, "anomaly_score"],
                "score_ae":     results.loc[idx, "score_autoencoder"],
                "score_if":     results.loc[idx, "score_isolation_forest"],
                "score_svm":    results.loc[idx, "score_ocsvm"],
                "top_features": top_features.index.tolist(),
                "top_z_scores": top_features.values.tolist(),
                "reason":       self._build_reason(top_features),
            })

        # Feature importance global (promedio de z-scores en anomalías)
        if len(anomalies) > 0:
            anom_rows = processed_df.loc[anomalies.index, feature_cols]
            global_z  = ((anom_rows - ref_mean) / ref_std).abs().mean()
            feature_importance = global_z.sort_values(ascending=False)
        else:
            feature_importance = pd.Series(dtype=float)

        return {
            "n_explained":        len(explanations),
            "explanations":       explanations,
            "feature_importance": feature_importance,
            "ref_mean":           ref_mean,
            "ref_std":            ref_std,
        }

    # ------------------------------------------------------------------
    def _build_reason(self, top_features: pd.Series) -> str:
        parts = []
        for feat, z in top_features.items():
            severity = "alta" if z > 3 else "moderada"
            parts.append(f"'{feat}' (desviación {severity}: z={z:.2f})")
        return "Anomalía por: " + ", ".join(parts)

    def summary_table(self, explanation: dict) -> pd.DataFrame:
        rows = []
        for e in explanation["explanations"]:
            rows.append({
                "Índice":        e["index"],
                "Score":         round(e["anomaly_score"], 4),
                "Variables clave": ", ".join(e["top_features"]),
                "Razón":         e["reason"],
            })
        return pd.DataFrame(rows)
