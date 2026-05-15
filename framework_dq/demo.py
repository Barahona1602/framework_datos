"""
=============================================================================
DEMO — FRAMEWORK DE CALIDAD DE DATOS
=============================================================================
Genera datos sintéticos con anomalías inyectadas y ejecuta el pipeline
completo para producir el dashboard de resultados.
=============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from framework import DataQualityFramework


# ------------------------------------------------------------------
# Generador de datos sintéticos con anomalías inyectadas
# ------------------------------------------------------------------
def generate_synthetic_data(n_normal: int = 800, n_anomaly: int = 50, seed: int = 42):
    """
    Simula un dataset transaccional empresarial con errores conocidos.
    Variables: monto, edad, días_activo, categoria, region
    """
    rng = np.random.default_rng(seed)

    # Datos normales
    normal = pd.DataFrame({
        "monto_transaccion": rng.normal(500, 120, n_normal).clip(50, 1500),
        "edad_cliente":      rng.integers(18, 70, n_normal).astype(float),
        "dias_activo":       rng.integers(1, 365, n_normal).astype(float),
        "num_transacciones": rng.integers(1, 50, n_normal).astype(float),
        "score_credito":     rng.normal(650, 80, n_normal).clip(300, 850),
        "categoria":         rng.choice(["Retail", "Servicios", "Tecnología", "Salud"], n_normal),
        "region":            rng.choice(["Norte", "Sur", "Centro", "Oriente"], n_normal),
        "es_anomalia_real":  [0] * n_normal,
    })

    # Anomalías inyectadas (outliers reales)
    anomalies = pd.DataFrame({
        "monto_transaccion": rng.choice(
            np.concatenate([rng.normal(15000, 500, n_anomaly // 2),
                            rng.normal(0.5, 0.2, n_anomaly // 2)]).clip(0, 20000),
            n_anomaly, replace=False
        ),
        "edad_cliente":      rng.choice([5, 130, 999], n_anomaly, replace=True).astype(float),
        "dias_activo":       rng.choice([-10, 5000], n_anomaly, replace=True).astype(float),
        "num_transacciones": rng.integers(500, 2000, n_anomaly).astype(float),
        "score_credito":     rng.choice([10, 1200], n_anomaly, replace=True).astype(float),
        "categoria":         rng.choice(["Retail", "Servicios", "Tecnología", "Salud"], n_anomaly),
        "region":            rng.choice(["Norte", "Sur", "Centro", "Oriente"], n_anomaly),
        "es_anomalia_real":  [1] * n_anomaly,
    })

    df = pd.concat([normal, anomalies], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# Ejecución del pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\nGenerando dataset sintético transaccional...")
    df = generate_synthetic_data(n_normal=800, n_anomaly=50)
    ground_truth = df.pop("es_anomalia_real")

    fw = DataQualityFramework(config={
        "contamination":        0.06,
        "ae_epochs":            40,
        "threshold_percentile": 94,
        "weights": {
            "autoencoder":      0.40,
            "isolation_forest": 0.35,
            "one_class_svm":    0.25,
        }
    })

    fw.run(
        source      = df,
        source_type = "dataframe",
        output_dir  = "reports"
    )

    # Evaluación con ground truth
    preds = fw.results["anomaly_label"]
    tp = ((preds == 1) & (ground_truth == 1)).sum()
    fp = ((preds == 1) & (ground_truth == 0)).sum()
    fn = ((preds == 0) & (ground_truth == 1)).sum()
    tn = ((preds == 0) & (ground_truth == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print("  EVALUACIÓN CON GROUND TRUTH")
    print("=" * 50)
    print(f"  Verdaderos Positivos  (TP): {tp}")
    print(f"  Falsos Positivos      (FP): {fp}")
    print(f"  Falsos Negativos      (FN): {fn}")
    print(f"  Verdaderos Negativos  (TN): {tn}")
    print(f"  Precisión             : {precision:.4f}")
    print(f"  Recall                : {recall:.4f}")
    print(f"  F1-Score              : {f1:.4f}")
    print("=" * 50)
    print("\n✓ Dashboard guardado en reports/dashboard.png")
    print("✓ Anomalías exportadas en reports/anomalias_detectadas.csv")
    print("✓ Métricas en reports/metricas.txt\n")
