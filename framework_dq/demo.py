"""
=============================================================================
DEMO — FRAMEWORK DE CALIDAD DE DATOS
=============================================================================
Lee tu propio CSV y ejecuta el pipeline completo:
  1. Ingesta
  2. Preprocesamiento automático
  3. Detección de anomalías (Motor Híbrido)
  4. Explicabilidad
  5. Reporte y dashboard
=============================================================================
"""
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
import pandas as pd
from framework import DataQualityFramework
 
 
# ------------------------------------------------------------------
# CONFIGURACIÓN — ajusta estos valores a tu dataset
# ------------------------------------------------------------------
CSV_PATH   = "test.csv"       # Ruta a tu archivo CSV
OUTPUT_DIR = "reports"     # Carpeta donde se guardan los reportes
 
CONFIG = {
    "contamination":        0.06,   # Proporción estimada de anomalías (ej: 0.05 = 5%)
    "ae_epochs":            40,     # Épocas de entrenamiento del Autoencoder
    "threshold_percentile": 94,     # Percentil para definir el umbral de anomalía
    "weights": {
        "autoencoder":      0.40,
        "isolation_forest": 0.35,
        "one_class_svm":    0.25,
    }
}
 
 
# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Cargar datos
    print(f"\nCargando datos desde: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset cargado: {len(df):,} registros | {df.shape[1]} columnas")
 
    # 2. Correr el framework
    fw = DataQualityFramework(config=CONFIG)
    fw.run(df, source_type="dataframe", output_dir=OUTPUT_DIR)
 
    # 3. Resumen de resultados
    results = fw.results
    n_total   = len(results)
    n_anomaly = int(results["anomaly_label"].sum())
    n_normal  = n_total - n_anomaly
 
    print("\n" + "=" * 50)
    print("  RESUMEN DE DETECCIÓN")
    print("=" * 50)
    print(f"  Total de registros    : {n_total:,}")
    print(f"  Registros normales    : {n_normal:,}")
    print(f"  Anomalías detectadas  : {n_anomaly:,} ({n_anomaly/n_total*100:.2f}%)")
    print("=" * 50)
 
    # 4. Mostrar top 10 anomalías con su explicación
    anomalies = results[results["anomaly_label"] == 1].sort_values(
        "anomaly_score", ascending=False
    ).head(10)
 
    print("\n  TOP 10 ANOMALÍAS (mayor score):")
    print(f"  {'ID':<6} {'Score':>8}  {'Variables clave'}")
    print("  " + "-" * 55)
 
    reason_map = {e["index"]: e for e in fw.explanation["explanations"]}
    for idx, row in anomalies.iterrows():
        exp = reason_map.get(idx, {})
        top_vars = ", ".join(exp.get("top_features", [])[:2])
        print(f"  {idx:<6} {row['anomaly_score']:>8.4f}  {top_vars}")
 
    print("\n" + "=" * 50)
    print(f"  ✓ Dashboard     → {OUTPUT_DIR}/dashboard.png")
    print(f"  ✓ Anomalías CSV → {OUTPUT_DIR}/anomalias_detectadas.csv")
    print(f"  ✓ Métricas      → {OUTPUT_DIR}/metricas.txt")
    print("=" * 50 + "\n")
