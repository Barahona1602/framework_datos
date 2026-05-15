# FRAMEWORK INNOVADOR PARA CONTROL DE CALIDAD DE DATOS
## Utilizando Aprendizaje No Supervisado

**Pablo Josué Barahona Luncey**
Universidad de San Carlos de Guatemala — Facultad de Ingeniería
Asesorado por M.A. Ing. Otto Abraham Hernández Ortega

---

## Estructura del Proyecto

```
framework_dq/
├── framework.py          # Orquestador principal (punto de entrada)
├── demo.py               # Script de demostración con datos sintéticos
├── core/
│   ├── ingestion.py      # Módulo de ingesta (CSV, Excel, SQLite, DataFrame)
│   ├── preprocessor.py   # Preprocesamiento automático
│   ├── explainability.py # Módulo de explicabilidad
│   └── reporter.py       # Generador de reportes y dashboard
├── models/
│   └── hybrid_engine.py  # Motor híbrido (Autoencoder + IF + One-Class SVM)
└── reports/              # Salidas generadas (dashboard.png, CSVs, métricas)
```

---

## Instalación de Dependencias

```bash
pip install scikit-learn pandas numpy matplotlib seaborn tensorflow pyod
```

---

## Uso Básico

### 1. Con un archivo CSV
```python
from framework import DataQualityFramework

fw = DataQualityFramework()
fw.run("mi_dataset.csv", output_dir="resultados")
```

### 2. Con un DataFrame de Pandas
```python
import pandas as pd
from framework import DataQualityFramework

df = pd.read_csv("datos.csv")
fw = DataQualityFramework()
fw.run(df, source_type="dataframe", output_dir="resultados")
```

### 3. Con archivo Excel
```python
fw.run("datos.xlsx", source_type="excel", output_dir="resultados")
```

### 4. Con configuración personalizada
```python
fw = DataQualityFramework(config={
    "contamination":        0.05,   # Proporción esperada de anomalías
    "ae_epochs":            50,     # Épocas del Autoencoder
    "threshold_percentile": 95,     # Percentil para el umbral
    "weights": {
        "autoencoder":      0.40,   # Peso del Autoencoder en el ensemble
        "isolation_forest": 0.35,   # Peso del Isolation Forest
        "one_class_svm":    0.25,   # Peso del One-Class SVM
    }
})
fw.run("datos.csv", output_dir="resultados")
```

---

## Pipeline de 4 Fases

```
[1] INGESTA      → CSV / Excel / SQLite / DataFrame
[2] PREPROCESO   → Detección de tipos, imputación, encoding, escalado
[3] DETECCIÓN    → Autoencoder + Isolation Forest + One-Class SVM (ensemble)
[4] EXPLICACIÓN  → Z-scores por feature, razón de anomalía
    └── REPORTE  → dashboard.png, anomalias_detectadas.csv, metricas.txt
```

---

## Resultados del Demo (datos sintéticos)

| Métrica       | Valor  |
|---------------|--------|
| Precisión     | 0.9804 |
| Recall        | 1.0000 |
| F1-Score      | 0.9901 |
| TP            | 50     |
| FP            | 1      |
| FN            | 0      |

---

## Tecnologías Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras** — Autoencoder de detección
- **scikit-learn** — Isolation Forest, One-Class SVM, StandardScaler
- **pandas / numpy** — Manipulación de datos
- **matplotlib** — Dashboard de visualización

---

## Para Usar con Tus Propios Datos

1. Ejecuta `demo.py` para verificar que el entorno funciona.
2. Reemplaza el DataFrame sintético con tu fuente real.
3. Ajusta `contamination` según tu estimación de anomalías esperadas.
4. Revisa `reports/dashboard.png` para los resultados visuales.
5. Exporta `reports/anomalias_detectadas.csv` para el análisis de resultados.
