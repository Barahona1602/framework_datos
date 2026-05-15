"""
=============================================================================
FRAMEWORK INNOVADOR PARA CONTROL DE CALIDAD DE DATOS
UTILIZANDO APRENDIZAJE NO SUPERVISADO
=============================================================================
Pablo Josué Barahona Luncey
Universidad de San Carlos de Guatemala - Facultad de Ingeniería
Asesorado por M.A. Ing. Otto Abraham Hernández Ortega
=============================================================================
"""

from core.ingestion import DataIngestion
from core.preprocessor import AutoPreprocessor
from models.hybrid_engine import HybridAnomalyEngine
from core.explainability import ExplainabilityModule
from core.reporter import ReportGenerator
import pandas as pd
import time


class DataQualityFramework:
    """
    Framework principal para control de calidad de datos.
    Orquesta todos los módulos: ingesta, preprocesamiento,
    detección de anomalías y explicabilidad.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ingestion     = DataIngestion()
        self.preprocessor  = AutoPreprocessor()
        self.engine        = HybridAnomalyEngine(self.config)
        self.explainer     = ExplainabilityModule()
        self.reporter      = ReportGenerator()

        self.data_raw       = None
        self.data_processed = None
        self.results        = None

    # ------------------------------------------------------------------
    # FASE 1 – Ingesta
    # ------------------------------------------------------------------
    def load(self, source, source_type: str = "auto", **kwargs):
        """Carga datos desde CSV, Excel, BD, dict o DataFrame."""
        print("\n[1/4] INGESTA DE DATOS...")
        self.data_raw = self.ingestion.load(source, source_type, **kwargs)
        print(f"      ✓ {len(self.data_raw):,} registros | {self.data_raw.shape[1]} columnas cargados.")
        return self

    # ------------------------------------------------------------------
    # FASE 2 – Preprocesamiento
    # ------------------------------------------------------------------
    def preprocess(self):
        """Identifica tipos, estandariza formatos y aplica reglas configurables."""
        print("\n[2/4] PREPROCESAMIENTO AUTOMÁTICO...")
        self.data_processed, self.preproc_report = self.preprocessor.fit_transform(self.data_raw)
        print(f"      ✓ Variables: {self.preproc_report['summary']}")
        return self

    # ------------------------------------------------------------------
    # FASE 3 – Detección de anomalías
    # ------------------------------------------------------------------
    def detect(self):
        """Motor híbrido: Autoencoder + One-Class SVM + Isolation Forest."""
        print("\n[3/4] DETECCIÓN DE ANOMALÍAS (MOTOR HÍBRIDO)...")
        t0 = time.time()
        self.results = self.engine.fit_detect(self.data_processed)
        elapsed = time.time() - t0
        n_anomalies = self.results["anomaly_label"].sum()
        print(f"      ✓ Anomalías detectadas: {n_anomalies:,} / {len(self.results):,} registros ({n_anomalies/len(self.results)*100:.2f}%)")
        print(f"      ✓ Tiempo de procesamiento: {elapsed:.2f}s")
        return self

    # ------------------------------------------------------------------
    # FASE 4 – Explicabilidad
    # ------------------------------------------------------------------
    def explain(self):
        """Genera explicaciones por cada anomalía detectada."""
        print("\n[4/4] MÓDULO DE EXPLICABILIDAD...")
        self.explanation = self.explainer.explain(
            self.data_processed,
            self.results,
            self.engine
        )
        print(f"      ✓ Explicaciones generadas para {self.explanation['n_explained']:,} anomalías.")
        return self

    # ------------------------------------------------------------------
    # Reporte final
    # ------------------------------------------------------------------
    def report(self, output_dir: str = "reports"):
        """Genera reporte completo con métricas y visualizaciones."""
        print("\n[✓] GENERANDO REPORTE FINAL...")
        self.reporter.generate(
            raw_data       = self.data_raw,
            processed_data = self.data_processed,
            results        = self.results,
            explanation    = self.explanation,
            preproc_report = self.preproc_report,
            output_dir     = output_dir
        )
        return self

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def run(self, source, source_type: str = "auto", output_dir: str = "reports", **kwargs):
        """Ejecuta el pipeline completo de principio a fin."""
        print("=" * 65)
        print("  FRAMEWORK DE CALIDAD DE DATOS — APRENDIZAJE NO SUPERVISADO")
        print("=" * 65)
        return (
            self.load(source, source_type, **kwargs)
                .preprocess()
                .detect()
                .explain()
                .report(output_dir)
        )
