"""
MÓDULO GENERADOR DE REPORTES
Produce visualizaciones y métricas del framework:
 - Distribución de scores
 - Comparativa entre modelos
 - Feature importance
 - Resumen ejecutivo CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


# Paleta de colores del reporte
PALETTE = {
    "bg":       "#0F1117",
    "surface":  "#1A1D2E",
    "accent1":  "#4F8EF7",   # azul
    "accent2":  "#F7A84F",   # naranja
    "accent3":  "#4FF7A8",   # verde
    "danger":   "#F74F6B",   # rojo anomalías
    "text":     "#E8EAF6",
    "subtext":  "#8B95B0",
}


class ReportGenerator:

    def generate(self, raw_data, processed_data, results, explanation,
                 preproc_report, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)

        self._plot_dashboard(results, explanation, preproc_report, output_dir)
        self._save_anomalies_csv(results, explanation, output_dir)
        self._save_metrics_txt(results, explanation, preproc_report, output_dir)
        print(f"      ✓ Reporte guardado en: {os.path.abspath(output_dir)}/")

    # ------------------------------------------------------------------
    def _plot_dashboard(self, results, explanation, preproc_report, out):
        fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
        gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ---- Título ----
        fig.text(0.5, 0.97,
                 "FRAMEWORK DE CALIDAD DE DATOS — DASHBOARD DE RESULTADOS",
                 ha="center", va="top", fontsize=15, fontweight="bold",
                 color=PALETTE["text"], fontfamily="monospace")
        fig.text(0.5, 0.945,
                 "Aprendizaje No Supervisado | Motor Híbrido (Autoencoder + Isolation Forest + One-Class SVM)",
                 ha="center", va="top", fontsize=9, color=PALETTE["subtext"])

        # --- Panel 1: Distribución del anomaly score ---
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_score_distribution(ax1, results)

        # --- Panel 2: Donut anomalías vs normales ---
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_donut(ax2, results)

        # --- Panel 3: Scores por modelo (box) ---
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_model_comparison(ax3, results)

        # --- Panel 4: Feature importance ---
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_feature_importance(ax4, explanation)

        # --- Panel 5: Scatter score ae vs if ---
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_scatter(ax5, results, "score_autoencoder", "score_isolation_forest")

        # --- Panel 6: Score timeline ---
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_timeline(ax6, results)

        # --- Panel 7: Métricas resumen ---
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_metrics_table(ax7, results, preproc_report)

        path = os.path.join(out, "dashboard.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close()

    # ------------------------------------------------------------------
    def _style_ax(self, ax, title=""):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["subtext"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2D3E")
        if title:
            ax.set_title(title, color=PALETTE["text"], fontsize=9,
                         fontweight="bold", pad=8)

    def _plot_score_distribution(self, ax, results):
        self._style_ax(ax, "Distribución del Anomaly Score (Ensemble)")
        normal  = results.loc[results["anomaly_label"] == 0, "anomaly_score"]
        anomaly = results.loc[results["anomaly_label"] == 1, "anomaly_score"]
        bins = np.linspace(0, 1, 50)
        ax.hist(normal,  bins=bins, color=PALETTE["accent3"], alpha=0.7, label="Normal",   density=True)
        ax.hist(anomaly, bins=bins, color=PALETTE["danger"],  alpha=0.8, label="Anomalía", density=True)
        thr = results["anomaly_score"].quantile(0.95)
        ax.axvline(thr, color=PALETTE["accent2"], linestyle="--", lw=1.5, label=f"Umbral ({thr:.3f})")
        ax.set_xlabel("Score", color=PALETTE["subtext"], fontsize=8)
        ax.set_ylabel("Densidad", color=PALETTE["subtext"], fontsize=8)
        ax.legend(fontsize=8, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    def _plot_donut(self, ax, results):
        self._style_ax(ax, "Proporción")
        n_anom   = int(results["anomaly_label"].sum())
        n_normal = len(results) - n_anom
        sizes  = [n_normal, n_anom]
        colors = [PALETTE["accent3"], PALETTE["danger"]]
        wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                           wedgeprops=dict(width=0.55, edgecolor=PALETTE["bg"]))
        pct = n_anom / len(results) * 100
        ax.text(0, 0, f"{pct:.1f}%\nAnomally", ha="center", va="center",
                color=PALETTE["danger"], fontsize=11, fontweight="bold")
        ax.legend(["Normal", "Anomalía"], fontsize=8,
                  facecolor=PALETTE["surface"], labelcolor=PALETTE["text"],
                  loc="lower center")

    def _plot_model_comparison(self, ax, results):
        self._style_ax(ax, "Scores por Modelo (Anomalías vs. Normales)")
        models = ["score_autoencoder", "score_isolation_forest", "score_ocsvm"]
        labels = ["Autoencoder", "Isolation Forest", "One-Class SVM"]
        colors_n = [PALETTE["accent1"], PALETTE["accent3"], PALETTE["accent2"]]
        colors_a = [PALETTE["danger"]] * 3
        x = np.arange(len(models))
        w = 0.35
        for i, (col, lab) in enumerate(zip(models, labels)):
            norm_mean = results.loc[results["anomaly_label"]==0, col].mean()
            anom_mean = results.loc[results["anomaly_label"]==1, col].mean() if results["anomaly_label"].sum() > 0 else 0
            ax.bar(x[i] - w/2, norm_mean, w, color=colors_n[i], alpha=0.85, label=f"Normal ({lab})" if i==0 else "")
            ax.bar(x[i] + w/2, anom_mean, w, color=colors_a[i], alpha=0.85, label=f"Anomalía" if i==0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=PALETTE["text"], fontsize=8)
        ax.set_ylabel("Score medio", color=PALETTE["subtext"], fontsize=8)
        ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    def _plot_feature_importance(self, ax, explanation):
        self._style_ax(ax, "Variables más Relevantes")
        fi = explanation.get("feature_importance", pd.Series(dtype=float))
        if fi.empty:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    color=PALETTE["subtext"], transform=ax.transAxes)
            return
        top = fi.head(10)
        colors = [PALETTE["danger"] if v > top.mean() else PALETTE["accent1"] for v in top.values]
        ax.barh(range(len(top)), top.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, color=PALETTE["text"], fontsize=7)
        ax.set_xlabel("Desviación media (z-score)", color=PALETTE["subtext"], fontsize=8)
        ax.invert_yaxis()

    def _plot_scatter(self, ax, results, col_x, col_y):
        self._style_ax(ax, f"{col_x.replace('score_','')} vs {col_y.replace('score_','')}")
        mask = results["anomaly_label"] == 0
        ax.scatter(results.loc[mask,  col_x], results.loc[mask,  col_y],
                   c=PALETTE["accent3"], s=8, alpha=0.5, label="Normal")
        ax.scatter(results.loc[~mask, col_x], results.loc[~mask, col_y],
                   c=PALETTE["danger"],  s=20, alpha=0.8, label="Anomalía", marker="x")
        ax.set_xlabel(col_x.replace("score_", ""), color=PALETTE["subtext"], fontsize=8)
        ax.set_ylabel(col_y.replace("score_", ""), color=PALETTE["subtext"], fontsize=8)
        ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    def _plot_timeline(self, ax, results):
        self._style_ax(ax, "Anomaly Score — Secuencia de registros")
        ax.plot(results["anomaly_score"].values, color=PALETTE["accent1"],
                lw=0.8, alpha=0.7, label="Score")
        anom_idx = results.index[results["anomaly_label"] == 1]
        if len(anom_idx):
            pos = [results.index.get_loc(i) for i in anom_idx]
            ax.scatter(pos, results.loc[anom_idx, "anomaly_score"].values,
                       c=PALETTE["danger"], s=20, zorder=5, label="Anomalía")
        ax.axhline(results["anomaly_score"].quantile(0.95),
                   color=PALETTE["accent2"], lw=1, ls="--", label="Umbral")
        ax.set_xlabel("Registro #", color=PALETTE["subtext"], fontsize=8)
        ax.legend(fontsize=7, facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

    def _plot_metrics_table(self, ax, results, preproc_report):
        ax.set_facecolor(PALETTE["surface"])
        ax.axis("off")
        ax.set_title("Métricas Resumen", color=PALETTE["text"],
                     fontsize=9, fontweight="bold", pad=8)
        n   = len(results)
        n_a = int(results["anomaly_label"].sum())
        metrics = [
            ("Total registros",         f"{n:,}"),
            ("Anomalías detectadas",     f"{n_a:,}"),
            ("Tasa de anomalías",        f"{n_a/n*100:.2f}%"),
            ("Score umbral (p95)",       f"{results['anomaly_score'].quantile(0.95):.4f}"),
            ("Score medio anomalías",    f"{results.loc[results['anomaly_label']==1,'anomaly_score'].mean():.4f}" if n_a > 0 else "N/A"),
            ("Variables numéricas",      str(len(preproc_report.get("numeric_cols", [])))),
            ("Variables categóricas",    str(len(preproc_report.get("categorical_cols", [])))),
        ]
        y = 0.9
        for label, val in metrics:
            ax.text(0.05, y, label + ":", color=PALETTE["subtext"], fontsize=8,
                    transform=ax.transAxes, va="top")
            ax.text(0.65, y, val, color=PALETTE["accent1"], fontsize=8,
                    fontweight="bold", transform=ax.transAxes, va="top")
            y -= 0.125

    # ------------------------------------------------------------------
    def _save_anomalies_csv(self, results, explanation, out):
        anomaly_df = results[results["anomaly_label"] == 1].copy()
        reason_map = {e["index"]: e["reason"] for e in explanation["explanations"]}
        anomaly_df["razon_explicacion"] = anomaly_df.index.map(reason_map)
        path = os.path.join(out, "anomalias_detectadas.csv")
        anomaly_df.to_csv(path, index=True, encoding="utf-8-sig")

    def _save_metrics_txt(self, results, explanation, preproc_report, out):
        n   = len(results)
        n_a = int(results["anomaly_label"].sum())
        lines = [
            "=" * 60,
            "  REPORTE DE MÉTRICAS — FRAMEWORK DE CALIDAD DE DATOS",
            "=" * 60,
            f"Total de registros analizados : {n:,}",
            f"Anomalías detectadas          : {n_a:,} ({n_a/n*100:.2f}%)",
            f"Umbral adaptativo (p95)       : {results['anomaly_score'].quantile(0.95):.6f}",
            "",
            "SCORES PROMEDIO POR MODELO:",
            f"  Autoencoder      : {results['score_autoencoder'].mean():.4f}",
            f"  Isolation Forest : {results['score_isolation_forest'].mean():.4f}",
            f"  One-Class SVM    : {results['score_ocsvm'].mean():.4f}",
            "",
            "PREPROCESAMIENTO:",
            f"  {preproc_report['summary']}",
            "=" * 60,
        ]
        path = os.path.join(out, "metricas.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
