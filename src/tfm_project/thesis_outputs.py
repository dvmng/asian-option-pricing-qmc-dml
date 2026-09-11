from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .paths import ProjectPaths
from .plotting import save_figure, set_thesis_style


STRATEGY_LABEL = {
    "no_hedge": "Sin cobertura",
    "oracle_rqmc": "Oracle RQMC",
    "mlp_ensemble": "MLP",
    "dml_ensemble": "DML",
    "mlp": "MLP",
    "dml": "DML",
}

# Strategy palette used consistently across thesis figures.
STRATEGY_COLOR = {
    "no_hedge": "0.55",
    "oracle_rqmc": "C2",
    "mlp_ensemble": "C0",
    "dml_ensemble": "C1",
    "mlp": "C0",
    "dml": "C1",
}

PRIMARY_CHINA_H100 = "SMM-H100-80G-整机-西部-包月"


def _labels(language: str) -> dict[str, str]:
    if language == "en":
        return {
            "training_states": "Training states",
            "price_rmse": "Price RMSE",
            "delta_rmse": "Delta RMSE",
            "strategy": "Strategy",
            "rmse": "RMSE (K units)",
            "shock": "Spot shock magnitude (%)",
            "local": "Local hedge RMSE",
            "fixings": "Number of future fixings",
            "boundary": "Distance to nearest fixing (days)",
            "share": "Share of total squared error",
            "top1": "Worst 1%",
            "top5": "Worst 5%",
        }
    return {
        "training_states": "Estados de entrenamiento",
        "price_rmse": "RMSE del precio",
        "delta_rmse": "RMSE de Delta",
        "strategy": "Estrategia",
        "rmse": "RMSE (unidades de K)",
        "shock": "Magnitud del shock de spot (%)",
        "local": "RMSE de cobertura local",
        "fixings": "Número de fixings futuros",
        "boundary": "Distancia al fixing más próximo (días)",
        "share": "Porcentaje del error cuadrático total",
        "top1": "Peor 1%",
        "top5": "Peor 5%",
    }


def _titles(language: str) -> dict[str, str]:
    if language == "en":
        return {
            "market": "Ornn compute-capacity rental prices by GPU",
            "global_china": "H100 global and China — normalized evolution",
            "price": "Price RMSE by training-set size",
            "delta": "Delta RMSE by training-set size",
            "boundary": "Delta error around scheduled fixings",
            "local": "Local first-order hedge",
            "dynamic": "Monthly dynamic forward hedge",
            "tails": "Concentration of hedging error in extreme paths",
            "dynamic_delta": "Delta RMSE at monthly rebalancing dates",
        }
    return {
        "market": "Precios de alquiler de capacidad de cómputo por GPU — Ornn",
        "global_china": "H100 global y China — evolución normalizada",
        "price": "RMSE de precio según el tamaño de entrenamiento",
        "delta": "RMSE de Delta según el tamaño de entrenamiento",
        "boundary": "Error de Delta alrededor de los fixings programados",
        "local": "Cobertura local de primer orden",
        "dynamic": "Cobertura dinámica mensual con forwards",
        "tails": "Concentración del error de cobertura en las trayectorias extremas",
        "dynamic_delta": "RMSE de Delta en los rebalanceos mensuales",
    }


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"SKIP missing: {path}")
        return None
    return pd.read_csv(path)


def _clean_boundary_labels(values: list[str], language: str) -> list[str]:
    if language == "en":
        mapping = {
            "<=0.25d": "≤ 0.25",
            "(0.25,0.5]d": "0.25–0.5",
            "(0.5,1]d": "0.5–1",
            "(1,2]d": "1–2",
            "(2,3]d": "2–3",
            "(3,7]d": "3–7",
            "(7,15]d": "7–15",
            ">15d": "> 15",
        }
    else:
        mapping = {
            "<=0.25d": "≤ 0,25",
            "(0.25,0.5]d": "0,25–0,5",
            "(0.5,1]d": "0,5–1",
            "(1,2]d": "1–2",
            "(2,3]d": "2–3",
            "(3,7]d": "3–7",
            "(7,15]d": "7–15",
            ">15d": "> 15",
        }
    return [mapping.get(v, v.replace("d", "")) for v in values]


def _global_china_base100(paths: ProjectPaths) -> pd.DataFrame | None:
    """Build a descriptive common-date H100 comparison without changing frozen inputs.

    The two price levels are not treated as economically identical contracts. Each series
    is independently rebased to 100 on the first common date, so the figure compares
    relative evolution rather than raw USD/CNY price levels.
    """
    ornn_path = paths.data_raw / "global" / "ornn_h100.csv"
    smm_path = paths.data_processed / "derived" / "smm_h100_daily.csv"
    ornn = _read(ornn_path)
    smm = _read(smm_path)
    if ornn is None or smm is None or ornn.empty or smm.empty:
        return None

    required_ornn = {"date", "price_avg"}
    required_smm = {"date", "index_id", "price_avg"}
    if not required_ornn.issubset(ornn.columns) or not required_smm.issubset(smm.columns):
        print("SKIP global/China H100 figure: unexpected Ornn/SMM column layout.")
        return None

    g = ornn[["date", "price_avg"]].dropna().copy()
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).sort_values("date").rename(columns={"price_avg": "global_price"})

    c = smm.loc[smm["index_id"].eq(PRIMARY_CHINA_H100), ["date", "price_avg"]].dropna().copy()
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c = c.dropna(subset=["date"]).sort_values("date").rename(columns={"price_avg": "china_price"})
    if c.empty:
        print(f"SKIP global/China H100 figure: primary SMM series not found: {PRIMARY_CHINA_H100}")
        return None

    m = g.merge(c, on="date", how="inner").sort_values("date").drop_duplicates("date")
    if len(m) < 2:
        print("SKIP global/China H100 figure: fewer than two common dates.")
        return None

    m["global_base100"] = 100.0 * m["global_price"] / float(m["global_price"].iloc[0])
    m["china_base100"] = 100.0 * m["china_price"] / float(m["china_price"].iloc[0])
    return m


def make_figures(paths: ProjectPaths, *, language: str = "es") -> list[Path]:
    set_thesis_style()
    lab = _labels(language)
    title = _titles(language)
    saved: list[Path] = []

    # Ornn multi-GPU price levels; H100 is the primary underlying.
    p = paths.results / "market" / "ornn_multigpu" / "ornn_gpu_price_wide_common_dates.csv"
    df = _read(p)
    if df is not None and len(df):
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in df.columns[1:]:
            is_h100 = "H100" in str(col).upper()
            ax.plot(
                df[date_col],
                df[col],
                linewidth=1.9 if is_h100 else 1.15,
                alpha=1.0 if is_h100 else 0.72,
                zorder=3 if is_h100 else 2,
                label=str(col),
            )
        ax.set_title(title["market"])
        ax.set_xlabel("Fecha" if language == "es" else "Date")
        ax.set_ylabel("USD / GPU-h")
        ax.grid(True)
        ax.legend(ncol=2)
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "market" / "01_ornn_gpu_price_levels")
        plt.close(fig)

    # Global and China H100 on common dates, independently rebased to 100.
    gc = _global_china_base100(paths)
    if gc is not None and len(gc):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            gc["date"], gc["global_base100"],
            color="C0", linewidth=1.5, marker="o", markersize=3.5,
            label="Ornn H100 SXM — global" if language == "es" else "Ornn H100 SXM — global",
        )
        ax.plot(
            gc["date"], gc["china_base100"],
            color="C1", linewidth=1.5, marker="o", markersize=3.5,
            label="SMM H100 80G — oeste de China" if language == "es" else "SMM H100 80G — Western China",
        )
        ax.axhline(100.0, color="0.55", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_title(title["global_china"])
        ax.set_xlabel("Fecha" if language == "es" else "Date")
        ax.set_ylabel("Índice (base 100)" if language == "es" else "Index (base 100)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "market" / "02_h100_global_china_base100")
        plt.close(fig)

    # MLP and DML learning curves on the untouched test set.
    p = paths.results / "ml" / "final_evaluation" / "aggregate_metrics.csv"
    agg = _read(p)
    if agg is not None and len(agg):
        test = agg.loc[agg["split"].eq("test")].copy()
        if test.empty:
            test = agg.loc[agg["split"].eq("reference")].copy()

        for metric, graph_title, ylabel, filename in [
            ("price_rmse_mean", title["price"], lab["price_rmse"], "02_price_rmse_learning_curve"),
            ("delta_rmse_mean", title["delta"], lab["delta_rmse"], "03_delta_rmse_learning_curve"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for model in ("mlp", "dml"):
                g = test.loc[test["model_type"].eq(model)].sort_values("train_size")
                if g.empty:
                    continue
                ax.plot(
                    g["train_size"],
                    g[metric],
                    color=STRATEGY_COLOR[model],
                    marker="o",
                    linewidth=1.5,
                    label=STRATEGY_LABEL[model],
                )
            ax.set_title(graph_title)
            ax.set_xlabel(lab["training_states"])
            ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            saved += save_figure(fig, paths.figures / "ml" / filename)
            plt.close(fig)

    # Error around scheduled fixing boundaries.
    p = paths.results / "diagnostics" / "fixing_boundary" / "error_by_fixing_boundary_distance.csv"
    b = _read(p)
    if b is not None and len(b):
        x_order = list(dict.fromkeys(b["fixing_boundary_distance_bin"].astype(str).tolist()))
        x_labels = _clean_boundary_labels(x_order, language)
        fig, ax = plt.subplots(figsize=(10, 5))
        for model in ("mlp", "dml"):
            g = b.loc[(b["split"].eq("test")) & (b["model_type"].eq(model))].copy()
            if g.empty:
                continue
            g["fixing_boundary_distance_bin"] = pd.Categorical(
                g["fixing_boundary_distance_bin"].astype(str), categories=x_order, ordered=True
            )
            g = g.sort_values("fixing_boundary_distance_bin")
            ax.plot(
                np.arange(len(g)),
                g["delta_abs_error_mean"],
                color=STRATEGY_COLOR[model],
                marker="o",
                linewidth=1.5,
                label=STRATEGY_LABEL[model],
            )
        ax.set_title(title["boundary"])
        ax.set_xlabel(lab["boundary"])
        ax.set_ylabel("Mean absolute Delta error" if language == "en" else "Error absoluto medio de Delta")
        ax.set_xticks(np.arange(len(x_order)))
        ax.set_xticklabels(x_labels, rotation=20, ha="right")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "ml" / "04_fixing_boundary_delta_error")
        plt.close(fig)

    # Local hedge results for all pre-specified spot shocks.
    p = paths.results / "hedging" / "local" / "final_local_hedge_metrics_pooled.csv"
    local = _read(p)
    if local is not None and len(local):
        fig, ax = plt.subplots(figsize=(10, 5))
        for strategy in ("no_hedge", "oracle_rqmc", "mlp_ensemble", "dml_ensemble"):
            g = local.loc[local["strategy"].eq(strategy)].sort_values("shock_size")
            if g.empty:
                continue
            ax.plot(
                100.0 * g["shock_size"],
                g["rmse"],
                color=STRATEGY_COLOR[strategy],
                marker="o",
                linewidth=1.5,
                label=STRATEGY_LABEL[strategy],
            )
        ax.set_title(title["local"])
        ax.set_xlabel(lab["shock"])
        ax.set_ylabel(lab["local"])
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "hedging" / "05_local_hedge_rmse")
        plt.close(fig)

    # Terminal RMSE for the dynamic monthly forward stress test.
    p = paths.results / "hedging" / "dynamic_forward" / "final_forward_metrics_pooled.csv"
    dyn = _read(p)
    if dyn is not None and len(dyn):
        order = [s for s in ("no_hedge", "oracle_rqmc", "mlp_ensemble", "dml_ensemble") if s in set(dyn["strategy"])]
        x = dyn.set_index("strategy").loc[order]
        pos = np.arange(len(order))
        values = x["rmse"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            pos,
            values,
            color=[STRATEGY_COLOR[s] for s in order],
            width=0.72,
        )
        ax.set_title(title["dynamic"])
        ax.set_xlabel(lab["strategy"])
        ax.set_ylabel(lab["rmse"])
        ax.set_xticks(pos)
        ax.set_xticklabels([STRATEGY_LABEL[s] for s in order])
        ax.grid(True, axis="y")
        offset = max(values) * 0.015 if len(values) else 0.0
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylim(top=max(values) * 1.12)
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "hedging" / "06_dynamic_forward_terminal_rmse")
        plt.close(fig)

    # Tail concentration in dynamic hedge error.
    p = paths.results / "hedging" / "dynamic_forward" / "final_forward_mse_concentration_pooled.csv"
    c = _read(p)
    if c is not None and len(c):
        order = [s for s in ("no_hedge", "oracle_rqmc", "mlp_ensemble", "dml_ensemble") if s in set(c["strategy"])]
        x = c.set_index("strategy").loc[order]
        pos = np.arange(len(order))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            pos - width / 2,
            x["top_1pct_share_squared_error"].to_numpy(),
            width=width,
            color="C0",
            label=lab["top1"],
        )
        ax.bar(
            pos + width / 2,
            x["top_5pct_share_squared_error"].to_numpy(),
            width=width,
            color="C1",
            label=lab["top5"],
        )
        ax.set_title(title["tails"])
        ax.set_ylabel(lab["share"])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(pos)
        ax.set_xticklabels([STRATEGY_LABEL[s] for s in order])
        ax.grid(True, axis="y")
        ax.legend()
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "hedging" / "07_dynamic_tail_concentration")
        plt.close(fig)

    # Delta error by fixing step in the dynamic stress test.
    p = paths.results / "hedging" / "dynamic_forward" / "final_forward_error_by_fixing_step.csv"
    step = _read(p)
    if step is not None and len(step):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            step["n_fix_future"],
            step["delta_rmse_mlp_ensemble_vs_oracle"],
            color=STRATEGY_COLOR["mlp"],
            marker="o",
            linewidth=1.5,
            label="MLP",
        )
        ax.plot(
            step["n_fix_future"],
            step["delta_rmse_dml_ensemble_vs_oracle"],
            color=STRATEGY_COLOR["dml"],
            marker="o",
            linewidth=1.5,
            label="DML",
        )
        ax.set_title(title["dynamic_delta"])
        ax.set_xlabel(lab["fixings"])
        ax.set_ylabel(lab["delta_rmse"])
        ax.invert_xaxis()
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        saved += save_figure(fig, paths.figures / "hedging" / "08_dynamic_delta_error_by_fixing")
        plt.close(fig)

    print(f"Generated {len(saved)} figure files.")
    for p in saved:
        print(f"  {p.relative_to(paths.root)}")
    return saved


def make_tables(paths: ProjectPaths) -> list[Path]:
    paths.tables.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    agg = _read(paths.results / "ml" / "final_evaluation" / "aggregate_metrics.csv")
    if agg is not None:
        final_ml = agg.loc[agg["train_size"].eq(65536), [
            "split", "model_type", "price_rmse_mean", "price_mae_mean",
            "delta_rmse_mean", "delta_mae_mean", "delta_r2_mean",
            "training_seconds_mean",
        ]].copy()
        p = paths.tables / "01_final_ml_metrics.csv"
        final_ml.to_csv(p, index=False)
        saved.append(p)
        try:
            tex = final_ml.to_latex(index=False, float_format=lambda x: f"{x:.6g}")
            ptex = paths.tables / "01_final_ml_metrics.tex"
            ptex.write_text(tex, encoding="utf-8")
            saved.append(ptex)
        except Exception as exc:
            print(f"WARNING: could not write LaTeX ML table: {exc}")

    local = _read(paths.results / "hedging" / "local" / "final_local_hedge_metrics_pooled.csv")
    if local is not None:
        primary = local.loc[np.isclose(local["shock_size"], 0.01)].copy()
        p = paths.tables / "02_local_hedge_primary_1pct.csv"
        primary.to_csv(p, index=False)
        saved.append(p)

    dyn = _read(paths.results / "hedging" / "dynamic_forward" / "final_forward_metrics_pooled.csv")
    if dyn is not None:
        p = paths.tables / "03_dynamic_forward_metrics.csv"
        dyn.to_csv(p, index=False)
        saved.append(p)

    print(f"Generated {len(saved)} table files.")
    for p in saved:
        print(f"  {p.relative_to(paths.root)}")
    return saved
