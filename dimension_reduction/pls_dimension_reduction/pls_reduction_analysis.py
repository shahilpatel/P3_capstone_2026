from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


LOC_COL_DEFAULT = "Shot.Location"
TYPE_COL_DEFAULT = "Shot.Type"
PLAYER_COL_DEFAULT = "PlayerID"
TARGET_COL_DEFAULT = "Made"

try:
    from IPython.display import Markdown as _IPyMarkdown  # type: ignore
    from IPython.display import display as _ipydisplay  # type: ignore

    _HAS_IPYTHON = True
except Exception:
    _HAS_IPYTHON = False


def _md(text: str) -> None:
    if _HAS_IPYTHON:
        _ipydisplay(_IPyMarkdown(text))
    else:
        print(text)


def _show(obj) -> None:
    if _HAS_IPYTHON:
        _ipydisplay(obj)
    else:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_dataset(path_candidates: list[Path]) -> Path:
    for p in path_candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not find dataset. Tried: " + ", ".join(str(p) for p in path_candidates))


def numeric_predictor_columns(
    df: pd.DataFrame,
    *,
    target_col: str,
    exclude_norm_keys: set[str],
) -> tuple[list[str], list[str]]:
    cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in cols:
        cols.remove(target_col)
    excluded = [c for c in cols if _norm_col(c) in exclude_norm_keys]
    cols = [c for c in cols if c not in excluded]
    return cols, excluded


def filter_players_with_min_shots(frame: pd.DataFrame, *, player_col: str, min_shots: int) -> pd.DataFrame:
    counts = frame[player_col].value_counts(dropna=False)
    keep = counts[counts >= min_shots].index
    return frame[frame[player_col].isin(keep)].copy()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pls_component_definition_table(pipe: Pipeline, *, feature_cols: list[str]) -> pd.DataFrame:
    imputer: SimpleImputer = pipe.named_steps["imputer"]
    scaler: StandardScaler = pipe.named_steps["scaler"]
    pls: PLSRegression = pipe.named_steps["pls"]

    w = pls.x_weights_
    comps = [f"PLS{i+1}" for i in range(w.shape[1])]
    weights = pd.DataFrame(w, index=feature_cols, columns=comps)
    base = pd.DataFrame(
        {
            "feature": feature_cols,
            "impute_median": imputer.statistics_,
            "scale_mean": scaler.mean_,
            "scale_std": scaler.scale_,
        }
    ).set_index("feature")
    return base.join(weights)


def pls_vip_scores(pipe: Pipeline, *, feature_cols: list[str]) -> pd.Series:
    pls: PLSRegression = pipe.named_steps["pls"]
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_

    p = w.shape[0]
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if q.shape[0] != 1:
        q = q[:1, :]

    ssy = np.sum(t**2, axis=0) * np.sum(q**2, axis=0)
    denom = np.sum(ssy)
    if denom <= 0:
        return pd.Series(np.nan, index=feature_cols, name="VIP")

    wnorm2 = np.sum(w**2, axis=0)
    wnorm2 = np.where(wnorm2 == 0, np.nan, wnorm2)
    vip = np.sqrt(p * np.nansum((ssy * (w**2 / wnorm2)), axis=1) / denom)
    return pd.Series(vip, index=feature_cols, name="VIP").sort_values(ascending=False)


class PLSXTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.model_ = PLSRegression(n_components=self.n_components)
        if y is None:
            raise ValueError("PLSXTransformer requires y for supervised fitting.")
        self.model_.fit(X, y)
        return self

    def transform(self, X):
        return self.model_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=y)
        return self.transform(X)


def evaluate_pls_classifier_cv(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    n_components_fit: int,
    cv_mode: str,
    group_col: str,
    cv_max_splits: int = 5,
) -> dict:
    X = frame[feature_cols]
    y = frame[target_col].astype(int)

    class_counts = y.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        return {"ok": False, "reason": f"Not enough class balance for CV metrics (counts={class_counts.to_dict()})."}

    cv_mode = str(cv_mode).strip().lower()
    if cv_mode not in {"stratified_shot", "group_by_player"}:
        return {
            "ok": False,
            "reason": f"Unknown cv_mode='{cv_mode}'. Use 'stratified_shot' or 'group_by_player'.",
        }

    groups = None
    n_groups = None
    if cv_mode == "group_by_player":
        if group_col not in frame.columns:
            return {"ok": False, "reason": f"Group column '{group_col}' not found for group CV."}
        groups = frame[group_col]
        n_groups = int(pd.Series(groups).nunique(dropna=True))
        n_splits = int(min(cv_max_splits, n_groups))
        if n_splits < 2:
            return {"ok": False, "reason": f"Not enough groups for GroupKFold (n_groups={n_groups})."}
        cv = GroupKFold(n_splits=n_splits)
    else:
        n_splits = int(min(cv_max_splits, class_counts.min()))
        if n_splits < 2:
            return {"ok": False, "reason": f"Not enough samples per class for CV splits (counts={class_counts.to_dict()})."}
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("plsx", PLSXTransformer(n_components=n_components_fit)),
            ("clf", LogisticRegression(max_iter=5000, solver="liblinear")),
        ]
    )

    try:
        prob = cross_val_predict(pipe, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    except Exception as e:
        return {
            "ok": False,
            "reason": f"CV failed for cv_mode='{cv_mode}': {e}",
        }
    pred = (prob >= 0.5).astype(int)
    base_prob = np.full_like(prob, float(y.mean()), dtype=float)

    return {
        "ok": True,
        "cv_mode": cv_mode,
        "cv_splits": n_splits,
        "n_groups": n_groups,
        "roc_auc": float(roc_auc_score(y, prob)),
        "avg_precision": float(average_precision_score(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prob)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "baseline_log_loss": float(log_loss(y, base_prob, labels=[0, 1])),
        "baseline_brier": float(brier_score_loss(y, base_prob)),
        "made_rate": float(y.mean()),
    }


@dataclass(frozen=True)
class PLSFitResult:
    ok: bool
    reason: str | None
    n_components_requested: int
    n_components_fit: int
    pipeline: Pipeline | None
    definition: pd.DataFrame | None
    vip: pd.Series | None
    ranking: pd.DataFrame | None
    scores: pd.DataFrame | None
    cv_eval: dict | None


def fit_pls_and_rank(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    n_components_requested: int,
    run_cv_eval: bool,
    cv_max_splits: int,
    cv_mode: str,
    group_col: str,
) -> PLSFitResult:
    X = frame[feature_cols]
    y = frame[target_col].astype(int).to_numpy().reshape(-1, 1)

    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_components_fit = max(0, min(n_components_requested, n_features, n_samples - 1))
    if n_components_fit < 1:
        return PLSFitResult(
            ok=False,
            reason=f"Not enough data to fit PLS (n_samples={n_samples}, n_features={n_features}).",
            n_components_requested=n_components_requested,
            n_components_fit=0,
            pipeline=None,
            definition=None,
            vip=None,
            ranking=None,
            scores=None,
            cv_eval=None,
        )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=n_components_fit)),
        ]
    )
    pipe.fit(X, y)

    X_pre = pipe[:-1].transform(X)
    x_scores = pipe.named_steps["pls"].transform(X_pre)

    vip = pls_vip_scores(pipe, feature_cols=feature_cols)
    definition = pls_component_definition_table(pipe, feature_cols=feature_cols)

    y_vec = y.ravel()
    rows: list[dict] = []
    for j in range(x_scores.shape[1]):
        comp = f"PLS{j+1}"
        corr = safe_corr(x_scores[:, j], y_vec)
        rows.append(
            {
                "component": comp,
                "corr_with_made": corr,
                "abs_corr_with_made": np.nan if np.isnan(corr) else abs(corr),
                "score_variance": float(np.var(x_scores[:, j], ddof=1)) if x_scores.shape[0] > 1 else float("nan"),
            }
        )
    ranking = (
        pd.DataFrame(rows)
        .sort_values(["abs_corr_with_made", "score_variance"], ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    scores_df = pd.DataFrame(x_scores, columns=[f"PLS{i+1}" for i in range(x_scores.shape[1])])

    cv_eval = None
    if run_cv_eval:
        cv_eval = evaluate_pls_classifier_cv(
            frame,
            feature_cols=feature_cols,
            target_col=target_col,
            n_components_fit=n_components_fit,
            cv_mode=cv_mode,
            group_col=group_col,
            cv_max_splits=cv_max_splits,
        )

    return PLSFitResult(
        ok=True,
        reason=None,
        n_components_requested=n_components_requested,
        n_components_fit=n_components_fit,
        pipeline=pipe,
        definition=definition,
        vip=vip,
        ranking=ranking,
        scores=scores_df,
        cv_eval=cv_eval,
    )


def plot_top_vip(vip: pd.Series, *, title: str, top_n: int = 15) -> None:
    top = vip.head(top_n)
    plt.figure(figsize=(7, 6))
    sns.barplot(x=top.values[::-1], y=top.index[::-1], color="#4C78A8")
    plt.title(title)
    plt.xlabel("VIP")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()


def run_pls_analysis(
    *,
    data_path: Path | None = None,
    loc_col: str = LOC_COL_DEFAULT,
    type_col: str = TYPE_COL_DEFAULT,
    player_col: str = PLAYER_COL_DEFAULT,
    target_col: str = TARGET_COL_DEFAULT,
    min_shots_per_player: int = 20,
    n_components_requested: list[int] = (10, 20),
    run_cv_eval: bool = True,
    cv_max_splits: int = 5,
    cv_mode: str = "group_by_player",
    exclude_norm_keys: set[str] | None = None,
    output_dir: Path = Path("outputs"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_path is None:
        data_path = find_dataset([Path("capstone2026.csv"), Path("..") / "capstone2026.csv"])

    df = pd.read_csv(data_path)
    for col in (loc_col, type_col, player_col, target_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if exclude_norm_keys is None:
        exclude_norm_keys = {"balldepth", "balldistancefromcenter", "b"}

    feature_cols, excluded = numeric_predictor_columns(df, target_col=target_col, exclude_norm_keys=exclude_norm_keys)
    # Ensure IDs never appear as predictors, even if they were numeric in a future export
    feature_cols = [c for c in feature_cols if c != player_col]
    shot_locations = sorted(df[loc_col].dropna().unique().tolist())

    vip_tables: dict[tuple[str, str, int], pd.Series] = {}
    perf_rows: list[dict] = []

    _md(f"**Loaded:** `{data_path}`")
    _md(f"**Excluded predictors found:** `{excluded if excluded else 'None found'}`")
    _md(f"**Numeric predictors used:** `{len(feature_cols)}`")

    for location in shot_locations:
        _md(f"## Shot Location: {location}")

        if location == "Free Throw":
            shot_type_values = [None]
        else:
            shot_type_values = ["Off the Dribble", "Catch and Shoot"]

        for shot_type in shot_type_values:
            st_label = "All" if shot_type is None else shot_type
            _md(f"### Shot Type: {st_label}")

            if shot_type is None:
                subset = df[df[loc_col] == location].copy()
            else:
                subset = df[(df[loc_col] == location) & (df[type_col] == shot_type)].copy()

            subset = filter_players_with_min_shots(subset, player_col=player_col, min_shots=min_shots_per_player)
            _show(
                pd.DataFrame(
                    {
                        "rows_after_filter": [len(subset)],
                        "players_after_filter": [subset[player_col].nunique(dropna=True)],
                        "made_rate": [float(subset[target_col].astype(int).mean()) if len(subset) else float("nan")],
                    }
                )
            )
            if len(subset) == 0:
                continue

            for k in n_components_requested:
                _md(f"#### PLS model: {k} components")
                fit = fit_pls_and_rank(
                    subset,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    n_components_requested=int(k),
                    run_cv_eval=run_cv_eval,
                    cv_max_splits=cv_max_splits,
                    cv_mode=cv_mode,
                    group_col=player_col,
                )
                if not fit.ok:
                    _md(f"_Skipped:_ {fit.reason}")
                    continue

                if fit.n_components_fit != fit.n_components_requested:
                    _md(
                        f"_Note:_ requested `{fit.n_components_requested}` components, fit `{fit.n_components_fit}` due to sample/feature limits."
                    )

                if run_cv_eval and fit.cv_eval is not None:
                    if fit.cv_eval.get("ok"):
                        perf_rows.append(
                            {
                                "location": location,
                                "shot_type": st_label,
                                "k": int(k),
                                **fit.cv_eval,
                            }
                        )
                        _md("**Out-of-sample predictive quality (CV):**")
                        _show(pd.DataFrame([fit.cv_eval]))
                    else:
                        _md(f"_CV metrics skipped:_ {fit.cv_eval.get('reason')}")

                assert fit.ranking is not None
                _md("**Ranking of new PLS variables (top = strongest |corr with Made|):**")
                _show(fit.ranking)

                assert fit.definition is not None and fit.vip is not None
                out_base = f"{_slug(location)}__{_slug(st_label)}__k{k}"
                def_path = output_dir / f"pls_definition__{out_base}.csv"
                vip_path = output_dir / f"pls_vip__{out_base}.csv"
                fit.definition.to_csv(def_path)
                fit.vip.to_frame().to_csv(vip_path)
                _md(f"**Component definitions (exact weights) saved:** `{def_path}`")
                _md(f"**VIP scores saved:** `{vip_path}`")

                vip_tables[(location, st_label, int(k))] = fit.vip

                _md("**Top 20 variables by VIP:**")
                _show(fit.vip.head(20).to_frame())
                plot_top_vip(
                    fit.vip,
                    title=f"Top VIP variables\n{location} | {st_label} | k={fit.n_components_fit}",
                    top_n=15,
                )

                _md("**Component makeup (preview): top contributors for PLS1–PLS3 by |weight|**")
                weight_cols = [c for c in fit.definition.columns if c.startswith("PLS")]
                for comp in weight_cols[:3]:
                    top = (
                        fit.definition[comp]
                        .abs()
                        .sort_values(ascending=False)
                        .head(15)
                        .to_frame("abs_weight")
                        .join(fit.definition[[comp]])
                        .reset_index(names="feature")
                    )
                    _md(f"- **{comp}**")
                    _show(top)

    perf_df = pd.DataFrame(perf_rows) if perf_rows else pd.DataFrame()
    if not perf_df.empty:
        perf_path = output_dir / "pls_cv_performance_summary.csv"
        perf_df.to_csv(perf_path, index=False)
        _md(f"**Saved CV performance summary:** `{perf_path}`")

    return {
        "data_path": data_path,
        "excluded_predictors_found": excluded,
        "feature_cols": feature_cols,
        "vip_tables": vip_tables,
        "perf_df": perf_df,
        "output_dir": output_dir,
    }


def cross_section_plots(results: dict, *, n_components_requested: list[int] = (10, 20)) -> None:
    vip_tables: dict[tuple[str, str, int], pd.Series] = results.get("vip_tables", {})
    perf_df: pd.DataFrame = results.get("perf_df", pd.DataFrame())

    _md("# Cross-Section Visualizations")
    _md("These plots use VIP as a proxy for contribution. Treat “significant” as **consistently high VIP across subsets**.")

    def vip_matrix_for_k(k: int) -> pd.DataFrame:
        cols: dict[str, pd.Series] = {}
        for (loc, st, kk), vip in vip_tables.items():
            if kk != k:
                continue
            cols[f"{loc} | {st}"] = vip
        if not cols:
            return pd.DataFrame()
        return pd.DataFrame(cols)

    for k in n_components_requested:
        vip_mat = vip_matrix_for_k(int(k))
        if vip_mat.empty:
            _md(f"## VIP summary (k={k})\n_No models available._")
            continue

        vip_mean = vip_mat.mean(axis=1).sort_values(ascending=False)
        top_vars = vip_mean.head(30).index
        vip_top = vip_mat.loc[top_vars]

        _md(f"## VIP summary (k={k})")
        _show(pd.DataFrame({"mean_VIP": vip_mean.head(30)}))

        plt.figure(figsize=(min(20, 0.45 * vip_top.shape[1] + 6), 10))
        sns.heatmap(vip_top, cmap="viridis", cbar_kws={"label": "VIP"})
        plt.title(f"VIP heatmap (top 30 variables) | k={k}")
        plt.xlabel("Shot Location | Shot Type")
        plt.ylabel("Variable")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=vip_mean.head(20).values[::-1], y=vip_mean.head(20).index[::-1], color="#F58518")
        plt.title(f"Overall contribution (mean VIP, top 20) | k={k}")
        plt.xlabel("Mean VIP")
        plt.ylabel("Variable")
        plt.tight_layout()
        plt.show()

    if not perf_df.empty:
        for k in n_components_requested:
            perf_k = perf_df[(perf_df["k"] == int(k)) & (perf_df["ok"] == True)].copy()
            if perf_k.empty:
                continue

            perf_k["subset"] = perf_k["location"] + " | " + perf_k["shot_type"]
            perf_k = perf_k.set_index("subset")

            plt.figure(figsize=(10, 7))
            top = perf_k.sort_values("roc_auc", ascending=False).head(20)
            sns.barplot(x=top["roc_auc"].values[::-1], y=top.index.values[::-1], color="#54A24B")
            plt.title(f"ROC AUC by subset (top 20) | k={k}")
            plt.xlabel("ROC AUC")
            plt.ylabel("Shot Location | Shot Type")
            plt.tight_layout()
            plt.show()

            try:
                pivot_auc = perf_k.reset_index().pivot(index="location", columns="shot_type", values="roc_auc")
                plt.figure(figsize=(10, max(6, 0.4 * len(pivot_auc))))
                sns.heatmap(pivot_auc, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "ROC AUC"})
                plt.title(f"ROC AUC heatmap | k={k}")
                plt.xlabel("Shot Type")
                plt.ylabel("Shot Location")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                _md(f"_Could not build AUC heatmap:_ {e}")
