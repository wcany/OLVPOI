import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
)

from args import parameter_parser
from dataloader import (
    FeaturePreprocessor,
    build_feature_schema,
    construct_training_adjacency,
    create_fold_assignments,
    load_feature_table,
)
from train import compute_binary_metrics, predict, set_seed, train


N_SPLITS = 5
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 2026
CALIBRATION_EPS = 1e-6
METRIC_NAMES = (
    "ROC_AUC",
    "Accuracy",
    "F1",
    "Sensitivity",
    "Specificity",
    "PR_AUC",
    "Average_Precision",
    "PPV",
    "NPV",
    "Brier_Score",
    "Calibration_Intercept",
    "Calibration_Slope",
    "Log_Loss",
)
PERCENT_METRICS = {
    "ROC_AUC",
    "Accuracy",
    "F1",
    "Sensitivity",
    "Specificity",
    "PR_AUC",
    "Average_Precision",
    "PPV",
    "NPV",
}
METRIC_LABELS = {
    "ROC_AUC": "ROC-AUC",
    "Accuracy": "Accuracy",
    "F1": "F1",
    "Sensitivity": "Sensitivity",
    "Specificity": "Specificity",
    "PR_AUC": "PR-AUC",
    "Average_Precision": "Average Precision",
    "PPV": "PPV",
    "NPV": "NPV",
    "Brier_Score": "Brier Score",
    "Calibration_Intercept": "Calibration Intercept",
    "Calibration_Slope": "Calibration Slope",
    "Log_Loss": "Log-loss",
}


def select_device(config):
    if config.use_cuda and torch.cuda.is_available():
        return torch.device(f"cuda:{config.cuda_device}")
    return torch.device("cpu")


def prepare_training_domain(config, raw_features, schema, train_indices):
    preprocessor = FeaturePreprocessor(schema).fit(raw_features.iloc[train_indices])
    training_features = preprocessor.transform(raw_features.iloc[train_indices])
    adjacency = construct_training_adjacency(
        training_features,
        k=config.k,
        algorithm="ball_tree",
        metric="minkowski",
        minkowski_p=2,
        minimum_common_neighbors=2,
    )
    return preprocessor, training_features, adjacency


def run_outer_fold(
    config,
    raw_features,
    labels,
    schema,
    train_indices,
    test_indices,
    fold,
    device,
):
    if np.intersect1d(train_indices, test_indices).size:
        raise AssertionError("Outer train and test samples overlap.")
    preprocessor, training_features, adjacency = prepare_training_domain(
        config, raw_features, schema, train_indices
    )
    test_features = preprocessor.transform(raw_features.iloc[test_indices])
    actual_model_seed = config.model_seed + fold
    set_seed(actual_model_seed)
    model = train(
        config,
        training_features,
        labels[train_indices],
        adjacency,
        device,
        seed=actual_model_seed,
        progress_description=f"Fold {fold + 1}/5 OLVPOI",
    )
    probabilities = predict(model, test_features, device)
    return pd.DataFrame(
        {
            "sample_id": test_indices,
            "fold": fold + 1,
            "label": labels[test_indices],
            "probability": probabilities,
        }
    )


def calibration_intercept_slope(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    clipped = np.clip(probabilities, CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(logits, labels)
        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0, 0])
    except (ValueError, FloatingPointError, ConvergenceWarning):
        return np.nan, np.nan
    if not np.isfinite([intercept, slope]).all():
        return np.nan, np.nan
    return intercept, slope


def compute_extended_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    ppv = float(tp / (tp + fp)) if tp + fp else np.nan
    npv = float(tn / (tn + fn)) if tn + fn else np.nan
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    intercept, slope = calibration_intercept_slope(labels, probabilities)
    clipped = np.clip(probabilities, CALIBRATION_EPS, 1.0 - CALIBRATION_EPS)
    return {
        **compute_binary_metrics(labels, probabilities, threshold),
        "PR_AUC": float(auc(recall, precision)),
        "Average_Precision": float(
            average_precision_score(labels, probabilities)
        ),
        "PPV": ppv,
        "NPV": npv,
        "Brier_Score": float(brier_score_loss(labels, probabilities)),
        "Calibration_Intercept": intercept,
        "Calibration_Slope": slope,
        "Log_Loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def build_fold_metrics(predictions, threshold):
    rows = []
    for fold, frame in predictions.groupby("fold", sort=True):
        rows.append(
            {
                "fold": int(fold),
                **compute_extended_metrics(
                    frame["label"], frame["probability"], threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_fold_metrics(fold_metrics):
    summary = {}
    for metric in METRIC_NAMES:
        values = fold_metrics[metric].to_numpy(dtype=float)
        summary[metric] = (
            float(np.nanmean(values)),
            float(np.nanstd(values, ddof=1)),
            int(np.isfinite(values).sum()),
        )
    return summary


def bootstrap_oof_metrics(predictions, threshold):
    labels = predictions["label"].to_numpy(dtype=np.int64)
    probabilities = predictions["probability"].to_numpy(dtype=float)
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    if not len(negative_indices) or not len(positive_indices):
        raise ValueError("Both classes are required for OOF confidence intervals.")

    estimate = compute_extended_metrics(labels, probabilities, threshold)
    bootstrap_values = {metric: [] for metric in METRIC_NAMES}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPEATS):
        sampled_indices = np.concatenate(
            [
                generator.choice(
                    negative_indices, size=len(negative_indices), replace=True
                ),
                generator.choice(
                    positive_indices, size=len(positive_indices), replace=True
                ),
            ]
        )
        sampled = compute_extended_metrics(
            labels[sampled_indices], probabilities[sampled_indices], threshold
        )
        for metric in METRIC_NAMES:
            bootstrap_values[metric].append(sampled[metric])

    output = {}
    for metric in METRIC_NAMES:
        values = np.asarray(bootstrap_values[metric], dtype=float)
        valid = values[np.isfinite(values)]
        if not len(valid):
            lower = upper = np.nan
        else:
            lower, upper = np.percentile(valid, [2.5, 97.5])
        output[metric] = (
            estimate[metric],
            float(lower),
            float(upper),
            int(len(valid)),
        )
    return output


def format_value(metric, value):
    if not np.isfinite(value):
        return "NaN"
    if metric in PERCENT_METRICS:
        return f"{value * 100:.2f}"
    return f"{value:.6f}"


def format_mean_std(metric, mean, std):
    if not np.isfinite(mean) or not np.isfinite(std):
        return "NaN"
    if metric in PERCENT_METRICS:
        return f"{mean * 100:.2f} ({std * 100:.2f})"
    return f"{mean:.6f} ({std:.6f})"


def format_estimate_ci(metric, estimate, lower, upper):
    if not np.isfinite([estimate, lower, upper]).all():
        return "NaN"
    if metric in PERCENT_METRICS:
        return f"{estimate * 100:.2f} ({lower * 100:.2f}-{upper * 100:.2f})"
    return f"{estimate:.6f} ({lower:.6f} to {upper:.6f})"


def write_metrics_results(
    path,
    config,
    device,
    num_samples,
    num_features,
    fold_metrics,
    fold_summary,
    oof_summary,
):
    lines = [
        datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "OLVPOI extended metrics on five-fold OOF predictions",
        (
            f"epoch={config.epochs} | learning_rate={config.learning_rate} | "
            f"weight_decay={config.weight_decay} | optimizer={config.optimizer}"
        ),
        (
            f"hidden_dim={config.hidden_dim} | latent_dim={config.latent_dim} | "
            f"dropout={config.dropout} | alpha={config.alpha} | "
            f"beta={config.beta} | k={config.k}"
        ),
        (
            f"threshold={config.classification_threshold} | cv_seed={config.cv_seed} | "
            f"model_seed={config.model_seed}+fold | device={device}"
        ),
        (
            f"bootstrap={BOOTSTRAP_REPEATS} | bootstrap_seed={BOOTSTRAP_SEED} | "
            "fixed OOF predictions resampled; models were not retrained"
        ),
        (
            "Calibration evaluation=LogisticRegression(penalty=None, solver=lbfgs, "
            f"max_iter=1000) on logit(clip(p, {CALIBRATION_EPS}, "
            f"{1.0 - CALIBRATION_EPS}))"
        ),
        f"samples={num_samples} | features={num_features} | folds={N_SPLITS} stratified",
        "PR-AUC uses trapezoidal integration; Average Precision is reported separately.",
        "Per-fold calibration intercept/slope are descriptive because each fold is small.",
        "",
        "Per-fold metrics",
        (
            f"{'Metric':<26}"
            + "".join(f"Fold {fold}".rjust(16) for fold in range(1, N_SPLITS + 1))
            + f"{'Mean (SD)':>24}{'Valid folds':>14}"
        ),
    ]
    for metric in METRIC_NAMES:
        mean, std, valid = fold_summary[metric]
        lines.append(
            f"{METRIC_LABELS[metric]:<26}"
            + "".join(
                format_value(metric, row[metric]).rjust(16)
                for _, row in fold_metrics.iterrows()
            )
            + format_mean_std(metric, mean, std).rjust(24)
            + f"{valid:>14}"
        )

    lines.extend(
        [
            "",
            "Pooled OOF estimate (95% percentile CI)",
            f"{'Metric':<26}{'Estimate (95% CI)':>40}{'Valid bootstrap':>20}",
        ]
    )
    for metric in METRIC_NAMES:
        estimate, lower, upper, valid = oof_summary[metric]
        lines.append(
            f"{METRIC_LABELS[metric]:<26}"
            f"{format_estimate_ci(metric, estimate, lower, upper):>40}"
            f"{f'{valid}/{BOOTSTRAP_REPEATS}':>20}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = "-" * 180
    with path.open("a", encoding="utf-8") as stream:
        if path.stat().st_size > 0:
            stream.write(f"\n{separator}\n")
        stream.write("\n".join(lines) + "\n")


def main():
    warnings.filterwarnings("ignore")
    config = parameter_parser()
    set_seed(config.model_seed)
    device = select_device(config)
    raw_features, labels = load_feature_table(config.data_path)
    schema = build_feature_schema(raw_features)
    fold_assignments = create_fold_assignments(labels, N_SPLITS, config.cv_seed)
    all_indices = np.arange(len(labels), dtype=int)

    fold_predictions = []
    for fold in range(N_SPLITS):
        test_indices = all_indices[fold_assignments == fold]
        train_indices = all_indices[fold_assignments != fold]
        fold_predictions.append(
            run_outer_fold(
                config,
                raw_features,
                labels,
                schema,
                train_indices,
                test_indices,
                fold,
                device,
            )
        )

    predictions = (
        pd.concat(fold_predictions, ignore_index=True)
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    if len(predictions) != len(labels):
        raise AssertionError("OOF predictions do not contain all patients.")
    if predictions["sample_id"].duplicated().any():
        raise AssertionError("OOF predictions contain duplicate patients.")
    if not np.array_equal(predictions["sample_id"].to_numpy(int), all_indices):
        raise AssertionError("OOF predictions do not cover each patient exactly once.")
    if not np.array_equal(predictions["label"].to_numpy(int), labels):
        raise AssertionError("OOF labels differ from source labels.")
    if not np.array_equal(
        predictions["fold"].to_numpy(int), fold_assignments + 1
    ):
        raise AssertionError("OOF fold assignments differ from the configured split.")

    fold_metrics = build_fold_metrics(predictions, config.classification_threshold)
    fold_summary = summarize_fold_metrics(fold_metrics)
    oof_summary = bootstrap_oof_metrics(
        predictions, config.classification_threshold
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    predictions["pred_05"] = (
        predictions["probability"] >= config.classification_threshold
    ).astype(int)
    predictions_path = config.results_dir / "oof_predictions.csv"
    metrics_path = config.results_dir / "metrics_results.txt"
    predictions.loc[
        :, ["sample_id", "fold", "label", "probability", "pred_05"]
    ].to_csv(predictions_path, index=False, float_format="%.10f")
    write_metrics_results(
        metrics_path,
        config,
        device,
        num_samples=len(labels),
        num_features=raw_features.shape[1],
        fold_metrics=fold_metrics,
        fold_summary=fold_summary,
        oof_summary=oof_summary,
    )
    print(f"Device: {device}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()
