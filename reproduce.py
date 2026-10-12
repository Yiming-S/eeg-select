"""Reanalyse the saved calibration and participant-level MI-EEG results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASETS = ("physionet", "cho2017", "bnci004")
PARTICIPANTS = {"physionet": 109, "cho2017": 50, "bnci004": 9}
SPLIT = ["dataset", "repeat", "fold"]
KEYS = SPLIT + ["subject"]
USES = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70)
SEED = 20260810
COMPARISONS = (
    ("complete_system_vs_gain_only", "direct_argmax", "legacy_gain_only"),
    ("default_preserving_vs_argmax", "stability_aware_direct", "direct_argmax"),
    ("fixed_u20_vs_argmax", "fixed_u20", "direct_argmax"),
    ("default_preserving_vs_fixed_u20", "stability_aware_direct", "fixed_u20"),
    ("grid_diagnostic_vs_fixed_u30", "stability_aware_expanded", "fixed_u30"),
    ("grid_diagnostic_vs_fixed_u40", "stability_aware_expanded", "fixed_u40"),
    ("fixed_u40_vs_fixed_u20", "fixed_u40", "fixed_u20"),
)


def verify_files():
    records = json.loads((ROOT / "provenance.json").read_text())
    for item in records["files"]:
        path = ROOT / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"Checksum differs: {item['path']}")


def read(name):
    return pd.read_csv(ROOT / "data" / name)


def check_support(frame):
    if frame.duplicated(KEYS).any():
        raise ValueError("Repeated participant/split rows")
    for dataset, group in frame.groupby("dataset", sort=False):
        if group.subject.nunique() != PARTICIPANTS[dataset]:
            raise ValueError(f"Participant coverage differs: {dataset}")
        matrix = group.pivot(index="subject", columns="repeat", values="accuracy")
        if matrix.shape[1] != 10 or matrix.isna().any().any():
            raise ValueError(f"Incomplete repeat coverage: {dataset}")
        expected_fits = 90 if dataset == "bnci004" else 50
        if len(group[SPLIT].drop_duplicates()) != expected_fits:
            raise ValueError(f"Outer-fit coverage differs: {dataset}")


def replay_selection(calibration, configurations, decisions, policies):
    """Select from calibration alone, then read the matching test results."""
    if set(calibration.partition) != {"calibration"}:
        raise ValueError("Selection input must contain calibration rows only")
    choices = []
    for key, group in calibration.groupby(SPLIT, sort=True):
        if len(group) != 12 or group.configuration.nunique() != 12:
            raise ValueError(f"Incomplete configuration library: {key}")
        feasible = group[group.normalized_cost <= 0.75 + 1e-12]
        if feasible.empty:
            raise ValueError(f"No feasible configuration: {key}")
        best = feasible.sort_values(
            ["accuracy", "normalized_cost", "configuration"],
            ascending=[False, True, True], kind="stable",
        ).iloc[0]
        choices.append(dict(zip(SPLIT, key), configuration=best.configuration))
    chosen = pd.DataFrame(choices)
    recorded = decisions[decisions.selector == "direct_argmax"]
    check = chosen.merge(recorded[SPLIT + ["selected_configuration"]],
                         on=SPLIT, validate="one_to_one")
    if len(check) != 190 or not (check.configuration == check.selected_configuration).all():
        raise ValueError("Calibration replay differs from the recorded decisions")
    selected = chosen.merge(configurations, on=SPLIT + ["configuration"],
                            validate="one_to_many")
    recorded_test = policies[policies.policy == "direct_argmax"]
    check = selected.merge(recorded_test[KEYS + ["accuracy"]], on=KEYS,
                           suffixes=("_replayed", "_recorded"), validate="one_to_one")
    if len(check) != 1680 or not np.allclose(check.accuracy_replayed,
                                           check.accuracy_recorded, atol=1e-12, rtol=0):
        raise ValueError("Held-out selection replay differs")
    return chosen


def crossed_draws(frame, metric, count, seed):
    matrix = frame.pivot(index="subject", columns="repeat", values=metric)
    matrix = matrix.sort_index().sort_index(axis=1).to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(count)
    for index in range(count):
        subjects = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        repeats = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        samples[index] = matrix[np.ix_(subjects, repeats)].mean()
    return samples


def outer_fit_draws(frame, metric, count, seed):
    subjects = sorted(frame.subject.astype(int).unique())
    fits = sorted(set(zip(frame.repeat.astype(int), frame.fold.astype(int))))
    subject_ids = {value: index for index, value in enumerate(subjects)}
    fit_ids = {value: index for index, value in enumerate(fits)}
    row_subject = np.array([subject_ids[int(value)] for value in frame.subject])
    row_fit = np.array([fit_ids[(int(r), int(f))] for r, f in zip(frame.repeat, frame.fold)])
    values = frame[metric].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(count)
    for index in range(count):
        subject_counts = np.bincount(rng.integers(0, len(subjects), len(subjects)),
                                    minlength=len(subjects))
        fit_counts = np.bincount(rng.integers(0, len(fits), len(fits)), minlength=len(fits))
        weights = subject_counts[row_subject] * fit_counts[row_fit]
        samples[index] = np.dot(weights, values) / weights.sum() if weights.sum() else values.mean()
    return samples


def interval(frame, metric, count, seed, dataset):
    check_support(frame)
    crossed = crossed_draws(frame, metric, count, seed)
    outer = crossed if dataset == "bnci004" else outer_fit_draws(frame, metric, count, seed + 1)
    return {
        "mean": float(frame[metric].mean()),
        "crossed_ci_lower": float(np.quantile(crossed, .025)),
        "crossed_ci_upper": float(np.quantile(crossed, .975)),
        "participant_outer_fit_ci_lower": float(np.quantile(outer, .025)),
        "participant_outer_fit_ci_upper": float(np.quantile(outer, .975)),
    }


def policy_frame(policies, fixed, dataset, name):
    if name.startswith("fixed_u"):
        use = int(name.removeprefix("fixed_u")) / 100
        return fixed[(fixed.dataset == dataset) & np.isclose(fixed.requested_use, use)]
    return policies[(policies.dataset == dataset) & (policies.policy == name)]


def contrasts(policies, fixed, count):
    rows = []
    for di, dataset in enumerate(DATASETS):
        for ci, (name, selected_name, comparator_name) in enumerate(COMPARISONS):
            selected = policy_frame(policies, fixed, dataset, selected_name)
            comparator = policy_frame(policies, fixed, dataset, comparator_name)
            check_support(selected)
            check_support(comparator)
            difference = selected[KEYS + ["accuracy"]].merge(
                comparator[KEYS + ["accuracy"]], on=KEYS,
                suffixes=("_selected", "_comparator"), validate="one_to_one")
            if len(difference) != len(selected) or len(difference) != len(comparator):
                raise ValueError("A paired contrast lost rows")
            difference["accuracy"] = difference.accuracy_selected - difference.accuracy_comparator
            seed = SEED + 2_000_000 + di * 100_000 + ci * 1_000
            if name in ("grid_diagnostic_vs_fixed_u30", "grid_diagnostic_vs_fixed_u40"):
                ui = USES.index(.30 if name.endswith("u30") else .40)
                seed = SEED + 1_000_000 + di * 100_000 + ui * 1_000
            result = interval(difference, "accuracy", count, seed, dataset)
            result["mean_difference"] = result.pop("mean")
            rows.append({"dataset": dataset, "estimand": name, **result})
    return pd.DataFrame(rows)


def fixed_summary(fixed, count):
    rows = []
    for di, dataset in enumerate(DATASETS):
        for ui, use in enumerate(USES):
            local = fixed[(fixed.dataset == dataset) & np.isclose(fixed.requested_use, use)]
            row = {"dataset": dataset, "requested_use": use,
                   "realized_use": float(local.use_rate.mean()),
                   "normalized_cost": float(local.normalized_cost.mean())}
            for mi, metric in enumerate(("accuracy", "balanced_accuracy")):
                result = interval(local, metric, count,
                                  SEED + di * 100_000 + ui * 1_000 + mi * 10, dataset)
                row[metric] = result.pop("mean")
                row.update({f"{metric}_{key}": value for key, value in result.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(frame, name):
    metrics = ["accuracy", "balanced_accuracy", "normalized_cost", "mean_latency_ms"]
    metrics += [field for field in (
        "primary_accuracy", "secondary_accuracy", "correction_opportunity", "harmful_replacement",
        "replacement_value", "use_rate", "matched_random_pair_value", "total_routed_gain",
        "confidence_routing_lift",
    ) if field in frame]
    participants = frame.groupby(["dataset", name, "subject"], sort=True)[metrics].mean()
    return participants.groupby(["dataset", name], sort=True).mean().reset_index()


def compare_reference(actual, name, keys):
    reference = pd.read_csv(ROOT / "reference" / name)
    checked = actual.merge(reference, on=keys, suffixes=("_new", "_saved"),
                           validate="one_to_one")
    if len(checked) != len(actual) or len(checked) != len(reference):
        raise ValueError(f"Reference coverage differs: {name}")
    if not set(reference.columns).issubset(actual.columns):
        raise ValueError(f"Reference columns are missing: {name}")
    for column in reference.columns.difference(keys):
        if not np.allclose(checked[column + "_new"], checked[column + "_saved"],
                           atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError(f"Numerical result differs: {name}/{column}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.resamples < 1000:
        parser.error("Use at least 1,000 resamples")
    verify_files()
    calibration = read("calibration.csv.gz")
    configurations = read("configurations.csv.gz")
    policies = read("policies.csv.gz")
    fixed = read("fixed_use.csv.gz")
    decisions = read("selection_decisions.csv.gz")
    if set(configurations.partition) != {"test"} or set(policies.partition) != {"test"}:
        raise ValueError("Evaluation tables must contain test rows only")
    output = ROOT / "results"
    output.mkdir(exist_ok=True)
    choices = replay_selection(calibration, configurations, decisions, policies)
    choices.to_csv(output / "replayed_selection.csv", index=False)
    configuration_summary = summarise(configurations, "configuration")
    configuration_summary.to_csv(output / "configuration_summary.csv", index=False)
    policy_summary = summarise(policies, "policy")
    policy_summary.to_csv(output / "policy_summary.csv", index=False)
    comparison = contrasts(policies, fixed, args.resamples)
    comparison.to_csv(output / "paired_contrasts.csv", index=False)
    fixed_result = fixed_summary(fixed, args.resamples)
    fixed_result.to_csv(output / "fixed_use_summary.csv", index=False)
    if args.resamples == 10_000:
        compare_reference(comparison, "paired_contrasts.csv", ["dataset", "estimand"])
        compare_reference(fixed_result, "fixed_use_summary.csv", ["dataset", "requested_use"])
        compare_reference(configuration_summary, "configuration_summary.csv", ["dataset", "configuration"])
        compare_reference(policy_summary, "policy_summary.csv", ["dataset", "policy"])
    print("Verified 190 calibration selections and 1,680 held-out participant/repeat rows.")
    print("Numerical reference checks passed." if args.resamples == 10_000 else "Custom resample count; reference interval checks skipped.")
    shown = comparison[comparison.estimand == "fixed_u20_vs_argmax"].copy()
    shown["difference_pp"] = 100 * shown.mean_difference
    print(shown[["dataset", "difference_pp"]].to_string(index=False))
    print("Outputs: results/")


if __name__ == "__main__":
    main()
