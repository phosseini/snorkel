from collections import defaultdict

import numpy as np
import pandas as pd

from snorkel.analysis.metrics import metric_score


def metrics_dict_to_dataframe(metrics_dict):
    # list of tuples to be converted to pd.DataFrame
    metrics = []

    for full_metric, score in metrics_dict.items():
        label_name, dataset_name, split, metric = tuple(full_metric.split("/"))
        metrics.append((label_name, dataset_name, split, metric, score))

    return pd.DataFrame(
        metrics, columns=["label", "dataset", "split", "metric", "score"]
    )


def slice_df(S_matrix, slice_names, target_slice, df):
    """Returns a dataframe subsets with examples belonging to specified target_slice."""

    assert len(df) == len(S_matrix)

    slice_idx = slice_names.index(target_slice)
    mask = S_matrix[:, slice_idx]
    df_idx = np.where(mask)[0]
    return df.iloc[df_idx]


def score_on_slices(
    S_matrix,
    slice_names,
    *,
    golds,
    preds,
    probs=None,
    metrics=["accuracy"],
    as_dataframe=False,
):
    """Return dictionary of slice scores."""

    slice_scores = defaultdict(dict)
    for slice_idx, slice_name in enumerate(slice_names):
        mask = S_matrix[:, slice_idx].astype(bool)
        for metric in metrics:
            masked_golds = golds[mask] if golds is not None else None
            masked_preds = preds[mask] if preds is not None else None
            masked_probs = probs[mask] if probs is not None else None
            slice_scores[slice_name][metric] = metric_score(
                masked_golds, masked_preds, masked_probs, metric
            )

    slice_scores = dict(slice_scores)
    if as_dataframe:
        return pd.DataFrame.from_dict(slice_scores).transpose()
    else:
        return slice_scores
