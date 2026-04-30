"""
Statistical helper functions for comparing term usage across advertisement groups.

This module provides the backend for group comparison in the Streamlit app.
It supports:

- building boolean masks from user-defined filtering rules
- aggregating term counts for selected advertisement IDs
- computing weighted log-odds with informative Dirichlet prior
- returning terms that are distinctive for one group versus another
"""

import pandas as pd
import numpy as np
import operator

# Mapping from string operators used in the UI to Python comparison functions.
OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def is_effectively_numeric(series, threshold=0.9):
    """
    Determine whether a pandas Series behaves like numeric data.

    A column is treated as effectively numeric when at least `threshold` proportion of its values can be converted to numbers.

    This heuristic is useful in the UI when deciding whether a column
    should be presented with numeric comparison operators or categorical value selection.
    """
    s_num = pd.to_numeric(series, errors="coerce")
    return s_num.notna().mean() >= threshold


def build_mask(df, spec=None):
    """
    Build a boolean mask from a dictionary of filtering rules.

    Examples
    --------
    {"Year": ("<", 1960)}
    {"Goal of advertisement": "Marriage"}
    {"Area number": ["1", "2"]}
    {"Sex (SS)": "Female", "Year": (">=", 1960)}

    Rule types
    ----------
    1. Numeric comparison:
       {"Year": ("<", 1960)}

    2. Single categorical value:
       {"Goal of advertisement": "Marriage"}

    3. Multiple allowed values:
       {"Area number": ["1", "2"]}
    """
    if spec is None or spec == "ALL" or spec == {}:
        return pd.Series(True, index=df.index)

    mask = pd.Series(True, index=df.index)

    for col, rule in spec.items():
        s = df[col]

        if isinstance(rule, tuple) and len(rule) == 2 and rule[0] in OPS:
            op, value = rule
            s_num = pd.to_numeric(s, errors="coerce")
            m = OPS[op](s_num, value)

        elif isinstance(rule, (list, set, tuple)):
            m = s.astype(str).isin([str(v) for v in rule])

        else:
            m = s.astype(str).eq(str(rule))

        mask &= m.fillna(False)

    return mask


def get_term_counts(occ_df, ad_ids):
    """
    Aggregate term counts for a selected set of advertisement IDs.

    Parameters
    ----------
    occ_df : pandas.DataFrame
        Occurrence table with columns:
        - ad_id
        - term
        - count
    ad_ids : iterable
        Advertisement IDs to include.

    Returns
    -------
    pandas.Series
        Series indexed by term with summed counts as values.
    """
    sub = occ_df[occ_df["ad_id"].isin(ad_ids)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("term")["count"].sum()


def weighted_log_odds(counts_a, counts_b, prior=None, min_count=5, prior_strength=0.01):
    """
    Compute weighted log-odds scores with an informative Dirichlet prior.

    This method is commonly used to identify terms that are relatively
    distinctive for one corpus or subgroup compared with another.

    Positive z-scores indicate terms more associated with Group A.
    Negative z-scores indicate terms more associated with Group B.

    Parameters
    ----------
    counts_a : pandas.Series
        Term counts for Group A.
    counts_b : pandas.Series
        Term counts for Group B.
    prior : pandas.Series or None, optional
        Prior term counts over the full corpus. If None, counts_a + counts_b
        is used as the prior.
    min_count : int, optional
        Minimum total term frequency across both groups required for retention.
    prior_strength : float, optional
        Small smoothing constant added to the prior.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - term
        - count_a
        - count_b
        - log_odds
        - z
        - total
    """
    vocab = counts_a.index.union(counts_b.index)

    a = counts_a.reindex(vocab, fill_value=0).astype(float)
    b = counts_b.reindex(vocab, fill_value=0).astype(float)

    if prior is None:
        prior = a + b
    else:
        prior = prior.reindex(vocab, fill_value=0).astype(float)

    prior = prior + prior_strength
    alpha0 = prior.sum()

    n_a = a.sum()
    n_b = b.sum()

    if n_a == 0 or n_b == 0:
        return pd.DataFrame(columns=["term", "count_a", "count_b", "log_odds", "z", "total"])

    delta = np.log((a + prior) / (n_a + alpha0 - a - prior)) - \
            np.log((b + prior) / (n_b + alpha0 - b - prior))

    var = 1.0 / (a + prior) + 1.0 / (b + prior)
    z = delta / np.sqrt(var)

    out = pd.DataFrame({
        "term": vocab,
        "count_a": a.values,
        "count_b": b.values,
        "log_odds": delta.values,
        "z": z.values
    })

    out["total"] = out["count_a"] + out["count_b"]
    out = out[out["total"] >= min_count].copy()
    out = out.sort_values("z", ascending=False)

    return out


def compare_groups(ads_df, occ_df, group_a, group_b, ad_id_col="Nr advertisement", min_count=5):
    """
    Compare two advertisement groups on term usage.

    The procedure:
    1. Build masks for Group A and Group B from their rule specifications.
    2. Collect advertisement IDs for both groups.
    3. Remove overlap from Group B so that advertisements already assigned
       to Group A are not counted twice.
    4. Aggregate term counts for both groups.
    5. Compute weighted log-odds with z-scores.

    Parameters
    ----------
    ads_df : pandas.DataFrame
        Main advertisement dataframe.
    occ_df : pandas.DataFrame
        Occurrence table produced from a precomputed list column.
    group_a : dict
        Rule specification for Group A.
    group_b : dict
        Rule specification for Group B.
    ad_id_col : str, optional
        Advertisement ID column, by default "Nr advertisement".
    min_count : int, optional
        Minimum total term frequency required to retain a term.

    Returns
    -------
    tuple
        A tuple of:
        - result : pandas.DataFrame
            Weighted log-odds result table.
        - ids_a : set
            Advertisement IDs assigned to Group A.
        - ids_b : set
            Advertisement IDs assigned to Group B (after overlap removal).
    """
    mask_a = build_mask(ads_df, group_a)
    mask_b = build_mask(ads_df, group_b)

    ids_a = set(ads_df.loc[mask_a, ad_id_col])
    ids_b = set(ads_df.loc[mask_b, ad_id_col])

    # remove overlap
    ids_b = ids_b - ids_a

    counts_a = get_term_counts(occ_df, ids_a)
    counts_b = get_term_counts(occ_df, ids_b)

    prior = occ_df.groupby("term")["count"].sum() if not occ_df.empty else pd.Series(dtype=float)

    result = weighted_log_odds(
        counts_a=counts_a,
        counts_b=counts_b,
        prior=prior,
        min_count=min_count
    )

    return result, ids_a, ids_b