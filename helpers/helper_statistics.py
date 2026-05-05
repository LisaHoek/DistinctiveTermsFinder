"""
Statistical helper functions for comparing term usage across advertisement groups.

This module provides the backend for group comparison in the Streamlit app.
It supports:

- building boolean masks from user-defined filtering rules
- aggregating term counts for selected advertisement IDs
- computing weighted log-odds with informative Dirichlet prior
- returning terms that are distinctive for one group versus another
- resolving groups either from sidebar conditions or from uploaded ID lists
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

    A column is treated as effectively numeric when at least `threshold` proportion of its 
    values can be converted to numbers.

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


def resolve_group_ids(ads_df, group_spec=None, ids=None, ad_id_col="Nr advertisement"):
    """
    Resolve a group to a set of advertisement IDs.

    A group can be defined in two ways:

    1. By explicit IDs, e.g. from an uploaded subset CSV
    2. By a sidebar condition specification (`group_spec`)

    If `ids` is provided, it takes precedence over `group_spec`.

    Parameters
    ----------
    ads_df : pandas.DataFrame
        Main advertisement dataframe.
    group_spec : dict or None
        Rule specification for filtering ads_df.
    ids : iterable or None
        Explicit advertisement IDs to use for the group.
    ad_id_col : str, optional
        Name of the advertisement ID column.

    Returns
    -------
    set
        Set of valid advertisement IDs found in the main dataframe.
    """
    valid_ids = set(ads_df[ad_id_col].dropna())

    if ids is not None:
        return set(ids) & valid_ids

    mask = build_mask(ads_df, group_spec)
    return set(ads_df.loc[mask, ad_id_col].dropna())


def compare_groups(
    ads_df,
    occ_df,
    group_a=None,
    group_b=None,
    ids_a=None,
    ids_b=None,
    ad_id_col="Nr advertisement",
    min_count=5,
    remainder_b=False
):
    """
    Compare two advertisement groups on term usage.

    Each group can be defined either:
    - by a rule specification (`group_a`, `group_b`)
    - or by explicit advertisement IDs (`ids_a`, `ids_b`), e.g. from uploaded subset CSVs

    Additionally, Group B can optionally be defined as all advertisements not assigned to 
    Group A.

    If explicit IDs are provided for a group, they take precedence over the condition 
    specification for that group.

    Parameters
    ----------
    ads_df : pandas.DataFrame
        Main advertisement dataframe.
    occ_df : pandas.DataFrame
        Occurrence table produced from a precomputed list column.
    group_a : dict or None
        Rule specification for Group A.
    group_b : dict or None
        Rule specification for Group B.
    ids_a : iterable or None
        Explicit advertisement IDs for Group A.
    ids_b : iterable or None
        Explicit advertisement IDs for Group B.
    ad_id_col : str, optional
        Advertisement ID column, by default "Nr advertisement".
    min_count : int, optional
        Minimum total term frequency required to retain a term.
    remainder_b : bool, optional
        If True, Group B is defined as all valid advertisements not in Group A.

    Returns
    -------
    tuple
        A tuple of:
        - result : pandas.DataFrame
            Weighted log-odds result table.
        - ids_a : set
            Advertisement IDs assigned to Group A.
        - ids_b : set
            Advertisement IDs assigned to Group B.
    """
    all_ids = set(ads_df[ad_id_col].dropna())

    ids_a = resolve_group_ids(
        ads_df=ads_df,
        group_spec=group_a,
        ids=ids_a,
        ad_id_col=ad_id_col
    )

    if remainder_b:
        ids_b = all_ids - ids_a
    else:
        ids_b = resolve_group_ids(
            ads_df=ads_df,
            group_spec=group_b,
            ids=ids_b,
            ad_id_col=ad_id_col
        )
        # Remove overlap so ads in Group A are not also counted in Group B.
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