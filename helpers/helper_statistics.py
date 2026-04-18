import pandas as pd
import numpy as np
import operator

# =========================
# Group filtering backend
# =========================

OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def is_effectively_numeric(series, threshold=0.9):
    s_num = pd.to_numeric(series, errors="coerce")
    return s_num.notna().mean() >= threshold


def build_mask(df, spec=None):
    """
    spec examples:
    {"Year": ("<", 1960)}
    {"Goal of advertisement": "Marriage"}
    {"Area number": ["1", "2"]}
    {"Sex (SS)": "Female", "Year": (">=", 1960)}
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
    sub = occ_df[occ_df["ad_id"].isin(ad_ids)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("term")["count"].sum()


def weighted_log_odds(counts_a, counts_b, prior=None, min_count=5, prior_strength=0.01):
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


def compare_groups(
    ads_df,
    occ_df,
    group_a,
    group_b,
    ad_id_col="Nr advertisement",
    min_count=5
):
    mask_a = build_mask(ads_df, group_a)
    mask_b = build_mask(ads_df, group_b)

    ids_a = set(ads_df.loc[mask_a, ad_id_col])
    ids_b = set(ads_df.loc[mask_b, ad_id_col])

    # overlap verwijderen
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