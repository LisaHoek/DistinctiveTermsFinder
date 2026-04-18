import streamlit as st
import pandas as pd
import ast

# =========================
# Build occurrence table
# =========================

def parse_list_cell(x):
    """
    Makes sure a dataframe cell becomes a Python list.
    Works for:
    - real lists (Parquet)
    - stringified lists like "['a', 'b']" (CSV)
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        if x.startswith("[") and x.endswith("]"):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
        return []
    return []


def build_occ_from_list_column(df, list_col, ad_id_col="Nr advertisement"):
    tmp = df[[ad_id_col, list_col]].copy()
    tmp[list_col] = tmp[list_col].apply(parse_list_cell)

    tmp = (
        tmp.explode(list_col)
        .dropna(subset=[list_col])
        .rename(columns={ad_id_col: "ad_id", list_col: "term"})
    )

    tmp["term"] = tmp["term"].astype(str).str.strip().str.lower()
    tmp = tmp[tmp["term"] != ""]

    occ_df = (
        tmp.groupby(["ad_id", "term"])
        .size()
        .reset_index(name="count")
    )

    return occ_df


@st.cache_data(show_spinner=True)
def build_occurrence_table(ads_df, text_col, unit_type, ad_id_col="Nr advertisement"):
    """
    Uses only precomputed list columns already present in ads_df.
    """
    list_col = f"{unit_type} {text_col}"

    if list_col not in ads_df.columns:
        raise ValueError(f"Missing precomputed column: {list_col}")

    occ_df = build_occ_from_list_column(ads_df, list_col, ad_id_col=ad_id_col)
    source_info = f"precomputed ({list_col})"

    return occ_df, source_info
