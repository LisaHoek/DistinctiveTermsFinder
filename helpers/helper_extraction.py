"""
Helper functions for turning precomputed list columns into occurrence tables.

This module is used by the Streamlit app to convert list-valued columns such as:

    - words OCR extended
    - single nouns SS extended
    - phrase nouns DS extended

into a long-format occurrence table with the following columns:

    - ad_id
    - term
    - count

The app assumes that the source dataframe already contains precomputed list columns. No linguistic extraction is performed here; this module only parses and reshapes those existing columns.
"""

import streamlit as st
import pandas as pd
import ast

def parse_list_cell(x):
    """
    Convert a dataframe cell into a Python list.
    This function supports two common storage formats:

    1. A real Python list
       Example:
           ["term1", "term2"]

    2. A stringified list, often found in CSV exports
       Example:
           "['term1', 'term2']"
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
    """
    Build a term-occurrence table from a list-valued dataframe column.

    The procedure:
    1. Keeps only the advertisement ID column and the selected list column.
    2. Parses each cell into a proper Python list.
    3. Explodes the lists into one row per term occurrence.
    4. Normalizes terms to lowercase and strips surrounding whitespace.
    5. Counts term frequency per advertisement.

    Parameters
    ----------
    df : pandas.DataFrame
        Input advertisement dataframe.
    list_col : str
        Name of the list-valued column to process.
    ad_id_col : str, optional
        Name of the advertisement ID column, by default "Nr advertisement".

    Returns
    -------
    pandas.DataFrame
        Long-format occurrence table with columns:
        - ad_id
        - term
        - count
    """
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
    Build an occurrence table from a precomputed list column.

    The app constructs the expected column name from:

        <unit_type> <text_col>

    For example:
        "words OCR extended"
        "single nouns SS extended"

    This function is cached with Streamlit so that repeated UI interactions do not unnecessarily rebuild the same occurrence table.

    Parameters
    ----------
    ads_df : pandas.DataFrame
        Full advertisement dataframe.
    text_col : str
        Base text column, e.g. "OCR extended", "SS extended", or "DS extended".
    unit_type : str
        Type of linguistic unit, e.g. "words" or "phrase nouns".
    ad_id_col : str, optional
        Advertisement ID column, by default "Nr advertisement".

    Returns
    -------
    tuple
        A tuple of:
        - occ_df : pandas.DataFrame
            Long-format occurrence table.
        - source_info : str
            Human-readable description of the source column.

    Raises
    ------
    ValueError
        If the expected precomputed list column is not present.
    """
    list_col = f"{unit_type} {text_col}"

    if list_col not in ads_df.columns:
        raise ValueError(f"Missing precomputed column: {list_col}")

    occ_df = build_occ_from_list_column(ads_df, list_col, ad_id_col=ad_id_col)
    source_info = f"precomputed ({list_col})"

    return occ_df, source_info
