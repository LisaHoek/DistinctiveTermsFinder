"""
UI helper functions for interactive group definition in the Streamlit app.

This module contains the logic for:
- selecting metadata columns that can be used for conditioning
- rendering filtering rules for Group A and Group B
- formatting group specifications for display in the app
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers.helper_statistics import is_effectively_numeric

EXCLUDED_CONDITIONING_COLS = {
# Columns excluded from the group-definition interface because they are: IDs, links / raw metadata, large text fields, or precomputed linguistic columns
    'Nr advertisement',
    'Link',
    'OCR - Delpher',
    'OCR remediated',
    'Structure',
    'Delimiter or manual split',
    'Advertisement nr/code/motto',
    'Date',
    'Age (SS)',
    'Oversampling',
    'Period',
    'Position delimiter',
    'Text before delimiter/SUPPLY SIDE',
    'Text as of delimiter/DEMAND SIDE',
    'OCR stripped',
    'OCR extended',
    'SS stripped',
    'DS stripped',
    'SS extended',
    'DS extended',
    'words OCR extended',
    'words SS extended',
    'words DS extended',
    'single nouns OCR extended',
    'phrase nouns OCR extended',
    'single and phrase nouns OCR extended',
    'single nouns SS extended',
    'phrase nouns SS extended',
    'single and phrase nouns SS extended',
    'single nouns DS extended',
    'phrase nouns DS extended',
    'single and phrase nouns DS extended',
}


def get_conditioning_columns(df):
    """
    Return metadata columns suitable for defining comparison groups.

    A column is included if:
    - it is not in the explicit exclusion list
    - it has at least one non-null value
    - it contains at least two distinct values
    """
    cols = []
    for c in df.columns:
        if c in EXCLUDED_CONDITIONING_COLS:
            continue
        non_null = df[c].dropna()
        if non_null.empty:
            continue
        if non_null.astype(str).nunique() < 2:
            continue
        cols.append(c)
    return sorted(cols)


def render_group_spec(df, title, key_prefix):
    """
    Render Streamlit sidebar widgets for defining one group specification.

    For each condition, the user selects:
    - a metadata column
    - either a numeric operator and threshold
      or one/more categorical values
    """
    st.sidebar.markdown(f"### {title}")

    metadata_cols = get_conditioning_columns(df)

    n_rules = st.sidebar.number_input(
        f"Number of conditions for {title}",
        min_value=1,
        max_value=8,
        value=1,
        step=1,
        key=f"{key_prefix}_n_rules"
    )

    spec = {}

    for i in range(n_rules):
        st.sidebar.markdown(f"**Condition {i+1}**")

        if i == 0 and "Year" in metadata_cols:
            default_col_index = metadata_cols.index("Year")
        else:
            default_col_index = 0

        col = st.sidebar.selectbox(
            f"Column {i+1}",
            metadata_cols,
            index=default_col_index,
            key=f"{key_prefix}_col_{i}"
        )

        s = df[col]

        if is_effectively_numeric(s):
            if i == 0 and col == "Year" and title == "Group A":
                default_op = "<"
                default_val = 1961
            elif i == 0 and col == "Year" and title == "Group B":
                default_op = ">="
                default_val = 1961
            else:
                default_op = "<"
                s_num = pd.to_numeric(s, errors="coerce")
                default_val = int(np.nanmedian(s_num)) if s_num.notna().any() else 0

            op_options = ["<", "<=", ">", ">=", "=="]
            op = st.sidebar.selectbox(
                f"Operator {i+1}",
                op_options,
                index=op_options.index(default_op),
                key=f"{key_prefix}_op_{i}"
            )

            value = st.sidebar.number_input(
                f"Value {i+1}",
                value=default_val,
                key=f"{key_prefix}_value_{i}"
            )

            spec[col] = (op, value)

        else:
            values = sorted([v for v in s.dropna().astype(str).unique().tolist() if v != ""])

            mode = st.sidebar.radio(
                f"Match type {i+1}",
                ["single value", "multiple values"],
                key=f"{key_prefix}_mode_{i}",
                horizontal=True
            )

            if mode == "single value":
                value = st.sidebar.selectbox(
                    f"Value {i+1}",
                    values,
                    key=f"{key_prefix}_single_{i}"
                )
                spec[col] = value
            else:
                selected = st.sidebar.multiselect(
                    f"Values {i+1}",
                    values,
                    default=values[:1] if values else [],
                    key=f"{key_prefix}_multi_{i}"
                )
                spec[col] = selected

    return spec


def render_groups(df):
    """
    Render the sidebar controls for both Group A and Group B.
    """
    group_a = render_group_spec(df, "Group A", "group_a")
    group_b = render_group_spec(df, "Group B", "group_b")
    return group_a, group_b


def format_group_spec(spec):
    """
    Format a group specification as a readable multiline string.

    This is used in the main app to display the current definitions
    of Group A and Group B.
    """
    lines = []
    for col, rule in spec.items():
        if isinstance(rule, tuple) and len(rule) == 2:
            lines.append(f"{col} {rule[0]} {rule[1]}")
        elif isinstance(rule, list):
            lines.append(f"{col} in {rule}")
        else:
            lines.append(f"{col} = {rule}")
    return "\n".join(lines)
