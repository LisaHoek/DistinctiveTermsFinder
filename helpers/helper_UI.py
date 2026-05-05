"""
UI helper functions for interactive group definition in the Streamlit app.

This module contains the logic for:
- selecting metadata columns that can be used for conditioning
- rendering filtering rules for Group A and Group B
- optionally defining groups via uploaded subset CSVs containing `Nr advertisement`
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


def render_group_spec(df, title, key_prefix, show_title=True):
    """
    Render Streamlit sidebar widgets for defining one group specification.

    For each condition, the user selects:
    - a metadata column
    - either a numeric operator and threshold
      or one/more categorical values
    """
    if show_title:
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


def load_uploaded_group_ids(uploaded_file, full_df, ad_id_col="Nr advertisement"):
    """
    Read uploaded CSV and extract valid advertisement IDs.

    Returns
    -------
    ids : list
        Unique IDs from the uploaded file that also occur in full_df.
    info : str
        Informational message for the UI.
    error : str or None
        Error message if the file is invalid.
    """
    if uploaded_file is None:
        return [], "", None

    subset_df = pd.read_csv(uploaded_file)

    if ad_id_col not in subset_df.columns:
        return [], "", f"Uploaded file must contain column '{ad_id_col}'."

    ids = subset_df[ad_id_col].dropna().tolist()
    ids = list(dict.fromkeys(ids))  # unique, preserve order

    valid_ids = set(full_df[ad_id_col].dropna().tolist())
    matched_ids = [x for x in ids if x in valid_ids]
    missing_count = len(ids) - len(matched_ids)

    info = f"Loaded {len(matched_ids)} advertisement IDs"
    if missing_count:
        info += f" ({missing_count} not found in the currently uploaded main dataframe)"

    return matched_ids, info, None


def render_group_input(df, title, key_prefix, ad_id_col="Nr advertisement", allow_remainder=False):
    """
    Render UI for one group, allowing either:
    - sidebar conditions
    - uploaded subset CSV with `Nr advertisement`
    - optionally: all remaining advertisements
    """
    st.sidebar.markdown(f"### {title}")

    source_options = ["Conditions", "Upload CSV"]
    if allow_remainder:
        source_options.append("All remaining ads")

    source_mode = st.sidebar.radio(
        f"{title} source",
        source_options,
        key=f"{key_prefix}_source_mode",
        horizontal=True
    )

    if source_mode == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            f"Upload subset CSV for {title}",
            type=["csv"],
            key=f"{key_prefix}_uploaded_file",
            help=f"The uploaded CSV must contain the column '{ad_id_col}'."
        )

        ids, info, error = load_uploaded_group_ids(uploaded_file, df, ad_id_col=ad_id_col)

        if error:
            st.sidebar.error(error)
        elif uploaded_file is not None:
            st.sidebar.caption(info)

        return {
            "mode": "upload",
            "ids": ids,
            "file_name": uploaded_file.name if uploaded_file is not None else None,
        }

    if source_mode == "All remaining ads":
        st.sidebar.caption(f"{title} will contain all advertisements not assigned to the other group.")
        return {
            "mode": "remainder"
        }

    spec = render_group_spec(df, title, key_prefix, show_title=False)

    return {
        "mode": "conditions",
        "spec": spec,
    }


def render_groups(df, ad_id_col="Nr advertisement"):
    """
    Render the sidebar controls for both Group A and Group B.

    Group A:
    - conditions
    - uploaded subset

    Group B:
    - conditions
    - uploaded subset
    - all remaining advertisements
    """
    group_a = render_group_input(
        df, "Group A", "group_a", ad_id_col=ad_id_col, allow_remainder=False
    )
    group_b = render_group_input(
        df, "Group B", "group_b", ad_id_col=ad_id_col, allow_remainder=True
    )
    return group_a, group_b


def format_group_spec(spec):
    """
    Format a group specification as a readable multiline string.
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


def format_group_definition(group_def):
    """
    Format either a condition-based, upload-based, or remainder-based group definition.
    """
    if group_def["mode"] == "upload":
        if group_def.get("file_name"):
            return (
                f"Source: uploaded subset\n"
                f"File: {group_def['file_name']}\n"
                f"IDs loaded: {len(group_def.get('ids', []))}"
            )
        return "Source: uploaded subset\nNo file uploaded yet."

    if group_def["mode"] == "remainder":
        return "Source: all remaining advertisements"

    return "Source: sidebar conditions\n" + format_group_spec(group_def.get("spec", {}))