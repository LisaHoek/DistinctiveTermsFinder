"""
Streamlit app for comparing language use across two advertisement groups.

This application allows the user to:

1. Upload a CSV dataframe of Dutch dating advertisements.
2. Select a text scope:
   - Whole text
   - Supply-side text
   - Demand-side text
3. Select a linguistic unit:
   - words
   - single nouns
   - phrase nouns
   - single and phrase nouns
4. Define two comparison groups using metadata filters.
5. Compute weighted log-odds scores to identify terms that are
   relatively distinctive for one group versus the other.

The app assumes that the uploaded dataframe already contains precomputed list-valued columns such as:
    - words OCR extended
    - single nouns SS extended
    - phrase nouns DS extended

These columns are converted into occurrence tables on demand.
"""

import streamlit as st
import pandas as pd

from helpers.helper_UI import render_groups, format_group_definition
from helpers.helper_statistics import compare_groups
from helpers.helper_extraction import build_occurrence_table

# ---------------------------------------------------------------------------
# Dynamic availability
# ---------------------------------------------------------------------------

SCOPE_LABEL_TO_COL = {
# Mapping from user-facing labels to the underlying dataframe columns.
    "Whole text": "OCR extended",
    "SS text": "SS extended",
    "DS text": "DS extended",
}

SUPPORTED_UNITS = [
# Supported types of precomputed language units.    
    "words", "single nouns", "phrase nouns", "single and phrase nouns", "single adjectives"
]


def get_available_scopes(df):
    """
    Return only those text scopes whose base text column is present.
    """
    return {
        label: col
        for label, col in SCOPE_LABEL_TO_COL.items()
        if col in df.columns
    }


def get_available_units(df, text_col):
    """
    For a given scope, only show units whose precomputed column exists.
    """
    units = []
    for unit in SUPPORTED_UNITS:
        colname = f"{unit} {text_col}"
        if colname in df.columns:
            units.append(unit)
    return units

# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Language comparison", layout="wide")
st.title("Language comparison for Dutch dating advertisements")

st.markdown(
    """
    This app compares linguistic patterns across two user-defined groups of advertisements.
    It uses **weighted log-odds with z-scores** to identify terms that are relatively distinctive for one group versus another.

    Upload a CSV file containing:
    - a unique advertisement ID column: `Nr advertisement`
    - at least one base text column such as `OCR extended`, `SS extended`, or `DS extended`
    - one or more precomputed list columns such as `single and phrase nouns OCR extended`
    """
)

uploaded_file = st.file_uploader("Upload dataframe (.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Upload your dataframe to start.")
    st.stop()


# ---------------------------------------------------------------------------
# Load and validate input data
# ---------------------------------------------------------------------------

ads_df = pd.read_csv(uploaded_file)

if "Nr advertisement" not in ads_df.columns:
    st.error("Missing required column: 'Nr advertisement'")
    st.stop()

# Convert Year to numeric if present, because it is commonly used for filtering.
if "Year" in ads_df.columns:
    ads_df["Year"] = pd.to_numeric(ads_df["Year"], errors="coerce")

available_scopes = get_available_scopes(ads_df)

if not available_scopes:
    st.error("No available text scopes found. Expected at least one of: 'OCR extended', 'SS extended', 'DS extended'.")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

st.sidebar.header("Analysis settings")

text_scope_label = st.sidebar.selectbox(
    "Text scope",
    list(available_scopes.keys())
)
text_col = available_scopes[text_scope_label]

available_units = get_available_units(ads_df, text_col)

if not available_units:
    st.error(f"No precomputed unit columns found for scope '{text_col}'.")
    st.stop()

unit_type = st.sidebar.selectbox(
    "Language unit",
    available_units
)

min_count = st.sidebar.slider(
    "Minimum total frequency", 
    min_value=1, 
    max_value=50, 
    value=5, 
    step=1)

remove_insignificant = st.sidebar.checkbox(
    "Remove insignificant z-scores",
    value=False,
    help="Keep only terms with |z| >= threshold"
)

z_threshold = st.sidebar.number_input(
    "Significance threshold |z|",
    min_value=0.0,
    value=1.96,
    step=0.1,
    help="Common default for approximate two-sided significance"
)

term_filters = st.sidebar.multiselect(
    "Filter terms containing",
    options=[],
    default=[],
    accept_new_options=True,
    help="Type a term and press Enter. Each term is matched as a substring. Example: huwelijk"
)

# Render the sidebar controls for defining Group A and Group B.
group_a, group_b = render_groups(ads_df)

group_a_spec = group_a.get("spec") if group_a["mode"] == "conditions" else None
group_b_spec = group_b.get("spec") if group_b["mode"] == "conditions" else None

ids_a_input = group_a.get("ids") if group_a["mode"] == "upload" else None
ids_b_input = group_b.get("ids") if group_b["mode"] == "upload" else None

remainder_b = group_b["mode"] == "remainder"

if group_a["mode"] == "upload" and not ids_a_input:
    st.warning("Group A is set to 'Upload CSV', but no valid Group A file has been uploaded.")
    st.stop()

if group_b["mode"] == "upload" and not ids_b_input:
    st.warning("Group B is set to 'Upload CSV', but no valid Group B file has been uploaded.")
    st.stop()

# ---------------------------------------------------------------------------
# Build occurrence table and run comparison
# ---------------------------------------------------------------------------

with st.spinner(f"Building occurrence table from precomputed column for {unit_type} / {text_col}..."):
    occ_df, source_info = build_occurrence_table(
        ads_df=ads_df,
        text_col=text_col,
        unit_type=unit_type,
        ad_id_col="Nr advertisement"
    )

result, ids_a, ids_b = compare_groups(
    ads_df=ads_df,
    occ_df=occ_df,
    group_a=group_a_spec,
    group_b=group_b_spec,
    ids_a=ids_a_input,
    ids_b=ids_b_input,
    ad_id_col="Nr advertisement",
    min_count=min_count,
    remainder_b=remainder_b
)

if ids_a_input is not None and ids_b_input is not None:
    overlap = set(ids_a_input) & set(ids_b_input)
    if overlap:
        st.warning(f"{len(overlap)} advertisements occur in both uploaded groups.")

if result.empty:
    st.warning("No terms available for this comparison after filtering/min_count.")
    st.stop()


# ---------------------------------------------------------------------------
# Optional filtering of the result table
# ---------------------------------------------------------------------------

term_filters = [t.strip() for t in term_filters if t.strip()]

if term_filters:
    term_series = result["term"].astype(str)
    mask = pd.Series(False, index=result.index)

    for filt in term_filters:
        mask |= term_series.str.contains(filt, case=False, na=False, regex=False)

    result = result[mask]

if result.empty:
    st.warning("No terms match the term filter.")
    st.stop()

if remove_insignificant:
    result = result[result["z"].abs() >= z_threshold]

if result.empty:
    st.warning("No terms remain after z-score significance filtering.")
    st.stop()

# Split results by sign of z-score:
# - z > 0 : distinctive for Group A
# - z < 0 : distinctive for Group B
res_a = result[result["z"] > 0].sort_values("z", ascending=False)
res_b = result[result["z"] < 0].sort_values("z", ascending=True)


# ---------------------------------------------------------------------------
# Output: summary and group definitions
# ---------------------------------------------------------------------------

st.subheader("Comparison summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Text scope", text_scope_label)
c2.metric("Unit type", unit_type)
c3.metric("Ads in Group A", len(ids_a))
c4.metric("Ads in Group B", len(ids_b))
c5.metric("Z filter", f"|z| ≥ {z_threshold}" if remove_insignificant else "off")
c6.metric("Term filter", "on" if term_filters else "off")

st.subheader("Group definitions")

g1, g2 = st.columns(2)

with g1:
    st.markdown("**Group A**")
    st.code(format_group_definition(group_a))

with g2:
    st.markdown("**Group B**")
    st.code(format_group_definition(group_b))


# ---------------------------------------------------------------------------
# Output: distinctive term tables
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Distinctive for Group A (z > 0)")
    st.dataframe(
        res_a[["term", "count_a", "count_b", "z", "log_odds", "total"]],
        width='stretch'
    )

with col2:
    st.markdown("### Distinctive for Group B (z < 0)")
    st.dataframe(
        res_b[["term", "count_a", "count_b", "z", "log_odds", "total"]],
        width='stretch'
    )

st.markdown("### Full result table")
st.dataframe(result, width='stretch')


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

csv = result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download full result as CSV",
    data=csv,
    file_name=f"log_odds_{unit_type}_{text_col.replace(' ', '_')}.csv",
    mime="text/csv"
)