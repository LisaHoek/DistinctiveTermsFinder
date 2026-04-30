# Find Distinctive Terms

A Streamlit app for comparing language in Dutch historical dating advertisements and identifying terms that are statistically distinctive for one group versus another.

## What this app does

This app is designed for corpus comparison. It lets you:

- upload a dataframe of advertisements with metadata
- define **Group A** and **Group B** using metadata filters
- choose a text scope such as:
  - whole text
  - supply-side text
  - demand-side text
- choose a precomputed language unit such as:
  - words
  - single nouns
  - phrase nouns
  - single and phrase nouns
- calculate **weighted log-odds** and **z-scores**
- inspect which terms are most distinctive for Group A versus Group B

Typical use cases include comparisons such as:

- before 1960 vs. after 1960
- marriage vs. alternative relationships
- men vs. women
- religious vs. non-religious
- certain age groups

The app is especially useful for large historical text datasets where you want to detect meaningful lexical differences between subcorpora.

---

## Public app

You can use the public app here:

**https://finddistinctiveterms.streamlit.app**

Upload the csv datasets you can find in this repository _(not yet publicly available)_.

If the public app is temporarily unavailable, slow, or has too many users, you can run your own copy locally or deploy your own version online. 

---

## How to run the app locally

### 1. Clone this repository

    git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
    cd YOUR-REPO-NAME

### 2. Create and activate a virtual environment

Mac / Linux:

    python -m venv .venv
    source .venv/bin/activate

Windows:

    python -m venv .venv
    .venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt


### 4. Start the app

    streamlit run appDistinctiveTerms.py

Your browser should open automatically. If not, Streamlit will print a local URL such as:

    http://localhost:8501

## How to make your own version

You are very welcome to copy this repository and adapt it for your own project.

### Option 1: Fork this repository

If you want to keep a visible link to the original repository, click **Fork** on GitHub.

### Option 2: Copy the code into your own repository

If you want a fully independent version, create a new repository and copy the files there.

### Typical reasons to make your own version

You want to:

- adjust the interface
- add more filter types or metadata options
- add plots or UMAP visualizations
- compare different kinds of language units
- support other corpora beyond advertisements

Or simply because:

- the public app has too many users or is temporarily unavailable
- you want your own stable deployment

## How to deploy your own Streamlit app

The easiest way is with **Streamlit Community Cloud**.

### 1. Push your code to GitHub

Make sure your repository includes at least:

- `app.py`
- `requirements.txt`

### 2. Go to Streamlit Community Cloud

Open: **https://share.streamlit.io**

### 3. Sign in with GitHub

### 4. Create a new app

Select:

- your repository
- the branch
- the main file, usually `app.py`

### 5. Deploy

After deployment, Streamlit will give you your own app URL.

## Expected data format

The app assumes that your dataset already contains pre-computed list columns following the pattern: [language unit] [column name]. This is because computing linguistic units is slow using NLP pipelines. Having the terms extracted beforehand makes it faster, lighter to deploy and easier to reuse. (ToDo: add reference how to compute linguistic units).

Current supported language units:

  - `words`
  - `single nouns`
  - `phrase nouns`
  - `single and phrase nouns`

With at least one of the following column names:

- `OCR extended`
- `SS extended`
- `DS extended`

It also expects a unique advertisement identifier column:

- `Nr advertisement`

## How the comparison works

The app compares two user-defined groups of advertisements and calculates distinctive terms using **weighted log-odds**.

In general:

- **positive z-scores** indicate terms more associated with **Group A**
- **negative z-scores** indicate terms more associated with **Group B**
- terms near zero are less distinctive
- optional filtering can remove low-frequency terms or statistically weak differences

## Citation

If you use this app or adapt the code, please cite the repository and, once available, the accompanying paper.

    <Placeholder> Author Name. Title of repository/app. GitHub repository, year. URL

Future paper citation:
To be added here.

## License
This project is licensed under \<placeholder\>.

## Contact
If you have any questions, please contact me using the contact details available on my personal webpage: **https://hookedondata.nl**.