# 📧 Automated Email Body Text Classifier

> **An NLP pipeline that cleans incoming email text and automatically classifies each message to the correct business unit — turning a manual triage/routing task into a single model prediction.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Model](https://img.shields.io/badge/Best%20Model-LinearSVC-brightgreen)]()
[![Vectorizer](https://img.shields.io/badge/Features-TF--IDF%20(1%2C2)-blueviolet)]()

---

## 📋 Overview

Organisations receiving a high volume of email queries often need to route each message to the right internal team. This project automates that step. Given the **body text of an email**, the model predicts which **business unit** the message belongs to, so incoming queries can be triaged automatically instead of by hand.

The solution is delivered as **two Jupyter notebooks** that form a sequential pipeline:

| Stage | Notebook | Purpose |
|---|---|---|
| **1. Data Prep & EDA** | `FQ_Data_Analysis.ipynb` | Clean and normalise raw email text, then explore the corpus (top words, n-grams, parts of speech). Exports a cleaned dataset. |
| **2. Classification** | `FQ_Text_Class.ipynb` | Vectorise the cleaned text with TF-IDF, compare several classifiers, and train a final model to predict the target unit. |

---

## 🔄 Pipeline

```
   Raw emails              Notebook 1: FQ_Data_Analysis            Notebook 2: FQ_Text_Class
 ┌──────────────┐        ┌───────────────────────────────┐      ┌──────────────────────────────┐
 │ FQ_Data.xlsx │───────▶│ • drop nulls / multi-tag rows │      │ • TF-IDF (unigrams+bigrams)  │
 │ (Summary +   │        │ • lowercase, strip greetings  │      │ • chi² top terms per class   │
 │  unit labels)│        │   & sign-offs, clean HTML     │─────▶│ • compare 4 models (5-fold CV)│
 └──────────────┘        │ • expand contractions         │ test │ • train final LinearSVC      │
                         │ • remove digits/punct/stops   │ .xlsx│ • evaluate + predict new mail│
                         │ • spaCy lemmatisation         │      └──────────────────────────────┘
                         │ • EDA: words / n-grams / POS  │                     │
                         └───────────────────────────────┘                     ▼
                                                                      Predicted business unit
```

---

## 📑 Table of Contents

- [Notebook 1 — Data Analysis & Preprocessing](#-notebook-1--data-analysis--preprocessing)
- [Notebook 2 — Text Classification](#-notebook-2--text-classification)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage](#-usage)
- [Model Selection & Results](#-model-selection--results)
- [Data Privacy Note](#-data-privacy-note)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🧹 Notebook 1 — Data Analysis & Preprocessing
**`FQ_Data_Analysis.ipynb`**

Reads the raw email export and prepares a clean, model-ready text column. Key steps:

**Loading & filtering**
- Loads emails from an Excel workbook (`Summary` = email body, plus categorical label columns).
- Drops null rows and counts unique category tags.
- Removes ambiguous emails tagged with **multiple** business areas/units (identified by a `#` delimiter), keeping only single-label messages.

**Text cleaning** (applied to the `Summary` column)
- Lowercasing
- Stripping the **greeting** (text before the first newline) and the **sign-off** (everything after `regards` / `kind regards` / `many thanks` / `thanks`)
- Removing **HTML** with BeautifulSoup
- **Expanding contractions** via a contractions dictionary (e.g. *can't → cannot*)
- Removing digits and words containing digits
- Removing punctuation, bullet points, and extra whitespace
- **Lemmatisation + stop-word removal** with spaCy (`en_core_web_sm`)
- Removing residual non-alphabetic characters and leftover greeting/sign-off words
- Dropping empty/whitespace-only rows

The cleaned dataset is exported to **`test.xlsx`** for the classification stage.

**Exploratory analysis**
- **Top 10 most frequent words** (bar plot)
- **Top 10 bigrams** and **trigrams** via `CountVectorizer`
- **Part-of-speech distribution** using TextBlob POS tagging

---

## 🤖 Notebook 2 — Text Classification
**`FQ_Text_Class.ipynb`**

Trains and evaluates a classifier that maps email text to its business unit.

**Feature engineering**
- Loads the cleaned `test.xlsx` and encodes the target label into numeric `category_id` via `factorize()`.
- Visualises class balance (number of mails per unit).
- **TF-IDF vectorisation**: `sublinear_tf=True`, `min_df=5`, `ngram_range=(1, 2)`, English stop-words.
- **chi² feature analysis** to surface the most correlated unigrams and bigrams for each category.

**Model comparison** (5-fold cross-validation)
- `RandomForestClassifier`
- `LinearSVC`
- `MultinomialNB`
- `LogisticRegression`

Mean accuracy and standard deviation are compared across folds (box plot), and the strongest performer — **`LinearSVC`** — is selected as the final model.

**Evaluation & inference**
- Train/test split (75/25), full `classification_report`, and a confusion matrix heatmap.
- A final TF-IDF + LinearSVC model is fitted and used to **predict the unit for brand-new, unseen email text**.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python (Jupyter Notebook) |
| **Data handling** | pandas, numpy, openpyxl |
| **NLP / preprocessing** | spaCy (`en_core_web_sm`), TextBlob, BeautifulSoup4, regex |
| **Feature extraction** | scikit-learn `TfidfVectorizer`, `CountVectorizer` |
| **Modelling** | scikit-learn — LinearSVC, MultinomialNB, LogisticRegression, RandomForest |
| **Evaluation** | cross-validation, classification report, confusion matrix, chi² |
| **Visualisation** | matplotlib, seaborn |
| **Progress** | tqdm |

---

## 📁 Repository Structure

```
email-text-classifier/
├── FQ_Data_Analysis.ipynb   # Stage 1: cleaning, preprocessing & EDA  ➜ exports test.xlsx
├── FQ_Text_Class.ipynb      # Stage 2: TF-IDF + model training/evaluation
├── FQ_Data.xlsx             # (not committed) raw email dataset — input to Stage 1
├── test.xlsx                # (generated) cleaned dataset — input to Stage 2
├── requirements.txt         # (recommended) pinned dependencies
└── README.md
```

---

## ✅ Prerequisites & Installation

```bash
# Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn \
            spacy textblob beautifulsoup4 openpyxl tqdm scipy

# Download the spaCy English model
python -m spacy download en_core_web_sm

# (TextBlob may need its corpora on first run)
python -m textblob.download_corpora
```

---

## 🕹 Usage

Run the notebooks **in order** — Stage 2 depends on the file Stage 1 produces.

1. **`FQ_Data_Analysis.ipynb`**
   - Update the input path to point at your email workbook (the notebook currently reads from a local path).
   - Run all cells. This cleans the text, produces the EDA charts, and writes **`test.xlsx`**.

2. **`FQ_Text_Class.ipynb`**
   - Run all cells to vectorise the cleaned data, compare models, train the final **LinearSVC**, and review the evaluation outputs.
   - Use the prediction cells at the end to classify new email text:
     ```python
     model.predict(fitted_vectorizer.transform([your_email_text]))
     ```

---

## 📊 Model Selection & Results

Four classifiers were benchmarked with 5-fold cross-validation on TF-IDF features (unigrams + bigrams). **LinearSVC** gave the best mean accuracy and was chosen for deployment. The final model is evaluated with a per-class classification report and a confusion matrix across the business-unit categories, and chi² analysis is used to interpret which terms most strongly signal each unit.

> ℹ️ Exact accuracy figures depend on the dataset you run against — re-run Notebook 2 to populate the comparison table and confusion matrix for your data.

---


---

## 🔮 Future Work

- Add a `requirements.txt` and parameterise file paths/config.
- Persist the fitted vectoriser + model (e.g. `joblib`) so inference doesn't require retraining.
- Handle the multi-label emails (currently filtered out) with a multi-label classifier.
- Address class imbalance across units (resampling or class weights).
- Wrap the final model in a small API or script for live email routing.
- Expand evaluation (precision/recall per unit, macro-F1) and add hyperparameter tuning.

---

## 👤 Author

**Darren McCann**

*Built as an applied NLP / text-classification project for automated email routing.*
