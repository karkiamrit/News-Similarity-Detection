#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/karkiamrit/News-Similarity-Detection/blob/main/combined.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # News Article Similarity Pipeline
# ## Siamese Embedding + 5W1H Extraction
# 
# ---
# 
# ## 📋 Overview
# 
# This pipeline performs two major analytical tasks on news articles:
# 
# ### 1. **Semantic Similarity Analysis**
# Extracts and compares the first two paragraphs of news articles using:
# - **SentenceTransformer** (MiniLM) embeddings
# - **Cosine similarity** metrics
# 
# ### 2. **5W1H Entity Extraction**
# Run manually in terminal using
# 
# `java -Xmx4g -cp "$(echo $HOME/.stanfordnlp_resources/stanford-corenlp-4.5.7/*.jar | tr ' ' ':')" \
# edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -port 9010 -timeout 500000 \
# -annotators tokenize,ssplit,pos,lemma,ner,parse,depparse,coref \
# -preload tokenize,ssplit,pos,lemma,ner,parse,depparse,coref \
# -coref.algorithm neural
# `
# 
# Extracts journalism's fundamental questions from article titles:
# - **Who** - People/Organizations involved
# - **What** - Events/Actions that occurred
# - **When** - Temporal information
# - **Where** - Locations/Places
# - **Why** - Reasons/Motivations
# - **How** - Methods/Processes
# 
# Uses **Giveme5W1H** library with **Stanford CoreNLP Server** backend.
# ---
# 
# ## 📚 Key Libraries
# 
# - **sentence-transformers**: Neural sentence embeddings
# - **trafilatura**: Web scraping & text extraction
# - **Giveme5W1H**: 5W1H entity extraction
# - **Stanford CoreNLP**: NLP backend for entity recognition
# - **scikit-learn**: Cosine similarity computation
# 

# ### Dependencies and Imports (Siamese)

# In[1]:


get_ipython().system('pip -q install sentence-transformers trafilatura readability-lxml bs4 lxml html5lib tqdm')
get_ipython().system('pip -q install giveme5w1h geopy')


# In[3]:


# ----------------------------
#  Import required libraries
# ----------------------------
import re, math, time, sys, warnings
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import trafilatura
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer


# #### Helper functions
# 
# > Add blockquote
# 
# 

# In[26]:


# ----------------------------
#  Helper: URL validation
# ----------------------------
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsSimilarityBot/1.0; +https://example.org/bot)"}

def is_valid_url(u: str) -> bool:
    """Check if a given string is a valid HTTP/HTTPS URL."""
    if not isinstance(u, str) or not u.strip():
        return False
    p = urlparse(u.strip())
    return p.scheme in {"http", "https"} and bool(p.netloc)


# ----------------------------
#  Extract first two paragraphs from HTML
# ----------------------------
def first_two_paragraphs_from_html(html: str):
    """Return the first two meaningful paragraphs from a raw HTML document."""
    soup = BeautifulSoup(html, "lxml")
    root = soup.find("article") or soup.find("main") or soup
    paras = []
    for p in root.find_all("p"):
        txt = re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
        if len(txt) >= 40:  # ignore very short boilerplate text
            paras.append(txt)
        if len(paras) >= 2:
            break
    # fallback: if not enough paragraphs found
    if len(paras) < 2:
        paras = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip()
                 for p in root.find_all("p") if p.get_text(strip=True)]
        paras = [x for x in paras if x][:2]
    p1 = paras[0] if len(paras) > 0 else None
    p2 = paras[1] if len(paras) > 1 else None
    return p1, p2


# ----------------------------
#  Fetch paragraphs (Trafilatura → BeautifulSoup fallback)
# ----------------------------
def fetch_first_two_paragraphs(url: str, timeout=12):
    """
    Try extracting readable text via Trafilatura first,
    then fall back to raw HTML parsing using BeautifulSoup.
    Returns (p1, p2, status) where status ∈ {'ok','invalid_url','fetch_error','no_content'}.
    """
    if not is_valid_url(url):
        return None, None, "invalid_url"

    # --- Attempt using Trafilatura ---
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if text:
                # Split into blocks on blank lines
                blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
                blocks = [b for b in blocks if len(b) >= 40]
                p1 = blocks[0] if len(blocks) > 0 else None
                p2 = blocks[1] if len(blocks) > 1 else None
                if p1 or p2:
                    return p1, p2, "ok"
    except Exception:
        pass  # fallback to HTML parsing

    # --- Fallback using raw HTML ---
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.ok and r.text:
            p1, p2 = first_two_paragraphs_from_html(r.text)
            if p1 or p2:
                return p1, p2, "ok"
            return None, None, "no_content"
    except (requests.Timeout, requests.ConnectionError):
        # Skip this URL silently if slow or unreachable
        return None, None, "fetch_error"
    except Exception:
        return None, None, "fetch_error"


# #### Embeddings and similarity *functions*

# In[24]:


# ----------------------------
#  Initialize embedding model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str):
    """Return a unit-normalized embedding for text."""
    if not isinstance(text, str) or not text.strip():
        return None
    v = model.encode([text.strip()], normalize_embeddings=True)[0]
    return v

def cosine_sim(v1, v2):
    """Compute cosine similarity between two vectors."""
    if v1 is None or v2 is None:
        return np.nan
    return float(np.dot(v1, v2))


# ##### Load and peek dataset

# In[7]:


from google.colab import drive
drive.mount('/content/drive')


# In[19]:


# ================================================================
#  STEP 1: Load Dataset
# ================================================================
df = pd.read_csv('drive/MyDrive/zenodo_with_url_exists.csv')
# Basic info and sanity check
print(f"✅ Loaded {len(df)} rows from zenodo_with_url_exists.csv")


# In[20]:


df.head()
df = df.head(2)


# ##### New URL Columns Appended

# In[21]:


url_cols = [c for c in ("content.url1", "content.url2") if c in df.columns]

# Sanity check: ensure both expected columns exist
assert len(url_cols) == 2, (
    f"Expected columns content.url1 & content.url2, but found: {url_cols}"
)
print(f"✅ Found URL columns: {url_cols}")


# ##### Extraction, Embedding and Comparison Using Siamese Model
#  - Extraction of text using helper functions
#  - Embedded text from each set of urls
#  - Calculated cosine similarity using the embeddings
#  - Results Storage and Display
# 

# In[25]:


# ================================================================
#  STEP 2: Extract, Embed, and Compare using Siamese Model
# ================================================================
rows = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Crawling + Encoding"):
    u1, u2 = row[url_cols[0]], row[url_cols[1]]

    # --- Fetch first 2 paragraphs from both URLs ---
    p1a, p1b, s1 = fetch_first_two_paragraphs(u1) if is_valid_url(u1) else (None, None, "invalid_url")
    p2a, p2b, s2 = fetch_first_two_paragraphs(u2) if is_valid_url(u2) else (None, None, "invalid_url")

    # --- Combine first + second paragraphs ---
    t1 = " ".join([x for x in (p1a, p1b) if x])
    t2 = " ".join([x for x in (p2a, p2b) if x])

    # --- Compute embeddings + cosine similarity ---
    v1 = embed(t1)
    v2 = embed(t2)
    cos = cosine_sim(v1, v2)

    rows.append({
        "content.url1": u1,
        "content.url2": u2,
        "url1.p1": p1a, "url1.p2": p1b, "url1.status": s1,
        "url2.p1": p2a, "url2.p2": p2b, "url2.status": s2,
        "siamese.text1": t1 or None,
        "siamese.text2": t2 or None,
        "similarity.cosine": cos,
    })

# --- Merge results into main dataframe ---
result = pd.DataFrame(rows)
siamese_output = df.join(result.drop(columns=["content.url1", "content.url2"]).set_index(result.index))
siamese_output.to_csv("siamese_output.csv", index=False)

# --- Preview results instead of saving ---
print("✅ Siamese comparison complete.")
display(siamese_output.head())
print(f"Total processed pairs: {len(siamese_output)}")


# ##### Final Result Siamese

# In[ ]:


siamese_output.head()


# 

# ### Dependencies and Imports (5W1H)

# In[14]:


import os
import subprocess
import time
import requests
import zipfile
from datetime import datetime, timezone
from Giveme5W1H.extractor.extractor import MasterExtractor
from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractors.environment_extractor import EnvironmentExtractor
from Giveme5W1H.extractor.preprocessors.preprocessor_core_nlp import Preprocessor
from geopy.geocoders import options
import pandas as pd


# ##### Giveme5W1H extractor and CoreNLP preprocessor setup

# ###### Install Dependencies and initialize stanza

# In[15]:


# Install Java and stanza
get_ipython().system('apt-get update -q')
get_ipython().system('apt-get install -y openjdk-17-jdk unzip wget -q')
get_ipython().system('pip install stanza requests -q')

# Download and install CoreNLP via stanza helper
import stanza
stanza.install_corenlp()   # This will download stanford-corenlp-4.5.5 to ~/.stanfordnlp_resources/


# In[ ]:


# from stanza.server import CoreNLPClient

# with CoreNLPClient(
#     annotators=['tokenize','ssplit','pos','lemma','ner','parse','depparse'],
#     timeout=60000,
#     memory='2G',
#     be_quiet=False
# ) as client:
#     ann = client.annotate("Barack Obama was born in Hawaii.")
#     print(ann)


# ###### Connect to preprocessor server and configure Giveme5W1H extractor

# In[28]:


def download_and_setup_corenlp():
    """Download and extract Stanford CoreNLP."""
    base_path = os.path.expanduser("~/.stanfordnlp_resources")
    corenlp_version = "4.5.7"
    corenlp_dir = f"{base_path}/stanford-corenlp-{corenlp_version}"
    os.makedirs(base_path, exist_ok=True)

    if os.path.exists(corenlp_dir):
        print(f"✓ CoreNLP already exists at {corenlp_dir}")
        return corenlp_dir

    # Download CoreNLP
    url = f"https://nlp.stanford.edu/software/stanford-corenlp-{corenlp_version}.zip"
    zip_path = f"{base_path}/stanford-corenlp-{corenlp_version}.zip"
    print(f"⬇️  Downloading CoreNLP {corenlp_version} (~500 MB)...")

    result = subprocess.run(["wget", "-q", "--show-progress", url, "-O", zip_path])
    if result.returncode != 0:
        raise RuntimeError(f"Failed to download CoreNLP from {url}")

    print("📦 Extracting CoreNLP...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(base_path)
    os.remove(zip_path)

    print(f"✓ CoreNLP installed at {corenlp_dir}")
    return corenlp_dir


def start_corenlp_server(corenlp_home, port=9020):
    """Start CoreNLP server in background."""
    try:
        requests.get(f"http://127.0.0.1:{port}", timeout=2)
        print(f"✓ CoreNLP server already running on port {port}")
        return
    except Exception:
        pass

    print(f"🚀 Starting CoreNLP server on port {port}...")
    cmd = [
          "java", "-Xmx4g",
          "-cp", f"{corenlp_home}/*",
          "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
          "--port", str(port),
          "--timeout", "500000",
          "--annotators", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref",
          "--preload", "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref",
          "--coref.algorithm", "neural"
      ]
    process = subprocess.Popen(cmd, cwd=corenlp_home,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    print("⏳ Waiting for server to start", end="")
    for i in range(90):
        try:
            requests.get(f"http://127.0.0.1:{port}", timeout=2)
            print(f"\n✓ CoreNLP server started successfully (took {i+1}s)")
            return
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    raise RuntimeError("❌ Failed to start CoreNLP server after 90 s")



# In[37]:


# Initialize Giveme5W1H extractor and CoreNLP preprocessor
corenlp_home = download_and_setup_corenlp()
# start_corenlp_server(corenlp_home, port=9020) only if not manually ran
options.default_user_agent = "colab-giveme5w1h"
try:
  pre = Preprocessor("http://127.0.0.1:9020")  # assumes CoreNLP server is running
  pre._Preprocessor__default_annotators = "tokenize,ssplit,pos,lemma,ner,parse,depparse,coref"

  print("✅ CoreNLP server is running!")
except Exception as e:
    print("❌ CoreNLP server not reachable:", e)

extractor = MasterExtractor(preprocessor=pre)

# remove environment extractor for speed
extractor.extractors = [e for e in extractor.extractors if not isinstance(e, EnvironmentExtractor)]

TITLE_COLS = [c for c in ("content.title1", "content.title2") if c in df.columns]


# Helper function to extract 5w1h from title and process it

# ###### Sanitize Title
# 

# In[19]:


import html, re, json

def sanitize_title(title: str) -> str:
    """Clean up titles for better NLP parsing."""
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'[^\w\s,\'"-]', '', title)
    return title.strip()


# In[26]:


def ner_fallback(text: str):
    """Fallback extractor using CoreNLP NER for short texts."""
    props = {"annotators": "tokenize,ssplit,pos,lemma,ner", "outputFormat": "json"}
    try:
        r = requests.post(
            "http://127.0.0.1:9020",
            params={"properties": json.dumps(props)},
            data=text.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=15
        )
        data = r.json()
        entities = [(ent["text"], ent["ner"])
                    for sent in data.get("sentences", [])
                    for ent in sent.get("entitymentions", [])]

        who = " ".join([t for t, n in entities if n in {"PERSON", "ORGANIZATION"}]) or None
        where = " ".join([t for t, n in entities if n in {"LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"}]) or None
        when = " ".join([t for t, n in entities if n in {"DATE", "TIME"}]) or None

        what = re.sub(r"[^\w\s,'-]", "", text).strip() or None
        return {"who": who, "what": what, "when": when, "where": where, "why": None, "how": None}
    except Exception as e:
        print(f"❌ NER fallback failed: {e}")
        return {k: None for k in ["who", "what", "when", "where", "why", "how"]}


def extract_5w1h_from_title(title: str):
    """Main extractor with GiveMe5W1H + fallback."""
    if not isinstance(title, str) or not title.strip():
        return {k: None for k in ["who", "what", "when", "where", "why", "how"]}
    try:
        doc = Document.from_text(title)
        doc = extractor.parse(doc)
        def safe_extract(q):
            try:
                return doc.get_top_answer(q).get_parts_as_text()
            except Exception:
                return None
        result = {q: safe_extract(q) for q in ["who", "what", "when", "where", "why", "how"]}
        if all(v is None for v in result.values()):
            result.update(ner_fallback(title))
        return result
    except Exception as e:
        print(f"⚠️ Fallback triggered for '{title[:60]}...': {e}")
        return ner_fallback(title)



# In[41]:


print(extract_5w1h_from_title("Porter’s factor conditions for countries’ competitiveness are demand condition, related industries, firm’s strategy, and the level of rivalry. Australia has been challenged for the trophy and there was an increased demand for the country to produce results in the game. The country through several failures to lift the trophy had learned its weaknesses, and in 2005, it went to the games with polished strategies to face their rivals. This led to its victory."))


# ##### Text Extraction

# In[ ]:


# ============================================================
# STEP 3: Apply Across Titles
# ============================================================
def process_titles_once(df, title_cols):
    """Apply 5W1H extraction to all given title columns with progress tracking."""
    result_df = df.copy()
    for col in title_cols:
        if col not in result_df.columns:
            print(f"⚠️ Skipping {col}: not found in dataframe")
            continue

        print(f"\n🚀 Processing {col} ({len(result_df)} titles)")
        results = []
        for idx, title in enumerate(result_df[col].fillna("")):
            if idx % 10 == 0:
                print(f"  → Progress: {idx}/{len(result_df)}", end="\r")
            if idx % 100 == 0 and idx > 0:
                time.sleep(0.3)

            clean_title = re.sub(r"\s+", " ", str(title)).strip()
            if clean_title:
                results.append(extract_5w1h_from_title(clean_title))
            else:
                results.append({k: None for k in ["who", "what", "when", "where", "why", "how"]})

        col_results = pd.DataFrame.from_records(results, index=result_df.index)
        for q in col_results.columns:
            result_df[f"{col}.{q}"] = col_results[q].values
        print(f"✅ Completed {col} ({len(result_df)} titles)")

    result_df = result_df.loc[:, ~result_df.columns.duplicated(keep="last")]
    return result_df


# In[ ]:


print("\n" + "="*60)
print("Starting 5W1H extraction...")
print("="*60)

TITLE_COLS = [c for c in ("content.title1", "content.title2") if c in df.columns]
result_df = process_titles_once(df, TITLE_COLS)

preview_cols = [f"{c}.{q}" for c in TITLE_COLS for q in ("who","what","when","where","why","how")]
keep = [c for c in ["content.url1","content.url2","content.title1","content.title2"] if c in result_df.columns] + [c for c in preview_cols if c in result_df.columns]
result_df = result_df[keep]


# In[ ]:


#Testing the connection
# import requests, json
# text = sanitize_title("NASA launches Artemis mission to the Moon")
# props = {
#     "annotators": "tokenize,ssplit,pos,lemma,ner",
#     "outputFormat": "json"
# }
# r = requests.post(
#     "http://127.0.0.1:9010",
#     params={"properties": json.dumps(props)},
#     data=text.encode("utf-8"),
#     headers={"Content-Type": "text/plain; charset=utf-8"}
# )
# print(r.status_code)
# print(r.text[:300])


# ##### Save and Display results

# In[16]:


display(result_df.head())
fivew1h_output = result_df
fivew1h_output.to_csv("fivew1h_output.csv", index=False)


# ##### Entity Based Similarity & Fusion with Siamese Network

# In[ ]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


# In[ ]:


# Use the previous 5W1H output directly (not from CSV)
# fivew1h_output already has content.title1.*, content.title2.* columns
fusion_df = fivew1h_output.copy()


# In[ ]:


# --- Helper functions ---
def calculate_entity_overlap(entity1, entity2):
    """Calculate overlap between two entities (Jaccard similarity)."""
    if not entity1 or not entity2:
        return 0.0

    set1 = set(str(entity1).lower().split())
    set2 = set(str(entity2).lower().split())

    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    if len(set1) == 0 or len(set2) == 0:
        return 0.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_5w1h_similarity(row):
    """Calculate entity-based similarity for a pair of articles."""
    questions = ['who', 'what', 'when', 'where', 'why', 'how']
    similarities = {}

    for q in questions:
        col1 = f'content.title1.{q}'
        col2 = f'content.title2.{q}'

        if col1 in row.index and col2 in row.index:
            entity1 = row[col1]
            entity2 = row[col2]
            similarities[q] = calculate_entity_overlap(entity1, entity2)
        else:
            similarities[q] = 0.0

    avg_entity_sim = np.mean(list(similarities.values()))
    return pd.Series({
        'entity_who_sim': similarities['who'],
        'entity_what_sim': similarities['what'],
        'entity_when_sim': similarities['when'],
        'entity_where_sim': similarities['where'],
        'entity_why_sim': similarities['why'],
        'entity_how_sim': similarities['how'],
        'avg_entity_similarity': avg_entity_sim
    })

# --- Apply entity similarity computation ---
entity_similarities = fusion_df.apply(calculate_5w1h_similarity, axis=1)
fusion_df = pd.concat([fusion_df, entity_similarities], axis=1)

print("✓ Entity-based similarities calculated successfully.")

# Handle duplicated or non-standard column safely
if 'avg_entity_similarity' in fusion_df.columns:
    # If there are duplicate columns, keep only the last one
    if isinstance(fusion_df['avg_entity_similarity'], pd.DataFrame):
        avg_sim_col = fusion_df['avg_entity_similarity'].iloc[:, -1]
    else:
        avg_sim_col = fusion_df['avg_entity_similarity']

    # Ensure Series is flat numeric
    avg_sim_flat = pd.to_numeric(avg_sim_col.astype(str).str.extract(r'([\d.]+)')[0], errors='coerce')
    mean_val = float(avg_sim_flat.mean()) if not avg_sim_flat.isna().all() else 0.0
    print(f"  - Average entity similarity across dataset: {mean_val:.4f}")
else:
    print("  - Warning: avg_entity_similarity column not found.")
# --- Preview output ---
preview_cols = [
    "content.url1", "content.url2",
    "content.title1", "content.title2",
    "avg_entity_similarity", "entity_who_sim", "entity_what_sim", "entity_when_sim",
    "entity_where_sim", "entity_why_sim", "entity_how_sim"
]
print("\nSample output:")
display(fusion_df[preview_cols].head())

# For later fusion with Siamese results
entity_output = fusion_df


# ###### STEP 5: Fusion Layer (Entity + Embedding Similarity)

# In[ ]:


# ================================================================
#  STEP 5: Fusion Layer — Combine Entity and Siamese Similarities
# ================================================================
print("\n" + "=" * 60)
print("STEP 5: Fusion Layer (Entity + Siamese Network)")
print("=" * 60)

# --- Merge Siamese and Entity-based results ---
fusion_df = pd.merge(
    siamese_output,
    entity_output[["content.url1", "content.url2", "avg_entity_similarity"]],
    on=["content.url1", "content.url2"],
    how="left"
)

# --- Rename for clarity ---
fusion_df = fusion_df.loc[:, ~fusion_df.columns.duplicated()]  # remove duplicates
fusion_df.rename(columns={"similarity.cosine": "siamese_similarity"}, inplace=True)

# --- Ensure numeric columns ---
fusion_df["avg_entity_similarity"] = pd.to_numeric(
    fusion_df["avg_entity_similarity"], errors="coerce"
).fillna(0)
fusion_df["siamese_similarity"] = pd.to_numeric(
    fusion_df["siamese_similarity"], errors="coerce"
).fillna(0)

# --- Define fusion function ---
def fusion_similarity(entity_sim, siamese_sim, entity_weight=0.4, siamese_weight=0.6):
    """Combine entity-based and embedding-based (Siamese) similarities."""
    return (entity_weight * entity_sim) + (siamese_weight * siamese_sim)

# --- Compute fusion vectorized (faster & safer than .apply) ---
fusion_df["fused_similarity"] = fusion_similarity(
    fusion_df["avg_entity_similarity"],
    fusion_df["siamese_similarity"],
    entity_weight=0.4,
    siamese_weight=0.6,
)

# --- Summary statistics ---
mean_val = fusion_df["fused_similarity"].mean()
print("✓ Fusion similarity calculated successfully.")
print(f"  - Average fused similarity: {mean_val:.4f}")
print(f"  - Min fused similarity: {fusion_df['fused_similarity'].min():.4f}")
print(f"  - Max fused similarity: {fusion_df['fused_similarity'].max():.4f}")

# --- Preview ---
display(
    fusion_df[
        [
            "content.url1",
            "content.url2",
            "siamese_similarity",
            "avg_entity_similarity",
            "fused_similarity",
        ]
    ].head()
)

# For final saving
final_output = fusion_df


# #####  STEP 7: Pearson Correlation Analysis
# 

# In[ ]:


# ================================================================
#  STEP 7: Pearson Correlation Analysis
# ================================================================
from scipy.stats import pearsonr

print("\n" + "=" * 60)
print("STEP 7: Pearson Correlation Analysis")
print("=" * 60)

# Use the latest unified dataframe from Step 5
correlation_df = final_output.copy()

# Identify available columns
correlation_cols = [
    "entity_who_sim", "entity_what_sim", "entity_when_sim",
    "entity_where_sim", "entity_why_sim", "entity_how_sim",
    "avg_entity_similarity", "siamese_similarity", "fused_similarity",
]
correlation_cols = [col for col in correlation_cols if col in correlation_df.columns]

# --- Compute correlation matrix ---
if correlation_cols:
    correlation_matrix = correlation_df[correlation_cols].corr(method="pearson")
    print("\nPearson Correlation Matrix:")
    print(correlation_matrix.round(3))
else:
    print("⚠️ No valid similarity columns found for correlation analysis.")

# --- Key correlations ---
print("\n" + "-" * 60)
print("Key Correlations:")
print("-" * 60)

def safe_corr(x, y):
    """Compute Pearson correlation safely."""
    try:
        corr, p_val = pearsonr(
            correlation_df[x].fillna(0).astype(float),
            correlation_df[y].fillna(0).astype(float),
        )
        return corr, p_val
    except Exception:
        return np.nan, np.nan

if "avg_entity_similarity" in correlation_df.columns and "siamese_similarity" in correlation_df.columns:
    corr, p_val = safe_corr("avg_entity_similarity", "siamese_similarity")
    print(f"Entity vs Siamese Similarity: r={corr:.4f}, p={p_val:.4e}")

if "fused_similarity" in correlation_df.columns and "siamese_similarity" in correlation_df.columns:
    corr, p_val = safe_corr("fused_similarity", "siamese_similarity")
    print(f"Fused vs Siamese Similarity: r={corr:.4f}, p={p_val:.4e}")

if "fused_similarity" in correlation_df.columns and "avg_entity_similarity" in correlation_df.columns:
    corr, p_val = safe_corr("fused_similarity", "avg_entity_similarity")
    print(f"Fused vs Entity Similarity: r={corr:.4f}, p={p_val:.4e}")


# #####  STEP 8: Save Results
# 

# In[ ]:


final_df = final_output.copy()


# In[ ]:


print("\n" + "="*60)
print("STEP 8: Saving Results")
print("="*60)

final_columns = [
    'content.url1', 'content.url2',
    'content.title1', 'content.title2',
    'content.title1.who', 'content.title1.what', 'content.title1.when',
    'content.title1.where', 'content.title1.why', 'content.title1.how',
    'content.title2.who', 'content.title2.what', 'content.title2.when',
    'content.title2.where', 'content.title2.why', 'content.title2.how',
    'entity_who_sim', 'entity_what_sim', 'entity_when_sim',
    'entity_where_sim', 'entity_why_sim', 'entity_how_sim',
    'avg_entity_similarity', 'siamese_similarity', 'fused_similarity'
]

final_columns = [c for c in final_columns if c in final_df.columns]

# --- save results ---
output_file = "final_similarity_results.csv"
final_df[final_columns].to_csv(output_file, index=False)
print(f"✓ Saved complete results to: {output_file}")

# --- save correlation matrix ---
correlation_output = "correlation_matrix.csv"
correlation_matrix.to_csv(correlation_output)
print(f"✓ Saved correlation matrix to: {correlation_output}")

# --- save summary statistics (only existing numeric cols) ---
summary_cols = [c for c in ["avg_entity_similarity", "siamese_similarity", "fused_similarity"]
                if c in final_df.columns]
if summary_cols:
    summary_stats = final_df[summary_cols].describe()
    summary_output = "similarity_statistics.csv"
    summary_stats.to_csv(summary_output)
    print(f"✓ Saved summary statistics to: {summary_output}")
else:
    print("⚠️ No similarity columns found for summary statistics.")


# #####  STEP 9: Visualization and Analysis
# 

# In[ ]:


print("\n" + "="*60)
print("STEP 9: Summary Statistics & Analysis")
print("="*60)

# Display summary statistics
print("\nSimilarity Score Distribution:")
print(final_df[correlation_cols].describe().round(4))

# Find top similar pairs
print("\n" + "-"*60)
print("Top 10 Most Similar Article Pairs (by Fused Similarity):")
print("-"*60)
top_similar = final_df.nlargest(10, 'fused_similarity')[
    ['content.title1', 'content.title2', 'avg_entity_similarity',
     'siamese_similarity', 'fused_similarity']
]
print(top_similar.to_string(index=False))

# Find least similar pairs
print("\n" + "-"*60)
print("Top 10 Least Similar Article Pairs (by Fused Similarity):")
print("-"*60)
least_similar = final_df.nsmallest(10, 'fused_similarity')[
    ['content.title1', 'content.title2', 'avg_entity_similarity',
     'siamese_similarity', 'fused_similarity']
]
print(least_similar.to_string(index=False))

print("\n" + "="*60)
print("✓ Analysis Complete!")
print("="*60)
print(f"\nGenerated files:")
print(f"  1. {output_file}")
print(f"  2. {correlation_output}")
print(f"  3. {summary_output}")


# #####  STEP 10: Quantitative Evaluation Metrics
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print("STEP 10: Quantitative Evaluation Metrics")
print("="*60)


# In[ ]:


# ================================================================
#  Convert Continuous Scores to Binary Predictions
# ================================================================

def evaluate_at_threshold(y_true, y_scores, threshold):
    """Evaluate predictions at a specific threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# In[ ]:


# ================================================================
#  STEP 10: Use Real Ground Truth Labels
# ================================================================
print("\n" + "="*60)
print("STEP 10: Using Real Ground Truth Labels (content.similarity)")
print("="*60)

if "content.similarity" in final_df.columns:
    # Use 0.5 as threshold to convert similarity into binary labels
    final_df["ground_truth"] = (final_df["content.similarity"] >= 0.5).astype(int)
    print("✅ Ground truth labels loaded from 'content.similarity'.")
else:
    raise ValueError("❌ 'content.similarity' column not found in dataset.")

print(f"Total labelled pairs: {len(final_df)}")
print(final_df[["content.similarity", "fused_similarity", "ground_truth"]].head())


# In[ ]:


# Test multiple thresholds
thresholds_to_test = np.arange(0.1, 1.0, 0.05)
y_true = final_df['ground_truth'].values

# Evaluate each similarity measure
similarity_measures = {
    'Entity Similarity': 'avg_entity_similarity',
    'Siamese Similarity': 'siamese_similarity',
    'Fused Similarity': 'fused_similarity'
}

results_by_measure = {}

for measure_name, measure_col in similarity_measures.items():
    if measure_col in final_df.columns:
        y_scores = final_df[measure_col].fillna(0).values

        # Evaluate at different thresholds
        threshold_results = []
        for thresh in thresholds_to_test:
            metrics = evaluate_at_threshold(y_true, y_scores, thresh)
            threshold_results.append(metrics)

        results_by_measure[measure_name] = {
            'scores': y_scores,
            'threshold_results': pd.DataFrame(threshold_results)
        }


# In[ ]:


print("\n" + "-"*60)
print("Optimal Thresholds (Maximum F1 Score):")
print("-"*60)

optimal_thresholds = {}

for measure_name, data in results_by_measure.items():
    df_thresh = data['threshold_results']
    optimal_idx = df_thresh['f1'].idxmax()
    optimal_row = df_thresh.iloc[optimal_idx]
    optimal_thresholds[measure_name] = optimal_row['threshold']

    print(f"\n{measure_name}:")
    print(f"  Optimal Threshold: {optimal_row['threshold']:.3f}")
    print(f"  Accuracy:  {optimal_row['accuracy']:.4f}")
    print(f"  Precision: {optimal_row['precision']:.4f}")
    print(f"  Recall:    {optimal_row['recall']:.4f}")
    print(f"  F1 Score:  {optimal_row['f1']:.4f}")


# ######   Confusion Matrix
# 
# 

# In[ ]:


print("\n" + "="*60)
print("Generating Confusion Matrices...")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices at Optimal Thresholds', fontsize=16, fontweight='bold')

for idx, (measure_name, measure_col) in enumerate(similarity_measures.items()):
    if measure_col in final_df.columns:
        y_scores = final_df[measure_col].fillna(0).values
        threshold = optimal_thresholds[measure_name]
        y_pred = (y_scores >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar=False, ax=axes[idx],
                   xticklabels=['Not Similar', 'Similar'],
                   yticklabels=['Not Similar', 'Similar'])
        axes[idx].set_title(f'{measure_name}\n(Threshold: {threshold:.3f})')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.show()


# In[ ]:


# ================================================================
#  Precision-Recall Curves
# ================================================================

print("\n" + "="*60)
print("Generating Precision-Recall Curves...")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

for measure_name, data in results_by_measure.items():
    y_scores = data['scores']
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, label=f'{measure_name} (AUC={pr_auc:.3f})', linewidth=2)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: precision_recall_curves.png")
plt.show()


# In[ ]:


# ================================================================
#  ROC Curves
# ================================================================

print("\n" + "="*60)
print("Generating ROC Curves...")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

for measure_name, data in results_by_measure.items():
    y_scores = data['scores']
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f'{measure_name} (AUC={roc_auc:.3f})', linewidth=2)

# Plot diagonal line
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: roc_curves.png")
plt.show()

# ============


# In[ ]:


# ================================================================
#  F1 Score vs Threshold Curves
# ================================================================

print("\n" + "="*60)
print("Generating F1 Score vs Threshold Curves...")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

for measure_name, data in results_by_measure.items():
    df_thresh = data['threshold_results']
    ax.plot(df_thresh['threshold'], df_thresh['f1'],
           label=measure_name, linewidth=2, marker='o', markersize=3)

ax.set_xlabel('Threshold', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('f1_vs_threshold.png', dpi=300, bbox_inches='tight')
print("✓ Saved: f1_vs_threshold.png")
plt.show()


# In[ ]:


# ================================================================
#  Precision, Recall, F1 vs Threshold (Combined)
# ================================================================

print("\n" + "="*60)
print("Generating Combined Metrics vs Threshold...")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Metrics vs Threshold for Different Similarity Measures',
             fontsize=16, fontweight='bold')

metric_names = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1 Score']

for idx, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    for measure_name, data in results_by_measure.items():
        df_thresh = data['threshold_results']
        axes[idx].plot(df_thresh['threshold'], df_thresh[metric],
                      label=measure_name, linewidth=2, marker='o', markersize=3)

    axes[idx].set_xlabel('Threshold', fontsize=12)
    axes[idx].set_ylabel(label, fontsize=12)
    axes[idx].set_title(f'{label} vs Threshold', fontsize=12, fontweight='bold')
    axes[idx].legend(loc='best', fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim([0, 1])
    axes[idx].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('metrics_vs_threshold.png', dpi=300, bbox_inches='tight')
print("✓ Saved: metrics_vs_threshold.png")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report

print("\n" + "="*60)
print("Classification Reports at Optimal Thresholds:")
print("="*60)

df_eval = final_df if 'fused_similarity' in final_df.columns else result_df.copy()

for measure_name, measure_col in similarity_measures.items():
    if measure_col in df_eval.columns:
        y_scores = df_eval[measure_col].fillna(0).values
        threshold = optimal_thresholds[measure_name]
        y_pred = (y_scores >= threshold).astype(int)

        print(f"\n{measure_name} (Threshold: {threshold:.3f}):")
        print("-" * 60)
        print(classification_report(
            y_true, y_pred,
            labels=[0, 1],  # <- ensures consistent output
            target_names=['Not Similar', 'Similar'],
            digits=4,
            zero_division=0
        ))
    else:
        print(f"⚠️ Skipping {measure_name}: column '{measure_col}' not found.")


# In[ ]:


print("\n" + "="*60)
print("Comparative Performance Summary:")
print("="*60)

# Pick correct dataframe dynamically
df_eval = final_df if 'fused_similarity' in final_df.columns else result_df.copy()

performance_data = []

for measure_name, measure_col in similarity_measures.items():
    if measure_col not in df_eval.columns:
        print(f"⚠️ Skipping '{measure_name}' — missing column '{measure_col}'.")
        continue

    y_scores = df_eval[measure_col].fillna(0).values
    threshold = optimal_thresholds.get(measure_name, np.median(y_scores))
    y_pred = (y_scores >= threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Compute AUC scores
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec, prec)

    performance_data.append({
        "Measure": measure_name,
        "Optimal Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

# --- Build dataframe safely ---
if len(performance_data) == 0:
    print("❌ No valid similarity columns found. Please check final_df contents.")
    print("Available columns:", df_eval.columns.tolist())
else:
    performance_df = pd.DataFrame(performance_data)
    print("\n", performance_df.to_string(index=False))
    performance_df.to_csv("performance_comparison.csv", index=False)
    print("\n✓ Saved: performance_comparison.csv")


# In[ ]:


# ================================================================
#  Summary
# ================================================================
print("\n" + "="*60)
print("Key Findings:")
print("="*60)

best_idx = performance_df["F1"].idxmax()
best_measure = performance_df.iloc[best_idx]

print(f"\n🏆 Best Performing Measure: {best_measure['Measure']}")
print(f"  F1 Score : {best_measure['F1']:.4f}")
print(f"  Accuracy : {best_measure['Accuracy']:.4f}")
print(f"  Precision: {best_measure['Precision']:.4f}")
print(f"  Recall   : {best_measure['Recall']:.4f}")
print(f"  ROC-AUC  : {best_measure['ROC-AUC']:.4f}")
print(f"  PR-AUC   : {best_measure['PR-AUC']:.4f}")
print("\n" + "="*60)

