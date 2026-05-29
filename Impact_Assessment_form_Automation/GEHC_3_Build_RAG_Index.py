
# build_default_na_index.py
# Build a clean Default-NA FAISS index + meta from Excel (structured meta with {q,j,phase})
import os, json, numpy as np, pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import faiss

ART_DIR = Path("artifacts"); ART_DIR.mkdir(exist_ok=True, parents=True)
OUT_PREFIX = ART_DIR / "default_na_index"  # -> .faiss + .meta.json
SRC_XLSX = r"Training\GEHC_IAF_Tracker_History_Cleaned_Default.xlsx"  # adjust if needed

# ---- Azure config ----
load_dotenv()
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
if not EMBED_DEPLOYMENT:
    raise RuntimeError("Missing AZURE_OPENAI_EMBED_DEPLOYMENT")

# ---- Load Default-NA from Excel ----
df = pd.read_excel(SRC_XLSX, engine="openpyxl")

def find_col(df, candidates):
    cand = {str(c).strip().lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in cand: return cand[k.lower()]
    return None

col_q = find_col(df, ["Impact_Analysis_Question", "Impact Analysis Question", "Question", "QA_Question"])
col_j = find_col(df, ["Justification_or_Implementation_Details", "Justification or Implementation Details", "Justification", "J"])

if not col_q or not col_j:
    raise ValueError("Could not find Impact_Analysis_Question and/or Justification_or_Implementation_Details columns in Default-NA Excel.")

rows = (
    df[[col_q, col_j]]
    .dropna(subset=[col_q, col_j])
    .astype(str)
    .applymap(lambda s: s.strip())
)
rows = rows[rows[col_q].str.len() > 0]
rows = rows[rows[col_j].str.len() > 0]
rows = rows.drop_duplicates().reset_index(drop=True)

# ---- Prepare texts for embedding (question only) ----
texts = rows[col_q].tolist()

# ---- Embed in batches ----
vecs = []
B = 32
for i in range(0, len(texts), B):
    batch = texts[i:i+B]
    resp = client.embeddings.create(model=EMBED_DEPLOYMENT, input=batch)
    vecs.extend([d.embedding for d in resp.data])

X = np.array(vecs, dtype="float32")
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

# ---- Build FAISS (cosine via inner product on normalized vectors) ----
d = X.shape[1]
index = faiss.IndexFlatIP(d)
index.add(X)

# ---- Meta: structured dict (q + j + phase=NA) ----
meta = [
    {"q": rows.iloc[i][col_q], "j": rows.iloc[i][col_j], "phase": "NA"}
    for i in range(len(rows))
]

# ---- Write artifacts ----
faiss.write_index(index, str(OUT_PREFIX) + ".faiss")
Path(str(OUT_PREFIX) + ".meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
)

