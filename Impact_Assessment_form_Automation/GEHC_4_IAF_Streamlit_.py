# GEHC_POC_5_Streamlit_v2.py — Impact Assessment Studio
# Reference UI kept intact (title + response bubble metrics/layout).
# Revised v2 logic preserved (Default-NA gate 0.94/0.00/0.82, RAG per release, no validation at inference).
# New: Pred vs Actual comparison (QNo alignment, NA normalization, per-label accuracy, Save matrix).
# Diagnostics: shows matched NA Q/Justification & Release Top-1 Context snippet (read-only).
# Buttons use use_container_width=True (no 'width=' usage).
# -----------------------------------------------------------------------------------------------

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import faiss

# ---- Project preprocessing helpers -------------------------------------------------------------
from GEHC_1_Preprocessing import (
    aggregate_release_context,
    enrich_release_context,
    expand_abbreviations,
    semantic_enrichment,
)

# =========================
# Paths & Constants
# =========================
ARTIFACT_DIR = Path("artifacts"); ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_NA_INDEX_PREFIX = ARTIFACT_DIR / "default_na_index"  # .faiss + .meta.json

# --- Re-calibrated NA gate thresholds (from v2) ---
NA_SIM_THRESHOLD = 0.94
REL_SIM_CEILING_FOR_NA = 0.82
NA_MARGIN = 0.00

TOP_K = 6
MAX_CONTEXT_CHARS = 12_000
MAX_JUST_CHARS = 1000

POSITIVE = {"YES", "Y", "TRUE", "IMPACT", "AFFECTED"}
NEGATIVE = {"NO", "N", "FALSE"}

# =========================
# Streamlit Page Setup + CSS  (UI from reference file)
# =========================
st.set_page_config(
    page_title="GE Healthcare · Impact Assessment Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

def inject_css():
    css_path = Path("assets/style.css")
    if not css_path.exists():
        css_path = Path("style.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8", errors="ignore")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.sidebar.warning("Custom stylesheet not found at assets/style.css or ./style.css. Using default theme.")

inject_css()

# Header (exactly as reference)
st.markdown(
    """
<div class="app-header">
  <div class="app-title">⚡ GE Healthcare — Impact Assessment Studio</div>
  <div class="app-subtitle">Chat · Release Intelligence · Similarity Matching</div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================
# Azure OpenAI Clients
# =========================
@st.cache_resource(show_spinner=False)
def get_azure_clients():
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")     # resource base URL
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    # Deployments (embedding + fine-tuned chat)
    embed_depl = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    chat_depl = os.getenv("AZURE_OPENAI_FT_DEPLOYMENT")  # optional

    # (Optional) distinct API version for embeddings
    embed_api_version = os.getenv("AZURE_OPENAI_EMBED_API_VERSION") or api_version

    missing = [k for k, v in {
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_VERSION": api_version,
        "AZURE_OPENAI_EMBED_DEPLOYMENT": embed_depl,
    }.items() if not v]
    if missing:
        st.sidebar.error(f"Missing environment variables: {', '.join(missing)}")
        st.stop()

    # Primary client (chat/FT)
    chat_client = AzureOpenAI(
        api_key=api_key, api_version=api_version, azure_endpoint=endpoint,
    )
    # Embeddings client
    if embed_api_version == api_version:
        embed_client = chat_client
    else:
        embed_client = AzureOpenAI(
            api_key=api_key, api_version=embed_api_version, azure_endpoint=endpoint,
        )
    return chat_client, embed_client, embed_depl, (chat_depl or "")

chat_client, embed_client, EMBED_DEPLOYMENT, CHAT_DEPLOYMENT = get_azure_clients()

# =========================
# Sidebar: Upload + Status (Overwrite logs every run)
# =========================
# Always reset the sidebar logs for the current run (no accumulation)
st.session_state["_sb_logs"] = []

def sb_log(message: str, level: str = "info"):
    st.session_state["_sb_logs"].append((level, message))

with st.sidebar:
    st.markdown("### 📄 Upload Release File(s)")
    uploaded_files = st.file_uploader(
        "Upload .xlsx/.xls/.csv",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )
    st.markdown("#### Status")
    sb_log_container = st.container()

def _load_any_table(uploaded):
    name = str(getattr(uploaded, "name", "")).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded, engine="openpyxl")
    return pd.read_csv(uploaded)

release_dfs: List[pd.DataFrame] = []
if uploaded_files:
    for f in uploaded_files:
        try:
            df = _load_any_table(f)
            df["__SourceFile__"] = getattr(f, "name", "uploaded")
            release_dfs.append(df)
            sb_log(f"Uploaded: {getattr(f,'name','uploaded')} ({len(df)} rows)", "success")
        except Exception as e:
            sb_log(f"Failed to read {getattr(f,'name','uploaded')}: {e}", "error")

with sb_log_container:
    for lvl, msg in st.session_state["_sb_logs"]:
        st.markdown(f"<div class='sb-log {lvl}'>{msg}</div>", unsafe_allow_html=True)

if not release_dfs:
    st.info("Upload one or more NEW release files to begin.")
    st.stop()

# =========================
# STEP 1: Preprocess (aggregate + enrich)
# =========================
with st.spinner("Processing release data…"):
    raw_release = pd.concat(release_dfs, ignore_index=True)
    if "Release_Number" not in raw_release.columns:
        st.error("Column 'Release_Number' not found in uploaded file(s).")
        st.stop()
    agg = aggregate_release_context(raw_release)
    # No external glossary required; functions handle defaults
    enriched = enrich_release_context(agg, {})
    sb_log(f"Preprocessed {enriched.shape[0]} rows.", "success")

# =========================
# STEP 2: Select release & Build in-memory RAG
# =========================
release_options = sorted(raw_release["Release_Number"].dropna().astype(str).unique())
selected_release = st.selectbox("Target Release", release_options, help="Used for retrieval and scope matching.")

# --- Build release context (enriched)
CONTEXT_FIELDS_BY_PRIORITY = [
    "Scope_Feature_Defect_Summary",
    "Design_Impact",
    "Risk_Management_Impact",
    "Regulatory_Impact",
    "Production_Process_Impact",
    "Training_Impact",
    "Conclusion",
    "Comments_Notes",
    "Description_from_SPR",
    "Additional_Info",
]

def join_release_fields(row: pd.Series) -> str:
    parts = []
    for c in CONTEXT_FIELDS_BY_PRIORITY:
        if c in row and str(row[c]).strip():
            parts.append(f"{c.replace('_',' ')}: {str(row[c]).strip()}")
    return "\n".join(parts)

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i: i + size] for i in range(0, len(text), step)]

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    batch = [str(t).strip() for t in texts if isinstance(t, str) and t.strip()]
    if not batch:
        raise ValueError("No valid text inputs for embedding.")
    vecs = []
    B = 16
    for i in range(0, len(batch), B):
        sub = batch[i:i+B]
        resp = embed_client.embeddings.create(model=EMBED_DEPLOYMENT, input=sub)
        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype=np.float32)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

# Build contexts for selected release only
rel_rows = enriched[enriched["Release_Number"].astype(str) == str(selected_release)]
contexts: List[str] = []
for _, r in rel_rows.iterrows():
    combined = join_release_fields(r)
    if not combined.strip():
        combined = r.get("Expanded_Context") or r.get("Combined_Context") or ""
    contexts.extend(chunk_text(combined))
rel_vecs = embed_texts(contexts)
sb_log(f"Built {len(contexts)} context chunks for release {selected_release}.", "success")

# Build scope summaries for YES case rendering
scopes = (
    raw_release[raw_release["Release_Number"].astype(str) == str(selected_release)]
    .get("Scope_Feature_Defect_Summary", pd.Series(dtype=str))
    .fillna("").astype(str).tolist()
)
scopes_enr = [semantic_enrichment(expand_abbreviations(s, {})) for s in scopes] if scopes else []
scope_vecs = embed_texts(scopes_enr) if scopes_enr else np.zeros((0, 1536), dtype=np.float32)

# =========================
# STEP 3: Load Default‑NA FAISS
# =========================
def extract_na_justification(meta_item: Any) -> str:
    """Extract 'Justification' text from NA meta; supports dict or 'Q:... J:...' strings."""
    if isinstance(meta_item, dict):
        for k in ["J", "Justification", "Justification_or_Implementation_Details", "justification", "j"]:
            if k in meta_item and str(meta_item[k]).strip():
                return str(meta_item[k]).strip()
    s = str(meta_item or "")
    import re
    m = re.search(r"(?is)\bJ(?:ustification)?\s*:\s*(.*)$", s)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?is)Justification[_\s]*or[_\s]*Implementation[_\s]*Details\s*:\s*(.*)$", s)
    if m:
        return m.group(1).strip()
    if s.strip().startswith("Q:") and "?" in s:
        tail = s.split("?", 1)[-1].strip()
        if tail:
            return tail
    return s.strip()

def extract_na_question(meta_item: Any) -> str:
    """
    Extract the 'question' part from a Default‑NA meta item.
    Supports:
      • dict meta with keys: Impact_Analysis_Question / Question / Q / question
      • string meta formatted as 'Q: ... J: ...'
    """
    if isinstance(meta_item, dict):
        for k in ["Impact_Analysis_Question", "Question", "Q", "question", "Impact Analysis Question"]:
            if k in meta_item and str(meta_item[k]).strip():
                return str(meta_item[k]).strip()

    s = str(meta_item or "")
    import re
    m = re.search(r"(?is)^\s*Q\s*:\s*(.*?)(?:\s+J(?:ustification)?\s*:|$)", s)
    if m:
        return m.group(1).strip()
    return ""

class NAIndex:
    def __init__(self, prefix: Path):
        faiss_path = str(prefix) + ".faiss"
        meta_path = str(prefix) + ".meta.json"
        if not (Path(faiss_path).exists() and Path(meta_path).exists()):
            raise FileNotFoundError(f"Default‑NA index missing at {faiss_path} / {meta_path}")
        self.index = faiss.read_index(faiss_path)
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

    def search(self, qvec: np.ndarray, k: int = 1) -> List[Tuple[float, Any]]:
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        scores, idxs = self.index.search(qvec.astype(np.float32), k)
        out: List[Tuple[float, Any]] = []
        for s, i in zip(scores[0], idxs[0]):
            if 0 <= i < len(self.meta):
                out.append((float(s), self.meta[i]))
        return out

try:
    na_index = NAIndex(DEFAULT_NA_INDEX_PREFIX)
    sb_log("Default‑NA index ready.", "success")
except Exception as e:
    st.error(f"Default‑NA index not available: {e}")
    st.stop()

# =========================
# Decision Logic (LLM JSON)
# =========================
def _normalize_label(label: str) -> str:
    return (label or "").strip().upper()

def is_positive(label: str) -> bool:
    return _normalize_label(label) in POSITIVE

def is_negative(label: str) -> bool:
    return _normalize_label(label) in NEGATIVE

def decide_with_llm(question: str, topk_texts: List[str]) -> Dict[str, str]:
    # Simple guardrail
    if any(k in question.lower() for k in ["security", "xss", "csrf", "encryption", "tls"]):
        return {"response":"YES","phase":"PRIOR TO ECO RELEASE",
                "justification":"Security-related keywords detected; treat as requiring attention.",
                "source":"RuleEngine"}

    ctx_join = "\n\n".join(topk_texts)[:MAX_CONTEXT_CHARS]
    hints = []
    signal_yes = ["will be updated", "will be performed", "verification", "validation", "report will be", "SAST", "DAST"]
    signal_no  = ["no change", "not applicable", "non-medical", "saaS based cloud", "no impact"]
    if any(s in ctx_join.lower() for s in signal_yes): hints.append("HINT_YES")
    if any(s in ctx_join.lower() for s in signal_no):  hints.append("HINT_NO")

    msgs = [
        {
            "role": "system",
            "content": (
                "You perform impact assessment. Use ONLY the provided context. "
                "Return JSON with keys: response(YES/NO/NA), justification, phase. "
                "If context suggests updates/tests/docs to be done, prefer YES and phase 'PRIOR TO ECO RELEASE'. "
                "If context clearly states 'no change' or 'not applicable', prefer NO or NA accordingly."
            ),
        },
        {"role": "user", "content": f"Question: {question}\nHints: {', '.join(hints) if hints else 'NONE'}\nContext: {ctx_join}"},
    ]
    try:
        resp = chat_client.chat.completions.create(
            model=CHAT_DEPLOYMENT, messages=msgs, temperature=0.0, max_tokens=300,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "response": str(data.get("response", "NA")).strip().upper() or "NA",
            "phase": str(data.get("phase", "NA")).strip() or "NA",
            "justification": str(data.get("justification", ""))[:MAX_JUST_CHARS],
            "source": ("FineTuned" if os.getenv("AZURE_OPENAI_FT_DEPLOYMENT") else "BaseChat"),
        }
    except Exception as e:
        return {"response":"NA","phase":"NA","justification":f"Model fallback. Reason: {e}","source":"Fallback"}

# =========================
# Response Bubble (reference UI layout)
# =========================
def render_response_bubble(rb: Dict[str, Any]):
    response = _normalize_label(rb.get("Response") or "NA")
    is_yes = is_positive(response)
    is_no = is_negative(response)
    pill_class = "pill-yes" if is_yes else ("pill-no" if is_no else "pill-na")
    conf = float(rb.get("Confidence") or 0.0)
    conf_pct = int(round(conf * 100))
    phase = rb.get("Phase", "NA")
    source = rb.get("Source", "-")
    justification = rb.get("Justification", "-")

    # Bubble UI (unchanged)
    st.markdown(
        f"""
<div class="bubble">
  <div class="meta">
    <span class="pill {pill_class}">{response or 'NA'}</span>
  </div>
  <div class="hr"></div>

  <div class="meta">Justification</div>
  <div class="justif">{justification}</div>
  <div class="hr"></div>

  <div class="meta">Phase: <b>{phase}</b></div>

  <div class="meta" style="margin-top:.35rem;">
    <span>Source: <b>{source}</b></span>
  </div>

  <div class="meta">
    <span>Confidence Score: <b>{conf:.3f}</b></span>
  </div>

  <div class="hr"></div>

  <div class="meter"><span style="width:{conf_pct}%;"></span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    # YES: show relevant scope; NO/NA: show standard message
    if is_yes:
        scope = rb.get("Best_Scope_Summary", "")
        score = rb.get("Scope_Similarity_Score", "")
        if scope:
            st.markdown(
                f"""
<div class="story">
  <div class="label">Relevant Scope/Feature/Defect</div>
  {scope}
  <div class="label" style="margin-top:.35rem;">Similarity: <b>{score}</b></div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="story">
  <div class="label">Relevant Scope/Feature/Defect</div>
  No relevant items were found in the provided release for this question.
</div>
""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
<div class="story">
  <div class="label">Relevant Scope/Feature/Defect</div>
  No relevant items were found in the provided release for this question.
</div>
""",
            unsafe_allow_html=True,
        )

#     # Diagnostics (labels adapted to v2) + Matched references
#     with st.expander("Diagnostics", expanded=False):
#         # Original metrics retained exactly
#         st.markdown(
#             f"""
# - **Default‑NA Similarity**: {rb.get('NA_Similarity','-')}
# - **Top Release Similarity**: {rb.get('Release_Top_Similarity','-')}
# - **Release**: {rb.get('Release','-')}
# - **Top‑K Contexts Used**: {rb.get('TopK','-')}
# """,
#             unsafe_allow_html=True,
#         )

#         # --- NEW: Matched references (for traceability only)
#         na_q = rb.get("Diag_NA_Matched_Question") or ""
#         na_j = rb.get("Diag_NA_Matched_Justification") or ""
#         rel_ctx = rb.get("Diag_Release_Top_Context") or ""

#         if na_q or na_j or rel_ctx:
#             st.markdown("**Matched references (for traceability)**")
#             if na_q or na_j:
#                 st.markdown("- **Default‑NA → Matched Question**")
#                 st.write(na_q if na_q else "–")
#                 st.markdown("- **Default‑NA → Matched Justification**")
#                 st.write(na_j if na_j else "–")

#             if rel_ctx:
#                 st.markdown("- **Release Top‑1 Context (snippet)**")
#                 st.write(rel_ctx)

# =========================
# Utilities: label & phase normalization (for comparison)
# =========================
YES_SET = {"YES","Y","TRUE","IMPACT","AFFECTED"}
NO_SET  = {"NO","N","FALSE"}
NA_ALIASES = {"NA","N/A","NOT APPLICABLE","NOT_APPLICABLE","NONE","NAN","NULL","NOT-APPLICABLE","-",""}

def norm_label(x: Any) -> str:
    s = str(x).strip().upper()
    if s in YES_SET: return "YES"
    if s in NO_SET:  return "NO"
    return "NA" if s in NA_ALIASES or s == "" else "NA"

def norm_phase(x: Any) -> str:
    s = str(x).strip()
    return "NA" if s == "" or s.lower() in {"nan","none"} else s.upper()

def norm_qno(x: Any) -> str:
    if pd.isna(x): return ""
    s = str(x).strip()
    if s in ("", "nan", "None", "NONE"): return ""
    if s == "--": return s
    try:
        f = float(s)
        return ("%f" % f).rstrip("0").rstrip(".")
    except Exception:
        try:
            if s.replace(".", "", 1).isdigit():
                f = float(s)
                return ("%f" % f).rstrip("0").rstrip(".")
        except Exception:
            pass
        if "99999" in s or "00000" in s:
            try:
                f = float(s)
                return ("%f" % f).rstrip("0").rstrip(".")
            except Exception:
                return s
        return s

# =========================
# Tabs: Chat · Batch(147) · Compare
# =========================
chat_tab, batch_tab, compare_tab = st.tabs(["💬 Chat", "📦 Batch (147)", "📊 Compare"])

# -------- Chat ----------
with chat_tab:
    st.markdown("### 💬 Chat")
    prompt = st.chat_input("Ask an impact assessment question…")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                # Enrich + embed question
                q_enr = semantic_enrichment(expand_abbreviations(prompt, {}))
                q_vec = embed_texts([q_enr])[0]

                # Similarities
                na_hits = na_index.search(q_vec, k=1)
                na_score = na_hits[0][0] if na_hits else 0.0

                sims = rel_vecs @ q_vec if len(rel_vecs) else np.array([])
                order = np.argsort(-sims)[:TOP_K] if sims.size else np.array([], dtype=int)
                topk = [(float(sims[j]), contexts[j]) for j in order] if sims.size else []
                rel_score = float(sims[order[0]]) if len(order) else 0.0

                # --- Diagnostics data capture (for reference only)
                na_match_q = ""
                na_match_j = ""
                if na_hits:
                    _meta = na_hits[0][1]
                    na_match_q = extract_na_question(_meta)
                    na_match_j = extract_na_justification(_meta)

                release_top_context = ""
                if len(order):
                    # Exact chunk that yielded the Top Release Similarity
                    release_top_context = str(contexts[order[0]]).strip()
                    # keep the UI compact
                    if len(release_top_context) > 600:
                        release_top_context = release_top_context[:600] + " ..."

                # Re-calibrated NA gate
                use_na = (
                    (na_score >= NA_SIM_THRESHOLD)
                    and ((na_score - rel_score) >= NA_MARGIN)
                    and (rel_score < REL_SIM_CEILING_FOR_NA)
                )

                if use_na and na_hits:
                    meta_item = na_hits[0][1]
                    just = extract_na_justification(meta_item)
                    res = {"response":"NA","phase":"NA","justification":just,"source":"Default NA Index"}
                    best_scope = ""; scope_sim = ""; conf = max(0.0, min(1.0, na_score))
                else:
                    res = decide_with_llm(prompt, [t for _, t in topk])
                    best_scope, scope_sim = "", ""
                    if is_positive(res.get("response")) and len(scope_vecs) > 0:
                        sc_sims = scope_vecs @ q_vec
                        j = int(np.argmax(sc_sims))
                        best_scope = scopes[j] if len(scopes) > 0 else ""
                        scope_sim = f"{float(sc_sims[j]):.3f}" if sc_sims.size else ""
                    conf = (max(0.0, min(1.0, 0.5 + 0.5 * (rel_score - 0.5) / 0.5)) if len(topk) else 0.5)

                rb = {
                    "Release": str(selected_release),
                    "Question": prompt,
                    "Response": res.get("response","NA"),
                    "Justification": res.get("justification",""),
                    "Phase": res.get("phase","NA"),
                    "Confidence": round(float(conf), 3),
                    "Source": res.get("source",""),

                    # existing fields
                    "Best_Scope_Summary": (best_scope if is_positive(res.get("response")) else
                        "No relevant items were found in the provided release for this question."),
                    "Scope_Similarity_Score": scope_sim if is_positive(res.get("response")) else "",
                    "NA_Similarity": round(float(na_score), 3),
                    "Release_Top_Similarity": round(float(rel_score), 3),
                    "TopK": len(topk),

                    # NEW: diagnostics data
                    "Diag_NA_Matched_Question": na_match_q,
                    "Diag_NA_Matched_Justification": na_match_j,
                    "Diag_Release_Top_Context": release_top_context,
                }
                render_response_bubble(rb)

# -------- Batch ----------
with batch_tab:
    st.markdown("### Run 147 Questions (Batch)")
    col1, col2 = st.columns(2)
    with col1:
        tmpl = st.file_uploader("Upload 147-question template (CSV/XLSX)",
                                type=["csv","xlsx","xls"], key="tmpl")
    with col2:
        gt = st.file_uploader("(Optional) Upload Ground-Truth IAF (CSV/XLSX) for OFFLINE evaluation",
                              type=["csv","xlsx","xls"], key="gt")

    # (reverted) use_container_width=True
    run = st.button("Run Batch", type="primary", use_container_width=True)

    def _load_template(upload, strict_147=True) -> pd.DataFrame:
        if not upload:
            raise ValueError("Template file is required.")
        df = _load_any_table(upload)
        cols = {c.lower().strip(): c for c in df.columns}
        qno = cols.get("question_no") or cols.get("question number") or cols.get("q no") or cols.get("q_no") or cols.get("qid") or "Question_No"
        qtx = cols.get("impact_analysis_question") or cols.get("impact analysis question") or cols.get("question") or cols.get("question text") or "Impact_Analysis_Question"
        if qtx not in df.columns:
            raise ValueError("Template must have an Impact Analysis Question column (or alias).")
        out = pd.DataFrame({
            "Question_No": (df[qno] if qno in df.columns else np.arange(1, len(df)+1)),
            "Impact_Analysis_Question": df[qtx].astype(str),
        })
        out["Question_No"] = out["Question_No"].astype(str).str.strip()
        out["Impact_Analysis_Question"] = out["Impact_Analysis_Question"].astype(str).str.strip()
        out = out[out["Impact_Analysis_Question"].str.len() > 0]
        out = out.drop_duplicates(subset=["Question_No","Impact_Analysis_Question"]).reset_index(drop=True)
        if strict_147 and len(out) != 147:
            raise ValueError(f"Expected 147 questions; found {len(out)}.")
        return out

    def _eval_offline_qno(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
        """Strict `Question_No` alignment with key cleanup, NA normalization, per-label accuracy."""
        # Identify and rename GT columns
        cols = {c.lower(): c for c in gt_df.columns}
        qcol = cols.get("question_no") or cols.get("question number") or cols.get("q no") or cols.get("q_no")
        rcol = cols.get("response")
        pcol = cols.get("target_completion_workflow_phase") or cols.get("phase")
        if not qcol or not rcol:
            raise ValueError("Ground truth needs columns: Question_No and Response.")

        gt = gt_df.rename(columns={qcol:"Question_No", rcol:"GT_Response"})
        if pcol and pcol in gt.columns:
            gt = gt.rename(columns={pcol:"GT_Phase"})
        else:
            gt["GT_Phase"] = "NA"

        # Normalize keys and labels
        pred = pred_df.copy()
        pred["KEY"] = pred["Question_No"].map(norm_qno)
        gt["KEY"]   = gt["Question_No"].map(norm_qno)

        pred = pred[pred["KEY"] != ""].copy()
        gt   = gt[gt["KEY"]   != ""].copy()

        # Inner join and de-dup per KEY
        m = pd.merge(
            pred[["KEY","Question_No","Impact_Analysis_Question","Response","Phase","Confidence","Source"]],
            gt[  ["KEY","Question_No","GT_Response","GT_Phase"]],
            on="KEY", how="inner", suffixes=("_Pred","_GT")
        ).sort_values(by="KEY").groupby("KEY", as_index=False).first()

        # Normalize labels and phase
        m["Pred_Response_N"] = m["Response"].map(norm_label)
        m["GT_Response_N"]   = m["GT_Response"].map(norm_label)
        m["Pred_Phase_N"]    = m["Phase"].map(norm_phase)
        m["GT_Phase_N"]      = m["GT_Phase"].map(norm_phase)

        labels = ["YES","NO","NA"]
        cm = pd.crosstab(m["GT_Response_N"], m["Pred_Response_N"]).reindex(index=labels, columns=labels, fill_value=0)
        overall_acc = float((m["GT_Response_N"] == m["Pred_Response_N"]).mean()) if len(m) else 0.0

        # Per-class accuracy (recall): TP / GT count per class
        per_label = []
        for lab in labels:
            tp = int(cm.at[lab, lab])
            gt_count = int(cm.loc[lab].sum())
            acc_lab = (tp / gt_count) if gt_count > 0 else 0.0
            per_label.append({"Label": lab, "GT_Count": gt_count, "TP": tp, "Accuracy": round(acc_lab,4)})

        # Distributions
        pred_dist = m["Pred_Response_N"].value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
        pred_dist.columns = ["Label","Pred_Count"]
        gt_dist   = m["GT_Response_N"].value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
        gt_dist.columns = ["Label","GT_Count"]

        # Phase accuracy on GT=YES
        yes_mask = (m["GT_Response_N"] == "YES")
        phase_yes_acc = float((m.loc[yes_mask, "Pred_Phase_N"] == m.loc[yes_mask, "GT_Phase_N"]).mean()) if yes_mask.any() else 0.0

        # Return payload
        return {
            "aligned": m,
            "cm": cm,
            "overall_acc": round(overall_acc,4),
            "per_label": pd.DataFrame(per_label),
            "pred_dist": pred_dist,
            "gt_dist": gt_dist,
            "phase_yes_acc": round(phase_yes_acc,4),
        }

    if run:
        try:
            tmpl_df = _load_template(tmpl, strict_147=True)
        except Exception as e:
            st.error(f"Template error: {e}")
            st.stop()

        st.write("Running batch prediction… (uses selected release + Default‑NA gate + the Fine‑tuned LLM)")
        recs: List[Dict[str, Any]] = []

        for _, row in tmpl_df.iterrows():
            qno = str(row["Question_No"]).strip()
            qtx = str(row["Impact_Analysis_Question"]).strip()

            q_enr = semantic_enrichment(expand_abbreviations(qtx, {}))
            q_vec = embed_texts([q_enr])[0]

            na_hits = na_index.search(q_vec, k=1)
            na_score = na_hits[0][0] if na_hits else 0.0

            sims = rel_vecs @ q_vec if len(rel_vecs) else np.array([])
            order = np.argsort(-sims)[:TOP_K] if sims.size else np.array([], dtype=int)
            topk = [(float(sims[j]), contexts[j]) for j in order] if sims.size else []
            rel_score = float(sims[order[0]]) if len(order) else 0.0

            use_na = (
                (na_score >= NA_SIM_THRESHOLD)
                and ((na_score - rel_score) >= NA_MARGIN)
                and (rel_score < REL_SIM_CEILING_FOR_NA)
            )

            if use_na and na_hits:
                meta_item = na_hits[0][1]
                just = extract_na_justification(meta_item)
                res = {"response":"NA","phase":"NA","justification": just,"source":"Default NA Index"}
                best_scope = ""; scope_sim = ""; conf = max(0.0, min(1.0, na_score))
            else:
                res = decide_with_llm(qtx, [t for _, t in topk])
                best_scope, scope_sim = "", ""
                if is_positive(res.get("response")) and len(scope_vecs) > 0:
                    sc_sims = scope_vecs @ q_vec
                    j = int(np.argmax(sc_sims))
                    best_scope = scopes[j] if len(scopes) > 0 else ""
                    scope_sim = f"{float(sc_sims[j]):.3f}" if sc_sims.size else ""
                conf = (max(0.0, min(1.0, 0.5 + 0.5 * (rel_score - 0.5) / 0.5)) if len(topk) else 0.5)

            recs.append({
                "Release_Number": str(selected_release),
                "Question_No": qno,
                "Impact_Analysis_Question": qtx,
                "Response": res.get("response","NA"),
                "Phase": res.get("phase","NA"),
                "Justification": str(res.get("justification",""))[:MAX_JUST_CHARS],
                "Confidence": round(float(conf), 3),
                "Source": res.get("source",""),
                "Best_Scope_Summary": (best_scope if is_positive(res.get("response")) else
                    "No relevant items were found in the provided release for this question."),
                "Scope_Similarity_Score": scope_sim if is_positive(res.get("response")) else "",
            })

        pred_df = pd.DataFrame(recs)
        st.success(f"Completed. Generated {len(pred_df)} predictions.")

        # Summary
        dist = pred_df["Response"].fillna("NA").str.upper().map(norm_label).value_counts(dropna=False)
        st.write("**Summary (Predicted):**", {
            "release": selected_release,
            "total": len(pred_df),
            "YES": int(dist.get("YES",0)),
            "NO": int(dist.get("NO",0)),
            "NA": int(dist.get("NA",0)),
            "avg_confidence": round(float(pred_df["Confidence"].mean() if len(pred_df) else 0.0), 4),
        })

        # Optional OFFLINE evaluation with GT (and Save Matrix button)
        if gt is not None:
            try:
                gt_df = _load_any_table(gt)
                comparison_payload = _eval_offline_qno(pred_df, gt_df)
                cm = comparison_payload["cm"]
                per_label_df = comparison_payload["per_label"]
                pred_dist_df = comparison_payload["pred_dist"]
                gt_dist_df   = comparison_payload["gt_dist"]

                st.markdown("#### Pred vs Actual — Confusion Matrix (GT rows aligned by Question_No)")
                st.dataframe(cm, use_container_width=True)

                # Show label distributions and per-label accuracy
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Actual (GT) Label Counts**")
                    st.dataframe(gt_dist_df, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("**Predicted Label Counts**")
                    st.dataframe(pred_dist_df, use_container_width=True, hide_index=True)
                with c3:
                    st.markdown("**Per-Label Accuracy (TP/GT)**")
                    st.dataframe(per_label_df, use_container_width=True, hide_index=True)

                st.info({
                    "overall_response_accuracy": comparison_payload["overall_acc"],
                    "phase_accuracy_on_GT_YES": comparison_payload["phase_yes_acc"]
                })

                # Prepare Excel report for download
                ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                comp_name = f"comparison_{selected_release}_{ts}.xlsx"
                xls_path = ARTIFACT_DIR / comp_name
                with pd.ExcelWriter(xls_path, engine="openpyxl") as w:
                    cm.to_excel(w, sheet_name="confusion_matrix")
                    comparison_payload["aligned"].to_excel(w, sheet_name="aligned_rows", index=False)
                    comparison_payload["per_label"].to_excel(w, sheet_name="per_label_accuracy", index=False)
                    comparison_payload["pred_dist"].to_excel(w, sheet_name="pred_distribution", index=False)
                    comparison_payload["gt_dist"].to_excel(w, sheet_name="gt_distribution", index=False)

                # (reverted) use_container_width=True
                st.download_button(
                    "💾 Save matrix (Excel)",
                    data=xls_path.read_bytes(),
                    file_name=comp_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption(f"Saved: {xls_path}")
            except Exception as e:
                st.warning(f"Ground-truth evaluation skipped: {e}")

        # Save + Download predictions CSV
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = ARTIFACT_DIR / f"predicted_iaf_{selected_release}_{ts}.csv"
        pred_df.to_csv(out_csv, index=False)
        st.download_button(
            "Download Predictions (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predicted_iaf_{selected_release}.csv",
            mime="text/csv",
            use_container_width=True,   # reverted
        )
        st.caption(f"Saved: {out_csv}")

        # Preview
        st.dataframe(pred_df.head(25), use_container_width=True)

# -------- Compare (standalone utility) ----------
with compare_tab:
    st.markdown("### Pred vs Actual — Quick Compare")
    c1, c2 = st.columns(2)
    with c1:
        pred_file = st.file_uploader("Predicted file (CSV/XLSX)", type=["csv","xlsx","xls"], key="pred_file")
    with c2:
        act_file  = st.file_uploader("Actual (Ground Truth) file (CSV/XLSX)", type=["csv","xlsx","xls"], key="act_file")

    # (reverted) use_container_width=True
    do_cmp = st.button("Run Comparison", type="primary", use_container_width=True)

    def load_any(path_or_buffer):
        name = getattr(path_or_buffer, "name", "")
        if str(name).lower().endswith((".xlsx",".xls")):
            return pd.read_excel(path_or_buffer, engine="openpyxl")
        return pd.read_csv(path_or_buffer)

    if do_cmp:
        try:
            pred_df = load_any(pred_file)
            gt_df = load_any(act_file)

            # Expect columns: Question_No, Response, Phase (pred); Question_No, Response, Phase (gt optional)
            cols = {c.lower(): c for c in gt_df.columns}
            qcol = cols.get("question_no") or cols.get("q no") or cols.get("question number") or cols.get("q_no")
            rcol = cols.get("response")
            pcol = cols.get("target_completion_workflow_phase") or cols.get("phase")
            if not qcol or not rcol:
                raise ValueError("GT file needs columns: Question_No and Response (and Phase optional).")

            gt = gt_df.rename(columns={qcol:"Question_No", rcol:"GT_Response"})
            gt["GT_Phase"] = gt_df[pcol] if pcol in gt_df.columns else "NA"

            pred = pred_df.rename(columns={
                next((c for c in pred_df.columns if c.lower() == "question_no"), "Question_No"): "Question_No",
                next((c for c in pred_df.columns if c.lower() == "response"), "Response"): "Response",
                next((c for c in pred_df.columns if c.lower() == "phase"), "Phase"): "Phase",
                next((c for c in pred_df.columns if c.lower() == "confidence"), "Confidence"): "Confidence",
                next((c for c in pred_df.columns if c.lower() == "source"), "Source"): "Source",
            })

            # Clean keys & align
            pred["KEY"] = pred["Question_No"].map(norm_qno)
            gt["KEY"]   = gt["Question_No"].map(norm_qno)
            pred = pred[pred["KEY"] != ""].copy()
            gt   = gt[gt["KEY"]   != ""].copy()

            aligned = pd.merge(
                pred[["KEY","Question_No","Response","Phase","Confidence","Source","Impact_Analysis_Question"]],
                gt[  ["KEY","Question_No","GT_Response","GT_Phase"]],
                on="KEY", how="inner", suffixes=("_Pred","_GT")
            ).sort_values("KEY").groupby("KEY", as_index=False).first()

            # Normalize labels and phase
            aligned["Pred_Response_N"] = aligned["Response"].map(norm_label)
            aligned["GT_Response_N"]   = aligned["GT_Response"].map(norm_label)
            aligned["Pred_Phase_N"]    = aligned["Phase"].map(norm_phase)
            aligned["GT_Phase_N"]      = aligned["GT_Phase"].map(norm_phase)

            labels = ["YES","NO","NA"]
            cm = pd.crosstab(aligned["GT_Response_N"], aligned["Pred_Response_N"]).reindex(index=labels, columns=labels, fill_value=0)
            overall_acc = float((aligned["GT_Response_N"] == aligned["Pred_Response_N"]).mean()) if len(aligned) else 0.0
            yes_mask = (aligned["GT_Response_N"] == "YES")
            phase_yes_acc = float((aligned.loc[yes_mask, "Pred_Phase_N"] == aligned.loc[yes_mask, "GT_Phase_N"]).mean()) if yes_mask.any() else 0.0

            # Per-label accuracy and distributions
            per_label = []
            for lab in labels:
                tp = int(cm.at[lab, lab])
                gt_count = int(cm.loc[lab].sum())
                acc_lab = (tp / gt_count) if gt_count > 0 else 0.0
                per_label.append({"Label": lab, "GT_Count": gt_count, "TP": tp, "Accuracy": round(acc_lab,4)})
            per_label_df = pd.DataFrame(per_label)

            pred_dist = aligned["Pred_Response_N"].value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
            pred_dist.columns = ["Label","Pred_Count"]
            gt_dist = aligned["GT_Response_N"].value_counts(dropna=False).reindex(labels, fill_value=0).reset_index()
            gt_dist.columns = ["Label","GT_Count"]

            st.markdown("#### Confusion Matrix")
            st.dataframe(cm, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Actual (GT) Label Counts**")
                st.dataframe(gt_dist, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Predicted Label Counts**")
                st.dataframe(pred_dist, use_container_width=True, hide_index=True)
            with c3:
                st.markdown("**Per-Label Accuracy (TP/GT)**")
                st.dataframe(per_label_df, use_container_width=True, hide_index=True)

            st.info({
                "overall_response_accuracy": round(overall_acc,4),
                "phase_accuracy_on_GT_YES": round(phase_yes_acc,4)
            })

            # Save matrix — button (reverted to use_container_width)
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            xname = f"comparison_quick_{ts}.xlsx"
            xpath = ARTIFACT_DIR / xname
            with pd.ExcelWriter(xpath, engine="openpyxl") as w:
                cm.to_excel(w, sheet_name="confusion_matrix")
                aligned.to_excel(w, sheet_name="aligned_rows", index=False)
                per_label_df.to_excel(w, sheet_name="per_label_accuracy", index=False)
                pred_dist.to_excel(w, sheet_name="pred_distribution", index=False)
                gt_dist.to_excel(w, sheet_name="gt_distribution", index=False)

            st.download_button(
                "💾 Save matrix (Excel)",
                data=xpath.read_bytes(),
                file_name=xname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption(f"Saved: {xpath}")

        except Exception as e:
            st.error(f"Comparison failed: {e}")
