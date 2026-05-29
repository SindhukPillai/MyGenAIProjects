import os
import json
import pandas as pd

# Constants
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# File paths
RELEASE_PATH = os.path.join(ARTIFACT_DIR, "enriched_release.csv")
IAF_PATH = os.path.join(ARTIFACT_DIR, "enriched_iaf.csv")
DEFAULT_NA_PATH = os.path.join(ARTIFACT_DIR, "enriched_default_na_questions.json")
CLEANED_IAF_PATH = os.path.join(ARTIFACT_DIR, "cleaned_iaf_for_validation.csv")
OUTPUT_JSONL_PATH = os.path.join(ARTIFACT_DIR, "ft_training.jsonl")
INVALID_ROWS_PATH = os.path.join(ARTIFACT_DIR, "invalid_iaf_rows_with_reasons.csv")

# Load data
release_df = pd.read_csv(RELEASE_PATH)
iaf_df = pd.read_csv(IAF_PATH)

with open(DEFAULT_NA_PATH, "r", encoding="utf-8") as f:
    default_na_data = json.load(f)

# Extract NA question numbers from default NA tracker
default_na_question_nos = {
    str(item["Question_No"]).strip()
    for item in default_na_data
    if item.get("Response", "").strip().upper() == "NA"
}

# Normalize IAF columns
iaf_df["Response"] = iaf_df["Response"].astype(str).str.strip().str.upper()
iaf_df["Release_Number"] = iaf_df["Release_Number"].astype(str).str.strip()
iaf_df["Question_No"] = iaf_df["Question_No"].astype(str).str.strip()
iaf_df["Impact_Analysis_Question"] = iaf_df["Impact_Analysis_Question"].astype(str).str.strip()
iaf_df["Justification_or_Implementation_Details"] = iaf_df["Justification_or_Implementation_Details"].astype(str).str.strip()
iaf_df["Target_Completion_Workflow_Phase"] = iaf_df["Target_Completion_Workflow_Phase"].astype(str).str.strip()

# Filter out NA questions from IAF using Question_No
filtered_iaf_df = iaf_df[
    (~iaf_df["Question_No"].isin(default_na_question_nos)) &
    (iaf_df["Response"].isin({"YES", "NO"}))
]

# Save cleaned IAF for validation
filtered_iaf_df.to_csv(CLEANED_IAF_PATH, index=False)

# Validate required columns
required_release_cols = ["Release_Number", "Expanded_Context"]
required_iaf_cols = [
    "Release_Number", "Question_No", "Impact_Analysis_Question", "Response",
    "Justification_or_Implementation_Details", "Target_Completion_Workflow_Phase"
]

for col in required_release_cols:
    if col not in release_df.columns:
        raise ValueError(f"Missing column in release data: {col}")

for col in required_iaf_cols:
    if col not in iaf_df.columns:
        raise ValueError(f"Missing column in IAF data: {col}")

# Map release number to context
context_map = {
    str(row["Release_Number"]).strip(): row["Expanded_Context"]
    for _, row in release_df.iterrows()
}

# Prepare fine-tuning samples
RESPONSE_CHOICES = {"YES", "NO", "NA"}
PHASE_CHOICES = {
    "PRIOR TO ECR APPROVAL", "PRIOR TO ECO RELEASE", "PRIOR TO ERP RELEASE",
    "PRIOR TO ECO IMPLEMENTED", "NA"
}

samples = []
invalid_rows = []

for _, row in filtered_iaf_df.iterrows():
    release_number = row["Release_Number"]
    question = row["Impact_Analysis_Question"]
    response = row["Response"]
    justification = row["Justification_or_Implementation_Details"]
    phase = row["Target_Completion_Workflow_Phase"]

    reason = None
    if response not in RESPONSE_CHOICES:
        reason = "Invalid Response"
    elif not release_number:
        reason = "Missing Release_Number"
    elif release_number not in context_map:
        reason = "Unmatched Release_Number"
    elif not question:
        reason = "Empty Impact_Analysis_Question"
    elif not justification:
        reason = "Empty Justification_or_Implementation_Details"

    if reason:
        row_data = row.to_dict()
        row_data["Exclusion_Reason"] = reason
        invalid_rows.append(row_data)
        continue

    if phase not in PHASE_CHOICES:
        phase = "NA"

    context = context_map.get(release_number, "")
    context_short = context[:10000]
    justification_short = justification[:1000]

    system_msg = (
        "You are an assistant performing impact assessment for software releases. "
        "Use only the provided context to answer. Output a JSON with keys: response, justification, phase."
    )
    user_msg = f"Question: {question}\nContext: {context_short}"
    assistant_msg = json.dumps({
        "response": response,
        "justification": justification_short,
        "phase": phase
    }, ensure_ascii=False)

    sample = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }
    samples.append(sample)

# Save fine-tuning dataset
with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# Save invalid rows
invalid_df = pd.DataFrame(invalid_rows)
invalid_df.to_csv(INVALID_ROWS_PATH, index=False)

print(f"✅ Cleaned IAF saved at: {CLEANED_IAF_PATH}")
print(f"✅ Fine-tuning dataset prepared with {len(samples)} samples at: {OUTPUT_JSONL_PATH}")
print(f"✅ Invalid rows exported to: {INVALID_ROWS_PATH}")