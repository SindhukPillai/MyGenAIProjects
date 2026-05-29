import os
import pandas as pd
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create output directory
output_dir = "artifacts"
os.makedirs(output_dir, exist_ok=True)

# Utility: Load CSV or Excel
def load_file(file_path):
    print(f"📂 Loading file: {file_path}")
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

# Abbreviation map
def load_abbreviation_map(abbr_df):
    print("🔍 Creating abbreviation map...")
    abbr_map = {}
    for _, row in abbr_df.iterrows():
        sf = str(row["Short_Form"]).strip().lower()
        ef = str(row["Expanded_Form"]).strip()
        if sf and ef:
            abbr_map[sf] = ef
    print(f"✅ Loaded {len(abbr_map)} abbreviations.")
    return abbr_map

# Expand abbreviations
def expand_abbreviations(text, abbr_map):
    words = text.split()
    return " ".join([abbr_map.get(w.lower(), w) for w in words])

# Semantic enrichment
def semantic_enrichment(text):
    keywords = {
        "security": ["xss", "csrf", "encryption", "tls", "security"],
        "data": ["pii", "phi", "dataset", "schema", "gdpr"],
        "ui": ["ui", "ux", "accessibility"],
        "backend": ["api", "service", "database"],
        "compliance": ["sox", "fda", "audit"],
        "testing": ["unit test", "qa", "automation"],
        "performance": ["latency", "throughput", "scalability"],
        "risk": ["risk", "impact", "severity"],
        "release": ["deploy", "release", "rollback"]
    }
    tags = [tag for tag, kws in keywords.items() if any(k in text.lower() for k in kws)]
    if tags:
        text += f" [TAGS: {', '.join(sorted(set(tags)))}]"
    return text

# Aggregate release context
def aggregate_release_context(df):
    print("🧩 Aggregating release context...")
    required_cols = ["Release_Number", "Scope_Feature_Defect_Summary", "Design_Impact", "Description_from_SPR"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    agg_rows = []
    for rel, group in df.groupby("Release_Number"):
        sections = []
        for _, row in group.iterrows():
            parts = []
            if row["Scope_Feature_Defect_Summary"]:
                parts.append(f"Scope_Feature_Defect_Summary: {str(row['Scope_Feature_Defect_Summary']).strip()}")
            if row["Design_Impact"]:
                parts.append(f"Design_Impact: {str(row['Design_Impact']).strip()}")
            if row["Description_from_SPR"]:
                parts.append(f"Description_from_SPR: {str(row['Description_from_SPR']).strip()}")
            if parts:
                sections.append("\n".join(parts))
        combined = "\n".join(sections)
        agg_rows.append({"Release_Number": rel, "Combined_Context": combined})
    print("✅ Aggregation complete.")
    return pd.DataFrame(agg_rows)

# Enrich release context
def enrich_release_context(df, abbr_map):
    print("🔄 Enriching release context...")
    df["Expanded_Context"] = df["Combined_Context"].map(lambda s: semantic_enrichment(expand_abbreviations(s, abbr_map)))
    print(f"✅ Release context enrichment complete. Shape: {df.shape}")
    return df

# Enrich IAF data
def enrich_iaf_data(df, abbr_map):
    print("🧠 Enriching IAF data...")
    df["Impact_Analysis_Question"] = df["Impact_Analysis_Question"].astype(str).map(lambda s: expand_abbreviations(s, abbr_map))
    df["Justification_or_Implementation_Details"] = df["Justification_or_Implementation_Details"].astype(str).map(lambda s: expand_abbreviations(s, abbr_map))
    print(f"✅ IAF enrichment complete. Shape: {df.shape}")
    return df


# Enrich default NA questions
def enrich_na_questions(na_df, abbr_map):
    print("🧠 Enriching default NA questions...")
    na_df["Impact_Analysis_Question"] = na_df["Impact_Analysis_Question"].astype(str).map(lambda s: semantic_enrichment(expand_abbreviations(s, abbr_map)))
    na_df["Justification_or_Implementation_Details"] = na_df["Justification_or_Implementation_Details"].astype(str).map(lambda s: semantic_enrichment(expand_abbreviations(s, abbr_map)))
    print(f"✅ NA questions enrichment complete. Shape: {na_df.shape}")
    return na_df



# Main
def main():
    # Replace with your actual file paths
    release_file = "Training/GEHC_Release_Tracker_History_Cleaned.xlsx"
    iaf_file = "Training/GEHC_IAF_Tracker_History_Cleaned.xlsx"
    abbr_file = "Training/GEHC_IP5_Glossary.xlsx"
    na_questions_path = "Training/default_na_questions.json"

    release_df = load_file(release_file)
    iaf_df = load_file(iaf_file)
    abbr_df = load_file(abbr_file)
    na_df = pd.read_json(na_questions_path)

    abbr_map = load_abbreviation_map(abbr_df)
    aggregated_df = aggregate_release_context(release_df)
    enriched_release_df = enrich_release_context(aggregated_df, abbr_map)
    enriched_iaf_df = enrich_iaf_data(iaf_df, abbr_map)
    enriched_na_df = enrich_na_questions(na_df, abbr_map)

    enriched_release_df.to_csv(os.path.join(output_dir, "enriched_release.csv"), index=False)
    enriched_iaf_df.to_csv(os.path.join(output_dir, "enriched_iaf.csv"), index=False)    
    enriched_na_df.to_json(os.path.join(output_dir, "enriched_default_na_questions.json"), orient="records", indent=2)

    print("💾 Preprocessed files saved to 'artifacts/' directory.")

if __name__ == "__main__":
    main()
