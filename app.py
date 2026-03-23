"""
Debate is Cool — Monitoring & Insight System
Civic Space Nepal Pvt. Ltd.  |  v2.0 Final

This file is run by Streamlit via launcher.py.
Do NOT run this file directly — use launcher.py instead.
"""
# ── IMPORTS ───────────────────────────────────────────────────────────────────
import io, os, sys, json, textwrap, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Tuple, List
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Debate is Cool | CSN Monitoring",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');
* { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; }

[data-testid="stSidebar"] {
    background:linear-gradient(160deg,#0f0c29,#302b63,#24243e); color:white;
}
[data-testid="stSidebar"] * { color:white !important; }
[data-testid="stSidebar"] .stRadio label { font-size:.93rem; padding:3px 0; }

.main-header {
    background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
    padding:2.2rem 2rem; border-radius:14px; margin-bottom:1.8rem; color:white;
}
.main-header h1 { font-family:'Syne',sans-serif; font-size:2.5rem;
                  font-weight:800; margin:0; letter-spacing:-1px; }
.main-header p  { font-size:1rem; opacity:.75; margin:.35rem 0 0; font-weight:300; }

.metric-card { background:white; border:1px solid #e8e8f0; border-radius:12px;
               padding:1.1rem 1.3rem; box-shadow:0 2px 10px rgba(48,43,99,.07); margin-bottom:4px; }
.metric-card .label { font-size:.75rem; text-transform:uppercase;
                      letter-spacing:1px; color:#888; font-weight:500; }
.metric-card .value { font-family:'Syne',sans-serif; font-size:1.9rem;
                      font-weight:700; color:#302b63; }

.insight-box { background:linear-gradient(135deg,#667eea18,#764ba218);
               border-left:4px solid #667eea; border-radius:0 10px 10px 0;
               padding:.9rem 1.1rem; margin:.5rem 0; font-size:.9rem; }
.risk-box    { background:#fff3cd28; border-left:4px solid #ffc107;
               border-radius:0 10px 10px 0; padding:.9rem 1.1rem; margin:.5rem 0; font-size:.9rem; }
.rec-box     { background:#d4edda28; border-left:4px solid #28a745;
               border-radius:0 10px 10px 0; padding:.9rem 1.1rem; margin:.5rem 0; font-size:.9rem; }
.warn-box    { background:#f8d7da28; border-left:4px solid #dc3545;
               border-radius:0 10px 10px 0; padding:.9rem 1.1rem; margin:.5rem 0; font-size:.9rem; }

.stButton>button {
    background:linear-gradient(135deg,#302b63,#24243e); color:white; border:none;
    border-radius:8px; padding:.45rem 1.4rem;
    font-family:'Syne',sans-serif; font-weight:600; letter-spacing:.5px;
}
.stButton>button:hover { opacity:.9; }
div[data-testid="stDataFrame"] { border-radius:10px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
SCORE_COLS   = ["argument_clarity","reasoning_depth","refutation_quality","structure_strategy"]
TEACHER_COLS = [f"teacher_q{i}" for i in range(1,6)]
GROWTH_COLS  = [f"growth_q{i}"  for i in range(1,6)]
COLORS       = ["#667eea","#764ba2","#f093fb","#f5576c","#4facfe","#00f2fe","#43e97b","#38f9d7"]
CAT_LABELS   = {"argument_clarity":"Argument Clarity","reasoning_depth":"Reasoning Depth",
                "refutation_quality":"Refutation Quality","structure_strategy":"Structure & Strategy"}

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dic_models")
os.makedirs(_MODEL_DIR, exist_ok=True)
def _mpath(n): return os.path.join(_MODEL_DIR, f"{n}.joblib")

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL FILE STORAGE — saves data permanently inside project_folder/data/
# No MongoDB, no setup, no internet needed. Data stays until manually deleted.
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _fpath(name: str) -> str:
    """Return full path to a CSV file in the data folder."""
    return os.path.join(DATA_DIR, f"{name}.csv")

def db_connected() -> bool:
    """Always True — local file storage is always available."""
    return True

def save_data(name: str, df: pd.DataFrame):
    """
    Save dataframe to data/name.csv permanently.
    If file already exists, APPEND new rows and deduplicate.
    So uploading new sessions adds to existing data, never overwrites it.
    """
    path = _fpath(name)
    if os.path.exists(path):
        try:
            existing = pd.read_csv(path)
            combined = pd.concat([existing, df], ignore_index=True).drop_duplicates()
            combined.to_csv(path, index=False)
        except Exception:
            df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)

def overwrite_data(name: str, df: pd.DataFrame):
    """Overwrite data/name.csv completely (used for roster)."""
    df.to_csv(_fpath(name), index=False)

def load_data(name: str) -> Optional[pd.DataFrame]:
    """Load data/name.csv and return DataFrame, or None if not found."""
    path = _fpath(name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return None if df.empty else df
    except Exception:
        return None

def delete_data(name: str):
    """Delete a stored dataset."""
    path = _fpath(name)
    if os.path.exists(path):
        os.remove(path)

def fetch_all(col_name: str) -> list:
    """Compatibility shim — returns list of dicts like MongoDB did."""
    df = load_data(col_name)
    return [] if df is None else df.to_dict("records")

def upsert_many(col_name: str, records: list, keys: list) -> int:
    """Save records to local CSV, merging with existing data."""
    if not records:
        return 0
    df_new = pd.DataFrame(records)
    save_data(col_name, df_new)
    return len(records)

def _df_or_none(col_name: str) -> Optional[pd.DataFrame]:
    return load_data(col_name)

# ── Student roster ────────────────────────────────────────────────────────────
def save_roster(df: pd.DataFrame):
    """Save roster permanently and update session state."""
    overwrite_data("student_roster", df)
    st.session_state["student_roster_df"] = df

def load_roster() -> Optional[pd.DataFrame]:
    if "student_roster_df" in st.session_state:
        return st.session_state["student_roster_df"]
    df = load_data("student_roster")
    if df is not None:
        st.session_state["student_roster_df"] = df
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# GOOGLE SHEETS LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def _get_gspread_client(creds_dict: dict):
    """Return authenticated gspread client from a service account dict."""
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds  = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def push_to_gsheet(df: pd.DataFrame, sheet_url: str, tab_name: str, creds_dict: dict) -> Tuple[bool,str]:
    """Push a dataframe to a Google Sheet tab (creates tab if not exists)."""
    try:
        gc = _get_gspread_client(creds_dict)
        sh = gc.open_by_url(sheet_url)
        try:
            ws = sh.worksheet(tab_name)
            ws.clear()
        except Exception:
            ws = sh.add_worksheet(title=tab_name, rows=max(len(df)+10,200), cols=max(len(df.columns)+5,20))
        ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
        return True, f"✅ Pushed {len(df)} rows to tab '{tab_name}'"
    except Exception as e:
        return False, f"❌ Google Sheets error: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# PDF GENERATION (fpdf2)
# ═══════════════════════════════════════════════════════════════════════════════
def generate_rubric_pdf() -> bytes:
    """Generate a printable teacher scoring rubric PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(48, 43, 99)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "DEBATE IS COOL — Teacher Assessment Rubric", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 20)
    pdf.cell(0, 8, "Civic Space Nepal Pvt. Ltd.  |  4-Category Scoring  |  Scale: 1–5  |  Total: /20", ln=True)
    pdf.set_text_color(0, 0, 0)

    # Debate motion
    pdf.set_xy(10, 40)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "DEBATE MOTION", ln=True)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_fill_color(240, 240, 255)
    pdf.multi_cell(190, 8,
        '"This House Believes That social media does more harm than good to young people."',
        border=1, fill=True)
    pdf.ln(3)

    # Task 1
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(48, 43, 99)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 8, "  TASK 1: Written Argument  (Word limit: 150 words | Time: 10 min)", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(190, 6,
        "Write ONE clear argument in SUPPORT of or AGAINST the motion above.\n"
        "Your argument must include:\n"
        "  • A clear claim (what you believe)\n"
        "  • A reason (why you believe it)\n"
        "  • Evidence or an example (a fact, statistic, or real-world case)\n"
        "  • A link back to the motion\n\n"
        "Write here:\n" + "_"*90 + "\n" + "_"*90 + "\n" + "_"*90 + "\n" + "_"*90 + "\n" + "_"*90,
        border=0)
    pdf.ln(2)

    # Task 2
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_fill_color(48, 43, 99)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(190, 8, "  TASK 2: Written Refutation  (Word limit: 100 words | Time: 7 min)", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(190, 6,
        "Read the argument below, then write a refutation (a counter-argument).\n\n"
        "OPPONENT'S ARGUMENT: \"Social media connects young people across the world and gives "
        "them a platform to raise awareness about important social issues, making it a net positive "
        "for society.\"\n\n"
        "Your refutation:\n" + "_"*90 + "\n" + "_"*90 + "\n" + "_"*90 + "\n" + "_"*90,
        border=0)
    pdf.ln(2)

    # Rubric table
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "SCORING RUBRIC  (Circle the score for each category)", ln=True)
    pdf.ln(2)

    categories = [
        ("Argument Clarity",
         "1 – Unclear/no argument\n2 – Partially clear\n3 – Mostly clear, minor gaps\n4 – Clear and direct\n5 – Exceptionally clear and precise"),
        ("Reasoning Depth",
         "1 – No reasoning present\n2 – Superficial reasoning\n3 – Adequate reasoning\n4 – Well-developed reasoning\n5 – Sophisticated, nuanced reasoning"),
        ("Refutation Quality",
         "1 – No refutation / irrelevant\n2 – Weak, misses key points\n3 – Addresses main point\n4 – Strong, well-reasoned counter\n5 – Excellent; anticipates and dismantles"),
        ("Structure & Strategy",
         "1 – No structure\n2 – Some organization\n3 – Clear structure, minor gaps\n4 – Well-organized, logical flow\n5 – Strategic, polished structure"),
    ]

    for i, (cat, desc) in enumerate(categories):
        pdf.set_fill_color(230, 230, 250) if i % 2 == 0 else pdf.set_fill_color(245, 245, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(190, 7, f"  {i+1}. {cat}   [ 1 ]  [ 2 ]  [ 3 ]  [ 4 ]  [ 5 ]", ln=True, fill=True, border=1)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(190, 5, "  " + desc.replace("\n","\n  "), border="LRB", fill=True)
        pdf.ln(1)

    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(95, 9, "Student Name: _______________________________", border=1)
    pdf.cell(95, 9, "School: _____________________________________", border=1, ln=True)
    pdf.cell(95, 9, "Cohort: _____________________________________", border=1)
    pdf.cell(47, 9, "Total Score:  __ / 20", border=1)
    pdf.cell(48, 9, "Score %:  ____%", border=1, ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120,120,120)
    pdf.cell(0, 6, "Debate is Cool Program  |  Civic Space Nepal Pvt. Ltd.  |  Confidential — For Teacher Use Only", ln=True)

    return bytes(pdf.output())

def generate_insight_pdf(title: str, sections: list) -> bytes:
    """
    Generate a structured analytical insight brief as PDF.
    sections = list of (heading: str, paragraphs: list[str])
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Cover header
    pdf.set_fill_color(48, 43, 99)
    pdf.rect(0, 0, 210, 38, "F")
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",16)
    pdf.set_xy(10,10)
    pdf.multi_cell(190, 9, title)
    pdf.set_font("Helvetica","",9)
    pdf.set_xy(10,28)
    pdf.cell(0,6,"Civic Space Nepal Pvt. Ltd.  |  Debate is Cool Program  |  Auto-Generated Analytical Report")
    pdf.set_text_color(0,0,0)
    pdf.ln(20)

    for heading, paras in sections:
        pdf.set_font("Helvetica","B",12)
        pdf.set_fill_color(220,220,245)
        pdf.cell(190,8,f"  {heading}",ln=True,fill=True,border=0)
        pdf.ln(1)
        pdf.set_font("Helvetica","",10)
        for para in paras:
            # wrap long lines
            wrapped = textwrap.fill(para, width=100)
            pdf.multi_cell(190,6,wrapped)
            pdf.ln(1)
        pdf.ln(3)

    pdf.set_font("Helvetica","",8)
    pdf.set_text_color(150,150,150)
    pdf.cell(0,6,"Auto-generated by Debate is Cool Monitoring System  |  Confidential",ln=True)
    return bytes(pdf.output())

# ═══════════════════════════════════════════════════════════════════════════════
# CSV PARSING
# ═══════════════════════════════════════════════════════════════════════════════
FRAMEWORK_SCHEMAS = {
    "participation": {
        "required": ["school","cohort","student_name","session_date",
                     "attendance","speaking_turns","leadership_role"],
        "types": {"speaking_turns":"numeric","attendance":"categorical",
                  "leadership_role":"categorical"},
    },
    "teacher_scores": {
        "required": ["school","cohort","student_name",
                     "argument_clarity","reasoning_depth",
                     "refutation_quality","structure_strategy"],
        "types": {"argument_clarity":"numeric","reasoning_depth":"numeric",
                  "refutation_quality":"numeric","structure_strategy":"numeric"},
    },
    "student_survey": {
        "required": ["school","cohort",
                     "teacher_q1","teacher_q2","teacher_q3","teacher_q4","teacher_q5",
                     "growth_q1","growth_q2","growth_q3","growth_q4","growth_q5"],
        "types": {k:"numeric" for k in TEACHER_COLS+GROWTH_COLS},
    },
    "student_roster": {
        "required": ["school","cohort","student_name"],
        "types": {},
    },
}

COLUMN_ALIASES = {
    "School":"school","school name":"school","Cohort":"cohort","group":"cohort",
    "Student Name":"student_name","Name":"student_name","student":"student_name",
    "Full Name":"student_name","name":"student_name",
    "Session Date":"session_date","date":"session_date","Date":"session_date",
    "Attendance":"attendance","Present/Absent":"attendance","Status":"attendance",
    "Speaking Turns":"speaking_turns","Turns":"speaking_turns","turns":"speaking_turns",
    "Leadership Role":"leadership_role","Leader":"leadership_role","leader":"leadership_role",
    "Argument Clarity":"argument_clarity","Reasoning Depth":"reasoning_depth",
    "Refutation Quality":"refutation_quality",
    "Structure & Strategy":"structure_strategy","Structure and Strategy":"structure_strategy",
    **{f"Teacher Q{i}":f"teacher_q{i}" for i in range(1,6)},
    **{f"Growth Q{i}":f"growth_q{i}" for i in range(1,6)},
}

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    df.columns = [c.lower().replace(" ","_").replace("&","and").replace("/","_")
                  for c in df.columns]
    return df

def _detect_framework(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    for fw, s in FRAMEWORK_SCHEMAS.items():
        if all(r in cols for r in s["required"]):
            return fw
    scores = {fw: sum(1 for r in s["required"] if r in cols)/len(s["required"])
              for fw, s in FRAMEWORK_SCHEMAS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 0.5 else None

def _clean(df: pd.DataFrame, fw: str) -> Tuple[pd.DataFrame, list]:
    warns, schema = [], FRAMEWORK_SCHEMAS[fw]
    missing = [c for c in schema["required"] if c not in df.columns]
    if missing:
        warns.append(f"⚠️ Missing columns: {', '.join(missing)}")
    for col, typ in schema["types"].items():
        if col not in df.columns: continue
        if typ == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            n = df[col].isna().sum()
            if n: warns.append(f"⚠️ {n} non-numeric values in '{col}' → NaN")
        else:
            df[col] = df[col].astype(str).str.strip()
    if fw == "participation":
        for c in ("attendance","leadership_role"):
            if c in df.columns: df[c] = df[c].str.title()
        if "speaking_turns" in df.columns:
            df["speaking_turns"] = df["speaking_turns"].fillna(0).clip(lower=0)
    if fw == "teacher_scores":
        for c in SCORE_COLS:
            if c in df.columns: df[c] = df[c].clip(1,5)
        df["total_score"] = df[[c for c in SCORE_COLS if c in df.columns]].sum(axis=1)
        df["score_pct"]   = (df["total_score"]/20*100).round(1)
    if fw == "student_survey":
        lc = [c for c in df.columns if c.startswith(("teacher_q","growth_q"))]
        for c in lc: df[c] = pd.to_numeric(df[c],errors="coerce").clip(1,5)
        tc = [c for c in lc if c.startswith("teacher_q")]
        gc = [c for c in lc if c.startswith("growth_q")]
        if tc: df["teacher_effectiveness_avg"] = df[tc].mean(axis=1).round(2)
        if gc: df["self_growth_avg"]           = df[gc].mean(axis=1).round(2)
    df = df.dropna(how="all").reset_index(drop=True)
    return df, warns

def parse_csv(file_obj) -> Tuple[Optional[pd.DataFrame], Optional[str], list]:
    try:
        raw = file_obj.read()
        try:    df = pd.read_csv(io.BytesIO(raw))
        except: df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        df = _normalize_cols(df)
        fw = _detect_framework(df)
        if fw is None:
            return None, None, ["❌ Could not detect framework from column names."]
        df, warns = _clean(df, fw)
        return df, fw, warns
    except Exception as e:
        return None, None, [f"❌ Parse error: {e}"]

# ── 70% coverage checker ──────────────────────────────────────────────────────
def check_coverage(data_df: pd.DataFrame, roster_df: pd.DataFrame,
                   name_col: str = "student_name") -> pd.DataFrame:
    """
    Compare assessed/surveyed students against master roster per school.
    Returns a dataframe with coverage % per school and a pass/fail flag.
    """
    roster_counts = roster_df.groupby("school")[name_col].nunique().reset_index()
    roster_counts.columns = ["school","total_students"]
    if name_col in data_df.columns:
        data_counts = data_df.groupby("school")[name_col].nunique().reset_index()
    else:
        data_counts = data_df.groupby("school").size().reset_index()
        data_counts.columns = ["school","assessed"]
        data_counts = data_counts.rename(columns={"assessed":name_col})
    data_counts.columns = ["school","assessed"]
    merged = roster_counts.merge(data_counts, on="school", how="left").fillna(0)
    merged["assessed"]    = merged["assessed"].astype(int)
    merged["coverage_pct"]= (merged["assessed"]/merged["total_students"].clip(1)*100).round(1)
    merged["passes_70pct"]= merged["coverage_pct"] >= 70
    return merged

# ═══════════════════════════════════════════════════════════════════════════════
# ML ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def run_participation_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["present_flag"] = (df["attendance"].str.title()=="Present").astype(int)
    df["leader_flag"]  = (df["leadership_role"].str.title()=="Yes").astype(int)
    agg = df.groupby(["school","student_name"]).agg(
        sessions_attended=("present_flag","sum"),
        total_sessions=("present_flag","count"),
        avg_speaking=("speaking_turns","mean"),
        leadership_count=("leader_flag","sum"),
        cohort=("cohort","first"),
    ).reset_index()
    agg["attendance_rate"]     = (agg["sessions_attended"]/agg["total_sessions"].clip(1)*100).round(1)
    agg["participation_score"] = (
        agg["attendance_rate"]*0.4 +
        agg["avg_speaking"].clip(0,10)*4 +
        agg["leadership_count"].clip(0,5)*4
    ).round(2)
    feats = agg[["attendance_rate","avg_speaking","leadership_count","participation_score"]].fillna(0)
    k    = min(3, len(agg))
    pipe = Pipeline([("imp",SimpleImputer(strategy="mean")),
                     ("sc",StandardScaler()),
                     ("km",KMeans(n_clusters=k,random_state=42,n_init=10))])
    agg["cluster"] = pipe.fit_predict(feats)
    cm = agg.groupby("cluster")["participation_score"].mean().sort_values()
    tm = {c:["Low","Medium","High"][min(i,2)] for i,(c,_) in enumerate(cm.items())}
    agg["engagement_tier"] = agg["cluster"].map(tm)
    p10 = agg["participation_score"].quantile(0.10)
    p90 = agg["participation_score"].quantile(0.90)
    agg["cohort_flag"] = "Middle"
    agg.loc[agg["participation_score"]>=p90,"cohort_flag"] = "Top 10%"
    agg.loc[agg["participation_score"]<=p10,"cohort_flag"] = "Bottom 10%"
    joblib.dump(pipe, _mpath("participation"))
    return agg

def school_participation_stats(agg: pd.DataFrame) -> pd.DataFrame:
    s = agg.groupby("school").agg(
        avg_score=("participation_score","mean"),
        std_score=("participation_score","std"),
        student_count=("student_name","count"),
        high_count=("engagement_tier",lambda x:(x=="High").sum()),
        low_count=("engagement_tier", lambda x:(x=="Low").sum()),
    ).reset_index()
    s["avg_score"]        = s["avg_score"].round(2)
    s["std_score"]        = s["std_score"].round(2)
    s["inequality_index"] = (s["std_score"]/s["avg_score"].clip(0.01)).round(3)
    return s

def run_teacher_score_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in SCORE_COLS:
        if c not in df.columns: df[c] = np.nan
    df["total_score"] = df[SCORE_COLS].sum(axis=1)
    df["score_pct"]   = (df["total_score"]/20*100).round(1)
    df["skill_band"]  = df["score_pct"].apply(
        lambda p: "High" if p>=70 else ("Medium" if p>=40 else "Low"))
    X  = df[SCORE_COLS].fillna(df[SCORE_COLS].mean())
    le = LabelEncoder(); y = le.fit_transform(df["skill_band"])
    if len(df) >= 5:
        pipe = Pipeline([("imp",SimpleImputer(strategy="mean")),
                         ("sc",StandardScaler()),
                         ("rf",RandomForestClassifier(n_estimators=50,random_state=42))])
        pipe.fit(X, y)
        df["predicted_band"] = le.inverse_transform(pipe.predict(X))
        joblib.dump((pipe,le), _mpath("teacher_scores"))
    else:
        df["predicted_band"] = df["skill_band"]
    return df

def school_score_stats(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("school").agg(
        avg_pct=("score_pct","mean"), student_count=("student_name","count"),
        high_band=("skill_band",lambda x:(x=="High").sum()),
        medium_band=("skill_band",lambda x:(x=="Medium").sum()),
        low_band=("skill_band", lambda x:(x=="Low").sum()),
        avg_arg_clarity=("argument_clarity","mean"),
        avg_reasoning=("reasoning_depth","mean"),
        avg_refutation=("refutation_quality","mean"),
        avg_structure=("structure_strategy","mean"),
    ).reset_index()
    for c in ["avg_pct","avg_arg_clarity","avg_reasoning","avg_refutation","avg_structure"]:
        s[c] = s[c].round(2)
    return s

def category_stats(df: pd.DataFrame) -> dict:
    return {c:round(df[c].mean(),2) for c in SCORE_COLS if c in df.columns}

def run_survey_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in TEACHER_COLS+GROWTH_COLS:
        if c not in df.columns: df[c] = np.nan
    df["teacher_effectiveness_avg"] = df[TEACHER_COLS].mean(axis=1).round(2)
    df["self_growth_avg"]           = df[GROWTH_COLS].mean(axis=1).round(2)
    X = df[TEACHER_COLS+GROWTH_COLS].fillna(3)
    if len(df) >= 10:
        iso = IsolationForest(contamination=0.1,random_state=42)
        df["is_anomaly"] = iso.fit_predict(X)==-1
        joblib.dump(iso, _mpath("survey"))
    else:
        df["is_anomaly"] = False
    return df

def school_survey_stats(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("school").agg(
        avg_teacher_score=("teacher_effectiveness_avg","mean"),
        avg_growth_score=("self_growth_avg","mean"),
        response_count=("school","count"),
    ).reset_index()
    s["avg_teacher_score"] = s["avg_teacher_score"].round(2)
    s["avg_growth_score"]  = s["avg_growth_score"].round(2)
    return s

def compute_cross_correlation(part_agg, score_df) -> Optional[pd.DataFrame]:
    try:
        p = part_agg.groupby("school")["participation_score"].mean().reset_index()
        p.columns = ["school","avg_participation"]
        s = score_df.groupby("school")["score_pct"].mean().reset_index()
        s.columns = ["school","avg_skill_pct"]
        m = p.merge(s,on="school",how="inner")
        if len(m)>=2: m["correlation"] = round(m["avg_participation"].corr(m["avg_skill_pct"]),3)
        return m
    except: return None

# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def metric(label: str, value):
    st.markdown(f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>""", unsafe_allow_html=True)

def insight(title: str, body: str):
    st.markdown(f"""<div class="insight-box">
        <strong>{title}</strong><br>
        <span style='color:#444;font-size:.88rem'>{body}</span>
    </div>""", unsafe_allow_html=True)

def risk(title: str, body: str):
    st.markdown(f"""<div class="risk-box">
        <strong>{title}</strong><br>
        <span style='color:#555;font-size:.88rem'>{body}</span>
    </div>""", unsafe_allow_html=True)

def rec(title: str, body: str):
    st.markdown(f"""<div class="rec-box">
        <strong>{title}</strong><br>
        <span style='color:#444;font-size:.88rem'>{body}</span>
    </div>""", unsafe_allow_html=True)

def warn_block(title: str, body: str):
    st.markdown(f"""<div class="warn-box">
        <strong>{title}</strong><br>
        <span style='color:#555;font-size:.88rem'>{body}</span>
    </div>""", unsafe_allow_html=True)

def coverage_banner(data_df, fw_label, name_col="student_name"):
    """Show 70% coverage check if roster is loaded."""
    roster = load_roster()
    if roster is None:
        st.caption("ℹ️ Upload student roster to enable 70% coverage check.")
        return
    cov = check_coverage(data_df, roster, name_col)
    failing = cov[~cov["passes_70pct"]]
    if failing.empty:
        st.success(f"✅ All schools meet the 70% {fw_label} coverage requirement.")
    else:
        st.markdown(f"""<div class="warn-box">
            <strong>⚠️ Coverage Warning — {fw_label}</strong><br>
            The following schools have fewer than 70% of students assessed:
        </div>""", unsafe_allow_html=True)
        disp = cov[["school","total_students","assessed","coverage_pct","passes_70pct"]].copy()
        disp.columns = ["School","Total Students","Assessed","Coverage %","Meets 70%"]
        disp["Meets 70%"] = disp["Meets 70%"].map({True:"✅ Yes", False:"❌ No"})
        st.dataframe(disp, use_container_width=True, hide_index=True)

def gsheet_push_widget(df: pd.DataFrame, tab_name: str):
    """Reusable Google Sheets push widget."""
    with st.expander("📤 Push to Google Sheets", expanded=False):
        st.markdown("Paste your **Google Sheets URL** and upload your **Service Account JSON** key file.")
        sheet_url = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...", key=f"gurl_{tab_name}")
        json_file = st.file_uploader("Service Account JSON key", type=["json"], key=f"gkey_{tab_name}")

        if st.button(f"Push '{tab_name}' data to Sheet", key=f"gpush_{tab_name}"):
            if not sheet_url or json_file is None:
                st.warning("Please provide both the Sheet URL and the JSON key file.")
            else:
                try:
                    creds_dict = json.loads(json_file.read().decode("utf-8"))
                    with st.spinner("Connecting to Google Sheets..."):
                        ok, msg = push_to_gsheet(df, sheet_url, tab_name, creds_dict)
                    if ok:
                        st.success(msg)
                        st.markdown(f"[🔗 Open Sheet]({sheet_url})", unsafe_allow_html=False)
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"❌ JSON parse error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE CSV DATA
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_ROSTER = """school,cohort,student_name
Sunrise School,A,Aarav Sharma
Sunrise School,A,Priya Thapa
Sunrise School,A,Rohan KC
Sunrise School,A,Meera Shrestha
Moonlight Academy,B,Sita Rai
Moonlight Academy,B,Bikram Lama
Moonlight Academy,B,Anisha Gurung
Moonlight Academy,B,Dipesh Bhatt
Valley High School,C,Kabir Magar
Valley High School,C,Nisha Tamang
Valley High School,C,Dev Karki
Valley High School,C,Puja Adhikari
"""

SAMPLE_PARTICIPATION = """school,cohort,student_name,session_date,attendance,speaking_turns,leadership_role
Sunrise School,A,Aarav Sharma,2024-01-10,Present,4,Yes
Sunrise School,A,Priya Thapa,2024-01-10,Present,2,No
Sunrise School,A,Rohan KC,2024-01-10,Absent,0,No
Sunrise School,A,Meera Shrestha,2024-01-10,Present,3,No
Sunrise School,A,Aarav Sharma,2024-01-17,Present,6,Yes
Sunrise School,A,Priya Thapa,2024-01-17,Present,1,No
Sunrise School,A,Rohan KC,2024-01-17,Present,3,No
Sunrise School,A,Meera Shrestha,2024-01-17,Present,2,Yes
Moonlight Academy,B,Sita Rai,2024-01-10,Present,5,Yes
Moonlight Academy,B,Bikram Lama,2024-01-10,Present,0,No
Moonlight Academy,B,Anisha Gurung,2024-01-10,Present,3,Yes
Moonlight Academy,B,Dipesh Bhatt,2024-01-10,Absent,0,No
Moonlight Academy,B,Sita Rai,2024-01-17,Present,7,Yes
Moonlight Academy,B,Bikram Lama,2024-01-17,Absent,0,No
Moonlight Academy,B,Anisha Gurung,2024-01-17,Present,2,No
Moonlight Academy,B,Dipesh Bhatt,2024-01-17,Present,1,No
Valley High School,C,Kabir Magar,2024-01-10,Present,1,No
Valley High School,C,Nisha Tamang,2024-01-10,Present,2,No
Valley High School,C,Dev Karki,2024-01-10,Absent,0,No
Valley High School,C,Puja Adhikari,2024-01-10,Present,1,No
Valley High School,C,Kabir Magar,2024-01-17,Absent,0,No
Valley High School,C,Nisha Tamang,2024-01-17,Present,1,No
Valley High School,C,Dev Karki,2024-01-17,Present,0,No
Valley High School,C,Puja Adhikari,2024-01-17,Present,2,No
"""

SAMPLE_TEACHER = """school,cohort,student_name,argument_clarity,reasoning_depth,refutation_quality,structure_strategy
Sunrise School,A,Aarav Sharma,4,4,3,5
Sunrise School,A,Priya Thapa,3,3,2,3
Sunrise School,A,Rohan KC,2,2,2,2
Sunrise School,A,Meera Shrestha,3,4,3,4
Moonlight Academy,B,Sita Rai,5,4,5,4
Moonlight Academy,B,Bikram Lama,2,1,2,2
Moonlight Academy,B,Anisha Gurung,3,3,3,4
Moonlight Academy,B,Dipesh Bhatt,2,2,2,3
Valley High School,C,Kabir Magar,2,2,1,2
Valley High School,C,Nisha Tamang,3,2,2,3
Valley High School,C,Dev Karki,1,1,1,1
Valley High School,C,Puja Adhikari,2,2,2,2
"""

SAMPLE_SURVEY = """school,cohort,teacher_q1,teacher_q2,teacher_q3,teacher_q4,teacher_q5,growth_q1,growth_q2,growth_q3,growth_q4,growth_q5
Sunrise School,A,4,5,4,4,5,4,3,4,4,3
Sunrise School,A,3,4,3,4,4,3,3,3,4,3
Sunrise School,A,5,5,5,5,5,4,4,5,5,4
Moonlight Academy,B,4,4,3,4,4,4,4,3,4,4
Moonlight Academy,B,2,2,3,2,3,2,2,2,3,2
Moonlight Academy,B,4,5,4,5,4,4,3,4,5,4
Valley High School,C,2,2,2,3,2,2,2,1,2,2
Valley High School,C,3,3,2,3,3,2,3,2,3,2
Valley High School,C,2,1,2,2,2,1,2,2,2,1
"""

FW_LABELS = {
    "participation":  "📊 Framework 1 — Student Participation",
    "teacher_scores": "📝 Framework 2 — Teacher Assessment",
    "student_survey": "⭐ Framework 3 — Student Survey",
    "student_roster": "📋 Student Roster",
}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.2rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    letter-spacing:-.5px;'>🎙️ Debate is Cool</div>
        <div style='font-size:.72rem;opacity:.55;margin-top:2px;'>Monitoring & Insight System · CSN</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Home",
        "👥  Student Roster",
        "📤  Upload CSV Data",
        "📊  Framework 1 — Participation",
        "📝  Framework 2 — Teacher Assessment",
        "⭐  Framework 3 — Student Survey",
        "📄  PDF Reports",
        "🗄️  Manage Stored Data",
    ], label_visibility="collapsed")

    st.markdown("---")
    roster = load_roster()
    db_ok  = db_connected()
    # Show what's saved on disk
    saved_files = [f.replace(".csv","") for f in os.listdir(DATA_DIR) if f.endswith(".csv")] if os.path.exists(DATA_DIR) else []
    st.markdown(f"""
    <div style='font-size:.72rem;opacity:.55;line-height:1.9;'>
        💾 Local storage: {len(saved_files)} dataset(s) saved<br>
        Roster: {'✅ ' + str(len(roster)) + ' students' if roster is not None else '❌ Not loaded'}<br>
        Civic Space Nepal Pvt. Ltd.<br>
        8 Schools · 3 Frameworks
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""<div class="main-header">
        <h1>🎙️ Debate is Cool</h1>
        <p>Monitoring & Insight Generation System · Civic Space Nepal Pvt. Ltd. · 8 Schools · 3 Frameworks</p>
    </div>""", unsafe_allow_html=True)

    p_data = fetch_all("participation")
    s_data = fetch_all("teacher_scores")
    q_data = fetch_all("student_survey")
    roster = load_roster()
    schools = set()
    for d in [p_data,s_data,q_data]:
        if d:
            df_ = pd.DataFrame(d)
            if "school" in df_.columns: schools.update(df_["school"].unique())

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val in zip(
        [c1,c2,c3,c4,c5],
        ["Roster Students","Participation Records","Teacher Assessments","Survey Responses","Schools"],
        [len(roster) if roster is not None else "—",
         len(p_data), len(s_data), len(q_data), f"{len(schools)}/8"]
    ):
        with col: metric(lbl, val)

    st.markdown("<br>", unsafe_allow_html=True)
    if roster is None:
        st.warning("⚠️ **Step 1:** Go to **👥 Student Roster** and upload your master student list before uploading any data.")

    st.subheader("📋 Framework Overview")
    c1,c2,c3 = st.columns(3)
    fw_info = [
        ("📊","Framework 1","Student Participation",
         "Tracks attendance, speaking turns, and leadership. KMeans clusters students into engagement tiers.",
         ["Participation scoring","Engagement clustering","Top/Bottom 10%","Risk school detection","70% coverage check","Google Sheets export"]),
        ("📝","Framework 2","Teacher Assessment",
         "4-category rubric (1–5) scored by teachers. Random Forest classifies skill bands. Printable PDF rubric.",
         ["Printable test + rubric PDF","Skill band classification","Category analysis","Plateau detection","70% coverage check","Google Sheets export"]),
        ("⭐","Framework 3","Student Survey",
         "Anonymous 10-question Likert survey. Isolation Forest flags anomalous responses.",
         ["Teacher effectiveness score","Self-growth score","Anomaly detection","Perception gap analysis","70% coverage check","Google Sheets export"]),
    ]
    for col,(icon,fw,title,desc,features) in zip([c1,c2,c3],fw_info):
        with col:
            feats = "".join(f"<li style='font-size:.8rem;margin:2px 0'>{f}</li>" for f in features)
            st.markdown(f"""
            <div style='background:white;border:1px solid #e8e8f0;border-radius:14px;
                        padding:1.3rem;box-shadow:0 2px 10px rgba(48,43,99,.07);'>
                <div style='font-size:1.8rem'>{icon}</div>
                <div style='font-family:Syne,sans-serif;font-size:.65rem;text-transform:uppercase;
                            letter-spacing:1.5px;color:#667eea;font-weight:600;margin:5px 0 2px'>{fw}</div>
                <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;
                            color:#302b63;margin-bottom:.5rem'>{title}</div>
                <p style='font-size:.82rem;color:#555;line-height:1.5'>{desc}</p>
                <ul style='padding-left:1rem;color:#444;margin-top:.7rem'>{feats}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Start with **👥 Student Roster** to upload your master list, then use **📤 Upload CSV Data** for session data.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: STUDENT ROSTER  (NEW — replaces dropdown limitation for 1000+ students)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥  Student Roster":
    st.markdown("""<div class="main-header">
        <h1>👥 Student Roster</h1>
        <p>Upload master student list · Enables name search across all frameworks · Powers 70% coverage check</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    The roster is your **master student list** across all 8 schools. It must be uploaded first.
    It powers:
    - **Searchable student lookup** in participation entry (handles 1000+ names efficiently)
    - **70% coverage enforcement** for Framework 2 and 3
    - **Cross-framework student matching**
    """)

    with st.expander("📥 Download Roster Template", expanded=False):
        st.download_button("⬇️ Student Roster Template", SAMPLE_ROSTER,
                           "student_roster_sample.csv", "text/csv")
        st.caption("Required columns: `school`, `cohort`, `student_name`  —  One row per student. No duplicates.")

    st.markdown("### Upload Master Student List")
    roster_file = st.file_uploader("Upload roster CSV", type=["csv"],
                                   label_visibility="collapsed", key="roster_upload")

    if roster_file:
        df_r, fw_r, warns_r = parse_csv(roster_file)
        if df_r is None or fw_r != "student_roster":
            # Try raw parse as roster
            roster_file.seek(0)
            try:
                df_r = pd.read_csv(roster_file)
                df_r = _normalize_cols(df_r)
                if "student_name" not in df_r.columns:
                    st.error("❌ Roster must have at least: school, cohort, student_name columns.")
                    df_r = None
            except Exception as e:
                st.error(f"❌ Could not read file: {e}")
                df_r = None

        if df_r is not None:
            df_r = df_r[["school","cohort","student_name"] +
                        [c for c in df_r.columns if c not in ["school","cohort","student_name"]]].copy()
            df_r = df_r.drop_duplicates(subset=["school","student_name"]).reset_index(drop=True)
            for w in warns_r: st.warning(w)
            st.success(f"✅ {len(df_r)} students loaded across {df_r['school'].nunique()} schools")

            c1,c2,c3 = st.columns(3)
            c1.metric("Total Students", len(df_r))
            c2.metric("Schools", df_r["school"].nunique())
            c3.metric("Cohorts", df_r["cohort"].nunique())

            # School breakdown
            sb = df_r.groupby(["school","cohort"]).size().reset_index(name="Students")
            sb.columns = ["School","Cohort","Students"]
            st.dataframe(sb, use_container_width=True, hide_index=True)

            if st.button("💾 Save Roster", use_container_width=True):
                save_roster(df_r)
                st.success("✅ Roster saved! You can now upload participation, score, and survey data.")

            # Push roster to Google Sheets
            gsheet_push_widget(df_r, "Student_Roster")

    # Show current roster if already loaded
    current = load_roster()
    if current is not None:
        st.markdown("---")
        st.markdown("### 📋 Current Loaded Roster")

        # Search box — efficient for 1000+ names
        search = st.text_input("🔍 Search student name or school",
                               placeholder="Type to filter...", key="roster_search")
        disp_r = current.copy()
        if search.strip():
            mask = (
                disp_r["student_name"].str.contains(search, case=False, na=False) |
                disp_r["school"].str.contains(search, case=False, na=False)
            )
            disp_r = disp_r[mask]

        st.caption(f"Showing {len(disp_r)} of {len(current)} students")
        st.dataframe(disp_r.reset_index(drop=True), use_container_width=True,
                     hide_index=True, height=400)

        csv_r = current.to_csv(index=False).encode()
        st.download_button("⬇️ Download Roster CSV", csv_r, "student_roster.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📤  Upload CSV Data":
    st.markdown("""<div class="main-header">
        <h1>📤 Upload CSV Data</h1>
        <p>Framework auto-detected · Validated · ML processed · Stored to MongoDB</p>
    </div>""", unsafe_allow_html=True)

    roster = load_roster()
    if roster is None:
        st.warning("⚠️ No student roster loaded. Upload the roster first for name validation and 70% coverage checks.")

    with st.expander("📥 Download Sample CSV Templates", expanded=False):
        tc1,tc2,tc3,tc4 = st.columns(4)
        with tc1: st.download_button("⬇️ Roster",        SAMPLE_ROSTER,        "roster_sample.csv",        "text/csv")
        with tc2: st.download_button("⬇️ Participation", SAMPLE_PARTICIPATION, "participation_sample.csv", "text/csv")
        with tc3: st.download_button("⬇️ Teacher Scores",SAMPLE_TEACHER,       "teacher_scores_sample.csv","text/csv")
        with tc4: st.download_button("⬇️ Student Survey",SAMPLE_SURVEY,        "student_survey_sample.csv","text/csv")

    with st.expander("📋 Required Columns per Framework", expanded=False):
        for fw, schema in FRAMEWORK_SCHEMAS.items():
            if fw == "student_roster": continue
            st.markdown(f"**{FW_LABELS[fw]}**")
            st.code(", ".join(schema["required"]))

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is None:
        st.markdown("""
        <div style='background:#f8f9ff;border:2px dashed #c8c8e8;border-radius:12px;
                    padding:2.5rem;text-align:center;color:#888;'>
            <div style='font-size:2.5rem;margin-bottom:.5rem'>☁️</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:600'>Upload a CSV to begin</div>
            <div style='font-size:.83rem;margin-top:.3rem'>Framework is auto-detected from column names</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("Parsing and validating..."):
            df, fw, warns = parse_csv(uploaded)

        if df is None:
            for w in warns: st.error(w)
        else:
            st.success(f"✅ Detected: **{FW_LABELS.get(fw,fw)}** — {len(df):,} rows loaded")
            for w in warns: st.warning(w)

            # Name validation against roster (warns on unknown names)
            if roster is not None and "student_name" in df.columns:
                known = set(roster["student_name"].str.lower().str.strip())
                df_names = set(df["student_name"].str.lower().str.strip())
                unknown = df_names - known
                if unknown:
                    st.warning(f"⚠️ {len(unknown)} student name(s) not found in roster: "
                               f"{', '.join(sorted(unknown)[:10])}"
                               f"{'...' if len(unknown)>10 else ''}")

            st.markdown("#### 🔍 Data Preview (first 50 rows)")
            st.dataframe(df.head(50), use_container_width=True)
            m1,m2,m3 = st.columns(3)
            m1.metric("Rows", f"{len(df):,}")
            m2.metric("Columns", len(df.columns))
            m3.metric("Schools", df["school"].nunique() if "school" in df.columns else "—")

            st.markdown("---")
            if st.button("▶️ Process & Store Data", use_container_width=True):
                with st.spinner("Running ML pipeline..."):
                    try:
                        if fw == "participation":
                            result = run_participation_model(df)
                            save_data("participation",    df)
                            save_data("ml_participation", result)
                            st.session_state["participation_raw"] = df
                            st.session_state["participation_agg"] = result
                        elif fw == "teacher_scores":
                            result = run_teacher_score_model(df)
                            save_data("teacher_scores", result)
                            st.session_state["teacher_scores_df"] = result
                        elif fw == "student_survey":
                            result = run_survey_model(df)
                            save_data("student_survey", result)
                            st.session_state["student_survey_df"] = result
                        elif fw == "student_roster":
                            save_roster(df)
                            st.success("✅ Roster saved permanently!")
                            st.stop()

                        st.success("✅ Data processed and stored!")
                        st.dataframe(result.head(30), use_container_width=True)
                        st.info(f"👉 Go to **{FW_LABELS.get(fw,fw)}** for full analysis.")
                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")
                        import traceback; st.code(traceback.format_exc())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAMEWORK 1 — PARTICIPATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Framework 1 — Participation":
    st.markdown("""<div class="main-header">
        <h1>📊 Framework 1 — Student Participation</h1>
        <p>Engagement scoring · KMeans clustering · Top/Bottom 10% · Risk school identification</p>
    </div>""", unsafe_allow_html=True)

    if "participation_agg" in st.session_state:
        agg_df = st.session_state["participation_agg"]
        raw_df = st.session_state.get("participation_raw", agg_df)
    else:
        agg_df = load_data("ml_participation")
        raw_df = load_data("participation")
        if agg_df is not None:
            st.session_state["participation_agg"] = agg_df
        if raw_df is not None:
            st.session_state["participation_raw"] = raw_df

    if agg_df is None or (isinstance(agg_df,pd.DataFrame) and agg_df.empty):
        st.info("No participation data found. Upload a Participation CSV first.")
        st.stop()

    # 70% coverage check
    coverage_banner(raw_df if raw_df is not None else agg_df, "Participation")
    st.markdown("---")

    stats = school_participation_stats(agg_df)
    top10 = agg_df[agg_df["cohort_flag"]=="Top 10%"]
    bot10 = agg_df[agg_df["cohort_flag"]=="Bottom 10%"]

    # ── Section 1 ─────────────────────────────────────────────────────────────
    st.markdown("## 📌 Section 1 — School-Level Participation Averages")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Overall Avg Score", f"{agg_df['participation_score'].mean():.1f}")
    with c2: metric("Total Students",    f"{len(agg_df):,}")
    with c3: metric("Top 10%",           len(top10))
    with c4: metric("Bottom 10%",        len(bot10))

    st.markdown("<br>", unsafe_allow_html=True)
    fig = px.bar(stats.sort_values("avg_score",ascending=False),
                 x="school",y="avg_score",color="avg_score",text="avg_score",
                 color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                 title="Average Participation Score per School",
                 labels={"avg_score":"Avg Score","school":"School"})
    fig.update_traces(texttemplate='%{text:.1f}',textposition='outside')
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                      coloraxis_showscale=False,xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    disp = stats[["school","avg_score","std_score","student_count",
                  "high_count","low_count","inequality_index"]].copy()
    disp.columns = ["School","Avg Score","Std Dev","Students",
                    "High Engaged","Low Engaged","Inequality Index"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Section 2 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 2 — Variance & Top/Bottom 10%")
    c1,c2 = st.columns(2)
    with c1:
        fig2 = px.box(agg_df,x="school",y="participation_score",color="school",
                      title="Score Distribution per School",color_discrete_sequence=COLORS)
        fig2.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                           showlegend=False,xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        tc = agg_df.groupby(["school","engagement_tier"]).size().reset_index(name="count")
        fig3 = px.bar(tc,x="school",y="count",color="engagement_tier",barmode="stack",
                      title="Engagement Tier Distribution",
                      color_discrete_map={"High":"#43e97b","Medium":"#ffc107","Low":"#f5576c"})
        fig3.update_layout(plot_bgcolor="white",paper_bgcolor="white",xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### 🌟 Top 10% Participants (Cross-School)")
    if not top10.empty:
        t = top10[["school","student_name","participation_score","attendance_rate",
                   "avg_speaking","leadership_count"]].copy()
        t.columns = ["School","Student","Score","Att %","Avg Turns","Leadership"]
        st.dataframe(t.sort_values("Score",ascending=False),
                     use_container_width=True, hide_index=True)

    st.markdown("#### ⚠️ Bottom 10% Participants (Cross-School)")
    if not bot10.empty:
        b = bot10[["school","student_name","participation_score","attendance_rate",
                   "avg_speaking","leadership_count"]].copy()
        b.columns = ["School","Student","Score","Att %","Avg Turns","Leadership"]
        st.dataframe(b.sort_values("Score"), use_container_width=True, hide_index=True)

    fig4 = px.bar(stats.sort_values("inequality_index",ascending=False),
                  x="school",y="inequality_index",text="inequality_index",
                  color="inequality_index",
                  color_continuous_scale=["#43e97b","#ffc107","#f5576c"],
                  title="Participation Inequality Index (Std Dev / Mean)")
    fig4.update_traces(texttemplate='%{text:.3f}',textposition='outside')
    fig4.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                       coloraxis_showscale=False,xaxis_tickangle=-20)
    st.plotly_chart(fig4, use_container_width=True)

    # ── Section 3 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 3 — Risk Schools, High-Engagement Schools & Recommendations")
    sorted_s = stats.sort_values("avg_score")
    risk_sch = sorted_s.head(2)["school"].tolist()
    high_sch = sorted_s.tail(2)["school"].tolist()

    rc1,rc2 = st.columns(2)
    with rc1:
        st.markdown("#### 🔴 Engagement-Risk Schools (Bottom 2)")
        for s in risk_sch:
            r = stats[stats["school"]==s].iloc[0]
            risk(s, f"Avg Score: {r['avg_score']:.1f} | Low-Engaged: {int(r['low_count'])} | Inequality: {r['inequality_index']:.3f}")
    with rc2:
        st.markdown("#### 🟢 High-Engagement Schools (Top 2)")
        for s in high_sch:
            r = stats[stats["school"]==s].iloc[0]
            rec(s, f"Avg Score: {r['avg_score']:.1f} | High-Engaged: {int(r['high_count'])} | Inequality: {r['inequality_index']:.3f}")

    fig5 = px.scatter(agg_df,x="attendance_rate",y="avg_speaking",
                      color="engagement_tier",size="participation_score",
                      hover_data=["student_name","school"],
                      title="Attendance Rate vs Average Speaking Turns",
                      color_discrete_map={"High":"#43e97b","Medium":"#ffc107","Low":"#f5576c"})
    fig5.update_layout(plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### 💡 Recommendations")
    wr = stats[stats["school"]==risk_sch[0]].iloc[0] if risk_sch else None
    insight(f"🎯 Participation Quotas at {risk_sch[0] if risk_sch else 'Risk Schools'}",
            f"Avg score {wr['avg_score']:.1f}/100 with {int(wr['low_count'])} low-engaged students. "
            "Mandatory speaking-turn minimums (≥2 turns/session) enforced by facilitators."
            if wr else "Implement speaking-turn minimums at risk schools.")
    insight("📅 Attendance-First Intervention",
            f"Bottom 10% avg attendance: {bot10['attendance_rate'].mean():.1f}% vs top 10%: "
            f"{top10['attendance_rate'].mean():.1f}%. Weekly guardian SMS follow-up recommended."
            if (not bot10.empty and not top10.empty) else "Follow up on attendance at risk schools.")
    insight(f"🏆 Peer-Mentoring from {high_sch[-1] if high_sch else 'Top Schools'}",
            f"{high_sch[-1] if high_sch else 'Top schools'} shows high leadership uptake. "
            "Cross-school peer-mentoring pairing top-10% with bottom-10% would raise floor scores.")

    # Full table + download + Google Sheets
    st.markdown("---")
    full = agg_df[["school","cohort","student_name","participation_score","attendance_rate",
                   "avg_speaking","leadership_count","engagement_tier","cohort_flag"]].copy()
    full.columns = ["School","Cohort","Student","Score","Att %","Avg Turns",
                    "Leadership","Tier","Flag"]
    st.dataframe(full.sort_values("Score",ascending=False),
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Report CSV",
                       full.to_csv(index=False).encode(),
                       "framework1_participation.csv","text/csv")
    gsheet_push_widget(full, "Framework1_Participation")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAMEWORK 2 — TEACHER ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📝  Framework 2 — Teacher Assessment":
    st.markdown("""<div class="main-header">
        <h1>📝 Framework 2 — Teacher Assessment</h1>
        <p>4-category rubric scoring · Skill band classification · Cross-school comparison · PDF rubric</p>
    </div>""", unsafe_allow_html=True)

    if "teacher_scores_df" in st.session_state:
        df = st.session_state["teacher_scores_df"]
    else:
        df = load_data("teacher_scores")
        if df is not None:
            st.session_state["teacher_scores_df"] = df

    if df is None or (isinstance(df,pd.DataFrame) and df.empty):
        st.info("No teacher assessment data found. Upload a Teacher Scores CSV first.")
        st.stop()

    if "skill_band" not in df.columns:
        df = run_teacher_score_model(df)

    # 70% coverage check
    coverage_banner(df, "Teacher Assessment")
    st.markdown("---")

    stats  = school_score_stats(df)
    catavg = category_stats(df)
    high_n = (df["skill_band"]=="High").sum()
    low_n  = (df["skill_band"]=="Low").sum()

    # ── Section 1 ─────────────────────────────────────────────────────────────
    st.markdown("## 📌 Section 1 — School-Level Skill Comparison")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Overall Avg Score", f"{df['score_pct'].mean():.1f}%")
    with c2: metric("Students Assessed", f"{len(df):,}")
    with c3: metric("High Band",         high_n)
    with c4: metric("Low Band",          low_n)

    st.markdown("<br>", unsafe_allow_html=True)
    fig1 = px.bar(stats.sort_values("avg_pct",ascending=False),
                  x="school",y="avg_pct",color="avg_pct",text="avg_pct",
                  color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                  title="Average Skill Score (%) per School",
                  labels={"avg_pct":"Avg %","school":"School"})
    fig1.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
    fig1.add_hline(y=70,line_dash="dash",line_color="#302b63",annotation_text="High (70%)")
    fig1.add_hline(y=40,line_dash="dot", line_color="#f5576c",annotation_text="Low (40%)")
    fig1.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                       coloraxis_showscale=False,xaxis_tickangle=-20)
    st.plotly_chart(fig1, use_container_width=True)

    disp = stats[["school","avg_pct","student_count","high_band","medium_band","low_band"]].copy()
    disp.columns = ["School","Avg %","Students","High","Medium","Low"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Section 2 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 2 — Skill Distribution (High / Medium / Low)")
    c1,c2 = st.columns(2)
    with c1:
        bc = df["skill_band"].value_counts().reset_index()
        bc.columns = ["Band","Count"]
        fig2 = px.pie(bc,names="Band",values="Count",hole=0.4,
                      title="Overall Skill Band Distribution",
                      color="Band",color_discrete_map={"High":"#43e97b","Medium":"#ffc107","Low":"#f5576c"})
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        bs = df.groupby(["school","skill_band"]).size().reset_index(name="count")
        fig3 = px.bar(bs,x="school",y="count",color="skill_band",barmode="stack",
                      title="Skill Band Breakdown per School",
                      color_discrete_map={"High":"#43e97b","Medium":"#ffc107","Low":"#f5576c"})
        fig3.update_layout(plot_bgcolor="white",paper_bgcolor="white",xaxis_tickangle=-20)
        st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(df,x="score_pct",color="school",nbins=20,opacity=0.7,
                        color_discrete_sequence=COLORS,
                        title="Score % Distribution",labels={"score_pct":"Score %"})
    fig4.update_layout(plot_bgcolor="white",paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Section 3 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 3 — Category-Level Analysis")
    sorted_cats = sorted(catavg.items(),key=lambda x:x[1],reverse=True)
    if sorted_cats:
        c1,c2 = st.columns(2)
        with c1:
            best = sorted_cats[0]
            rec("💪 Strongest Category (Overall)",
                f"{CAT_LABELS.get(best[0],best[0])} — Avg: {best[1]:.2f}/5.0")
        with c2:
            worst = sorted_cats[-1]
            risk("⚠️ Weakest Category (Overall)",
                 f"{CAT_LABELS.get(worst[0],worst[0])} — Avg: {worst[1]:.2f}/5.0")

    # Radar per school
    fig5 = go.Figure()
    rc_cols = ["avg_arg_clarity","avg_reasoning","avg_refutation","avg_structure"]
    cn = ["Arg. Clarity","Reasoning","Refutation","Structure"]
    for i,(_,row) in enumerate(stats.iterrows()):
        vals = [row[c] for c in rc_cols]+[row[rc_cols[0]]]
        fig5.add_trace(go.Scatterpolar(r=vals,theta=cn+[cn[0]],fill='toself',
                                        name=row["school"],line_color=COLORS[i%len(COLORS)],opacity=0.7))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,5])),
                       title="Category Scores Radar — All Schools",paper_bgcolor="white")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Strongest & Weakest Category per School")
    lmap = {"avg_arg_clarity":"Argument Clarity","avg_reasoning":"Reasoning Depth",
            "avg_refutation":"Refutation Quality","avg_structure":"Structure & Strategy"}
    rows = []
    for _,sr in stats.iterrows():
        cv = {k:sr[k] for k in rc_cols}
        rows.append({"School":sr["school"],
                     "Strongest":lmap[max(cv,key=cv.get)], "Best Score":f"{max(cv.values()):.2f}",
                     "Weakest":lmap[min(cv,key=cv.get)],   "Worst Score":f"{min(cv.values()):.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Section 4 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 4 — Correlation, Plateau Detection & Recommendations")

    if "participation_agg" in st.session_state:
        cdf = compute_cross_correlation(st.session_state["participation_agg"], df)
        if cdf is not None and not cdf.empty:
            cv  = cdf["correlation"].iloc[0]
            insight(f"📈 Cross-Framework Pearson Correlation: {cv:.3f}",
                    "Strong positive: higher participation correlates with higher skill." if cv>0.5 else
                    "Moderate: participation and skill scores partially diverge." if cv>0 else
                    "Negative correlation detected — review data consistency.")
            fig6 = px.scatter(cdf,x="avg_participation",y="avg_skill_pct",text="school",
                              title="Avg Participation vs Avg Skill %",
                              trendline="ols" if len(cdf)>=3 else None)
            fig6.update_traces(textposition="top center")
            fig6.update_layout(plot_bgcolor="white",paper_bgcolor="white")
            st.plotly_chart(fig6, use_container_width=True)

    plateau = stats[stats["medium_band"]>stats["high_band"]+stats["low_band"]]
    st.markdown("#### 📉 Skill Plateau Detection")
    if not plateau.empty:
        for _,r in plateau.iterrows():
            risk(f"Plateau: {r['school']}",
                 f"{int(r['medium_band'])}/{int(r['student_count'])} students stuck in Medium band.")
    else:
        st.success("No significant skill plateau patterns detected.")

    avg_all = df["score_pct"].mean()
    under   = stats[stats["avg_pct"]<avg_all*0.85]
    if not under.empty:
        st.markdown(f"#### 🔴 Underperforming Schools (>15% below avg {avg_all:.1f}%)")
        for _,r in under.iterrows():
            risk(r["school"], f"Avg: {r['avg_pct']:.1f}% vs system avg {avg_all:.1f}%")

    st.markdown("#### 💡 Instructional Recommendations")
    wk = sorted_cats[-1] if sorted_cats else None
    insight(f"📚 Targeted {CAT_LABELS.get(wk[0],wk[0]) if wk else 'Skill'} Workshops",
            f"Lowest category avg: {wk[1]:.2f}/5.0. Dedicate 30-min modules to this skill."
            if wk else "Run targeted skill workshops.")
    insight("🔄 Differentiated Instruction for Low-Band Students",
            f"{low_n} students ({low_n/max(len(df),1)*100:.1f}%) score below 40%. "
            "Teachers need individual rubric feedback sessions before next assessment cycle.")
    insight("📊 Bi-Weekly Score Tracking",
            "Introduce bi-weekly score sheets shared with teachers to create accountability "
            "and flag stagnating students early.")

    av = [c for c in ["school","cohort","student_name"]+SCORE_COLS+
          ["total_score","score_pct","skill_band"] if c in df.columns]
    disp2 = df[av].copy()
    disp2.columns = [c.replace("_"," ").title() for c in av]
    st.dataframe(disp2, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download Report CSV",
                       disp2.to_csv(index=False).encode(),
                       "framework2_teacher_scores.csv","text/csv")
    gsheet_push_widget(disp2, "Framework2_TeacherScores")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FRAMEWORK 3 — STUDENT SURVEY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐  Framework 3 — Student Survey":
    st.markdown("""<div class="main-header">
        <h1>⭐ Framework 3 — Student Survey</h1>
        <p>Teacher effectiveness · Self-growth · Anomaly detection · Perception gaps</p>
    </div>""", unsafe_allow_html=True)

    if "student_survey_df" in st.session_state:
        df = st.session_state["student_survey_df"]
    else:
        df = load_data("student_survey")
        if df is not None:
            st.session_state["student_survey_df"] = df

    if df is None or (isinstance(df,pd.DataFrame) and df.empty):
        st.info("No survey data found. Upload a Student Survey CSV first.")
        st.stop()

    if "teacher_effectiveness_avg" not in df.columns:
        df = run_survey_model(df)

    # 70% coverage check (survey = responses per school vs roster)
    roster = load_roster()
    if roster is not None:
        cov = check_coverage(df, roster, name_col="school")
        # For survey: compare response count per school vs total students per school
        resp_per_school  = df.groupby("school").size().reset_index(name="responses")
        ros_per_school   = roster.groupby("school")["student_name"].nunique().reset_index()
        ros_per_school.columns = ["school","total_students"]
        cov_survey = ros_per_school.merge(resp_per_school,on="school",how="left").fillna(0)
        cov_survey["responses"]    = cov_survey["responses"].astype(int)
        cov_survey["coverage_pct"] = (cov_survey["responses"]/cov_survey["total_students"].clip(1)*100).round(1)
        cov_survey["passes_70pct"] = cov_survey["coverage_pct"]>=70
        failing = cov_survey[~cov_survey["passes_70pct"]]
        if failing.empty:
            st.success("✅ All schools meet the 70% survey response rate requirement.")
        else:
            st.markdown("""<div class="warn-box">
                <strong>⚠️ Survey Coverage Warning</strong><br>
                The following schools have fewer than 70% survey response rate:
            </div>""", unsafe_allow_html=True)
            dc = cov_survey[["school","total_students","responses","coverage_pct","passes_70pct"]].copy()
            dc.columns = ["School","Total Students","Responses","Coverage %","Meets 70%"]
            dc["Meets 70%"] = dc["Meets 70%"].map({True:"✅ Yes",False:"❌ No"})
            st.dataframe(dc, use_container_width=True, hide_index=True)
    else:
        st.caption("ℹ️ Upload student roster to enable 70% survey response rate check.")

    st.markdown("---")
    stats = school_survey_stats(df)
    anom  = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0

    # ── Section 1 ─────────────────────────────────────────────────────────────
    st.markdown("## 📌 Section 1 — Teacher Effectiveness Scores per School")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric("Teacher Effectiveness", f"{df['teacher_effectiveness_avg'].mean():.2f}/5")
    with c2: metric("Self-Growth Score",      f"{df['self_growth_avg'].mean():.2f}/5")
    with c3: metric("Anomalous Responses",    anom)
    with c4: metric("Total Responses",        f"{len(df):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    fig1 = px.bar(stats.sort_values("avg_teacher_score",ascending=False),
                  x="school",y="avg_teacher_score",color="avg_teacher_score",text="avg_teacher_score",
                  color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                  title="Average Teacher Effectiveness Score per School (1–5)")
    fig1.update_traces(texttemplate='%{text:.2f}',textposition='outside')
    fig1.add_hline(y=3.5,line_dash="dash",line_color="#302b63",annotation_text="Threshold (3.5)")
    fig1.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                       coloraxis_showscale=False,xaxis_tickangle=-20,yaxis_range=[0,5.5])
    st.plotly_chart(fig1, use_container_width=True)

    tqa = {f"Q{i+1}":df[c].mean() for i,c in enumerate(TEACHER_COLS) if c in df.columns}
    if tqa:
        fig2 = px.bar(pd.DataFrame(list(tqa.items()),columns=["Q","Avg"]),
                      x="Q",y="Avg",color="Avg",text="Avg",
                      color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                      title="Teacher Effectiveness — Sub-Question Averages")
        fig2.update_traces(texttemplate='%{text:.2f}',textposition='outside')
        fig2.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                           coloraxis_showscale=False,yaxis_range=[0,5])
        st.plotly_chart(fig2, use_container_width=True)

    # ── Section 2 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 2 — Self-Perceived Growth per School")
    c1,c2 = st.columns(2)
    with c1:
        fig3 = px.bar(stats.sort_values("avg_growth_score",ascending=False),
                      x="school",y="avg_growth_score",color="avg_growth_score",text="avg_growth_score",
                      color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                      title="Average Self-Growth Score per School")
        fig3.update_traces(texttemplate='%{text:.2f}',textposition='outside')
        fig3.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                           coloraxis_showscale=False,xaxis_tickangle=-20,yaxis_range=[0,5])
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        gqa = {f"Q{i+1}":df[c].mean() for i,c in enumerate(GROWTH_COLS) if c in df.columns}
        if gqa:
            fig4 = px.bar(pd.DataFrame(list(gqa.items()),columns=["Q","Avg"]),
                          x="Q",y="Avg",color="Avg",text="Avg",
                          color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                          title="Self-Growth — Sub-Question Averages")
            fig4.update_traces(texttemplate='%{text:.2f}',textposition='outside')
            fig4.update_layout(plot_bgcolor="white",paper_bgcolor="white",
                               coloraxis_showscale=False,yaxis_range=[0,5])
            st.plotly_chart(fig4, use_container_width=True)

    # ── Section 3 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 3 — Cross-Framework Comparison & Anomaly Detection")
    merged = stats.copy()

    if "teacher_scores_df" in st.session_state:
        sd  = st.session_state["teacher_scores_df"]
        skg = sd.groupby("school")["score_pct"].mean().reset_index()
        skg.columns = ["school","avg_skill_pct"]
        merged = merged.merge(skg,on="school",how="left")

        fig5 = px.scatter(merged.dropna(subset=["avg_skill_pct"]),
                          x="avg_teacher_score",y="avg_skill_pct",text="school",
                          size="response_count",color="avg_growth_score",
                          color_continuous_scale=["#f5576c","#ffc107","#43e97b"],
                          title="Teacher Effectiveness Rating vs Actual Skill Score (%)")
        fig5.update_traces(textposition="top center")
        fig5.update_layout(plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig5, use_container_width=True)

        if "avg_skill_pct" in merged.columns:
            ar   = merged["avg_teacher_score"].mean()
            as_  = merged["avg_skill_pct"].mean()
            hrlp = merged[(merged["avg_teacher_score"]>=ar)&(merged["avg_skill_pct"]<as_)]
            lrhp = merged[(merged["avg_teacher_score"]<ar) &(merged["avg_skill_pct"]>=as_)]
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**⚠️ High-Rated Teacher, Low-Performing Students**")
                if not hrlp.empty:
                    for _,r in hrlp.iterrows():
                        risk(r["school"],f"Rating: {r['avg_teacher_score']:.2f}/5 | Skill: {r['avg_skill_pct']:.1f}%")
                else: st.success("None detected.")
            with c2:
                st.markdown("**🌟 Low-Rated Teacher, High-Performing Students**")
                if not lrhp.empty:
                    for _,r in lrhp.iterrows():
                        insight(r["school"],f"Rating: {r['avg_teacher_score']:.2f}/5 | Skill: {r['avg_skill_pct']:.1f}%")
                else: st.success("None detected.")

            merged["growth_pct"] = merged["avg_growth_score"]/5*100
            fig6 = go.Figure([
                go.Bar(name="Perceived Growth (scaled %)",x=merged["school"],
                       y=merged["growth_pct"],marker_color="#667eea"),
                go.Bar(name="Measured Skill %",x=merged["school"],
                       y=merged["avg_skill_pct"],marker_color="#43e97b"),
            ])
            fig6.update_layout(barmode="group",title="Perceived vs Measured Growth per School",
                               plot_bgcolor="white",paper_bgcolor="white",xaxis_tickangle=-20)
            st.plotly_chart(fig6, use_container_width=True)
    else:
        st.info("Upload Framework 2 teacher scores to enable cross-framework comparison.")

    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"]==True]
        if not anomalies.empty:
            st.markdown("#### 🔬 Anomalous Survey Responses (Isolation Forest)")
            ac = [c for c in ["school","cohort","teacher_effectiveness_avg","self_growth_avg"]
                  if c in anomalies.columns]
            st.dataframe(anomalies[ac], use_container_width=True, hide_index=True)

    # ── Section 4 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📌 Section 4 — Perception Gaps & Recommendations")

    disp = stats.copy()
    if "avg_skill_pct" in merged.columns:
        disp = disp.merge(merged[["school","avg_skill_pct"]],on="school",how="left")
    cols_disp = ["School","Teacher Eff.","Self-Growth","Responses"] + \
                (["Skill %"] if "avg_skill_pct" in merged.columns else [])
    disp.columns = cols_disp
    st.dataframe(disp, use_container_width=True, hide_index=True)

    ot  = df["teacher_effectiveness_avg"].mean()
    og  = df["self_growth_avg"].mean()
    gap = abs(ot-og)
    if gap > 0.5:
        risk("🔴 System-Level Perception Gap Detected",
             f"Teacher Effectiveness: {ot:.2f} | Self-Growth: {og:.2f} | Gap: {gap:.2f}. "
             + ("Students over-credit teaching relative to perceived growth."
                if ot>og else "Students credit personal growth more than teaching quality."))
    else:
        rec("✅ Perception Well-Aligned",f"Gap: {gap:.2f} — teacher ratings and self-growth scores are consistent.")

    insight("🔎 Audit High-Rating Schools with Low Outcomes",
            "Schools with effectiveness ≥3.5 but skill below average may reflect 'feel-good' teaching. "
            "Schedule rubric-aligned observation visits.")
    insight("📢 Close Perception Gap with Structured Reflection",
            f"Gap of {gap:.2f} between teacher rating ({ot:.2f}) and self-growth ({og:.2f}). "
            "Introduce session-end prompts: 'What debate skill did I practice today?'")
    insight("🔄 Re-Survey After Intervention",
            "Run survey at program start and end. Compare pre/post scores against measured skill changes.")

    dc = [c for c in ["school","cohort","teacher_effectiveness_avg","self_growth_avg","is_anomaly"]
          if c in df.columns]
    st.download_button("⬇️ Download Survey Report CSV",
                       df[dc].to_csv(index=False).encode(),
                       "framework3_student_survey.csv","text/csv")
    gsheet_push_widget(df[dc], "Framework3_StudentSurvey")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PDF REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️  Manage Stored Data":
    st.markdown("""<div class="main-header">
        <h1>🗄️ Manage Stored Data</h1>
        <p>View, download, or delete permanently stored datasets from this computer</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    All data you upload is saved permanently in the **`data/`** folder inside the project folder.
    It stays here until you delete it. You can also download any dataset as CSV.
    """)

    datasets = {
        "student_roster":   "👥 Student Roster",
        "participation":    "📊 Participation (raw)",
        "ml_participation": "📊 Participation (ML results)",
        "teacher_scores":   "📝 Teacher Scores",
        "student_survey":   "⭐ Student Survey",
    }

    found_any = False
    for key, label in datasets.items():
        df_stored = load_data(key)
        if df_stored is not None:
            found_any = True
            with st.expander(f"{label} — {len(df_stored):,} rows", expanded=False):
                st.dataframe(df_stored.head(20), use_container_width=True, hide_index=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        f"⬇️ Download {label}",
                        df_stored.to_csv(index=False).encode(),
                        f"{key}.csv", "text/csv",
                        key=f"dl_{key}"
                    )
                with c2:
                    if st.button(f"🗑️ Delete {label}", key=f"del_{key}"):
                        delete_data(key)
                        # Clear session state too
                        ss_map = {
                            "student_roster":   "student_roster_df",
                            "participation":    "participation_raw",
                            "ml_participation": "participation_agg",
                            "teacher_scores":   "teacher_scores_df",
                            "student_survey":   "student_survey_df",
                        }
                        sk = ss_map.get(key)
                        if sk and sk in st.session_state:
                            del st.session_state[sk]
                        st.success(f"✅ {label} deleted.")
                        st.rerun()

    if not found_any:
        st.info("No data stored yet. Upload CSVs from the **📤 Upload CSV Data** page.")

    st.markdown("---")
    st.markdown("#### 📁 Storage Location")
    st.code(DATA_DIR)
    st.caption("This is where all your data files are saved on this computer.")

elif page == "📄  PDF Reports":
    st.markdown("""<div class="main-header">
        <h1>📄 PDF Reports</h1>
        <p>Printable debate skills test · Teacher scoring rubric · Auto-generated analytical briefs</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 Printable Debate Skills Test + Teacher Rubric")
    st.markdown("""
    This generates the **official ToR-compliant assessment document** for teachers containing:
    - The debate motion
    - Task 1: Written Argument (150 words / 10 min)
    - Task 2: Written Refutation (100 words / 7 min)
    - The 4-category scoring rubric (1–5 scale) with descriptors for each level
    - Student scoring sheet (name, school, cohort, total /20, %)
    """)
    if st.button("⬇️ Generate & Download Rubric PDF", use_container_width=True):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_rubric_pdf()
                st.download_button(
                    label="📥 Click to Download Rubric PDF",
                    data=pdf_bytes,
                    file_name="DIC_Teacher_Assessment_Rubric.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("✅ PDF generated! Click the button above to download.")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")
                import traceback; st.code(traceback.format_exc())

    st.markdown("---")
    st.markdown("### 📊 Auto-Generated Analytical Insight Briefs")
    st.markdown("These PDFs compile the data-driven findings into formatted analytical reports.")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("**Framework 1 — Participation Brief (3 pages)**")
        if st.button("Generate Participation Brief PDF", use_container_width=True):
            if "participation_agg" in st.session_state:
                agg   = st.session_state["participation_agg"]
                pstats = school_participation_stats(agg)
                ov    = agg["participation_score"].mean()
                rs    = pstats.sort_values("avg_score").head(2)["school"].tolist()
                hs    = pstats.sort_values("avg_score").tail(2)["school"].tolist()
                top_  = agg[agg["cohort_flag"]=="Top 10%"]
                bot_  = agg[agg["cohort_flag"]=="Bottom 10%"]
                sections = [
                    ("School-Level Participation Averages", [
                        f"Overall average participation score: {ov:.1f}/100 across {len(agg)} students in {agg['school'].nunique()} schools.",
                        "School averages range from " +
                        f"{pstats['avg_score'].min():.1f} ({pstats.sort_values('avg_score').iloc[0]['school']}) to " +
                        f"{pstats['avg_score'].max():.1f} ({pstats.sort_values('avg_score').iloc[-1]['school']}).",
                    ]),
                    ("Participation Variance and Top/Bottom 10%", [
                        f"Top 10% participants ({len(top_)} students): average score {top_['participation_score'].mean():.1f}, "
                        f"average attendance {top_['attendance_rate'].mean():.1f}%."
                        if not top_.empty else "Insufficient data for top 10% analysis.",
                        f"Bottom 10% participants ({len(bot_)} students): average score {bot_['participation_score'].mean():.1f}, "
                        f"average attendance {bot_['attendance_rate'].mean():.1f}%."
                        if not bot_.empty else "Insufficient data for bottom 10% analysis.",
                        "Inequality Index (std/mean) is highest at " +
                        f"{pstats.sort_values('inequality_index').iloc[-1]['school']} ({pstats['inequality_index'].max():.3f}), "
                        "indicating the widest spread of participation within a single school.",
                    ]),
                    ("Engagement-Risk Schools and Recommendations", [
                        f"Engagement-risk schools (bottom 2): {', '.join(rs)}. "
                        "Both exhibit low average scores and high proportions of low-engaged students.",
                        f"High-engagement schools (top 2): {', '.join(hs)}. "
                        "These schools show consistent attendance and active speaking-turn uptake.",
                        "Recommendation 1: Implement mandatory speaking-turn minimums (min. 2 turns/session) at risk schools.",
                        "Recommendation 2: Deploy a weekly attendance SMS follow-up protocol for bottom 10% students.",
                        "Recommendation 3: Establish a cross-school peer-mentoring programme pairing top-10% students "
                        "from high-engagement schools with bottom-10% participants from risk schools.",
                    ]),
                ]
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_bytes = generate_insight_pdf(
                            "Framework 1 — Student Participation Analytical Brief\nDebate is Cool · Civic Space Nepal",
                            sections
                        )
                        st.download_button("📥 Download Participation Brief PDF", pdf_bytes,
                                           "Framework1_Participation_Brief.pdf","application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF error: {e}")
            else:
                st.warning("Upload and process Participation data first.")

    with c2:
        st.markdown("**Framework 2 — Teacher Assessment Report (4 pages)**")
        if st.button("Generate Assessment Report PDF", use_container_width=True):
            if "teacher_scores_df" in st.session_state:
                df2    = st.session_state["teacher_scores_df"]
                stats2 = school_score_stats(df2)
                catavg2= category_stats(df2)
                sc     = sorted(catavg2.items(),key=lambda x:x[1])
                ov2    = df2["score_pct"].mean()
                sections = [
                    ("School-Level Skill Comparison", [
                        f"Overall average skill score: {ov2:.1f}% across {len(df2)} students.",
                        f"Highest scoring school: {stats2.sort_values('avg_pct').iloc[-1]['school']} "
                        f"({stats2['avg_pct'].max():.1f}%).",
                        f"Lowest scoring school: {stats2.sort_values('avg_pct').iloc[0]['school']} "
                        f"({stats2['avg_pct'].min():.1f}%).",
                    ]),
                    ("Skill Distribution — High / Medium / Low", [
                        f"High band (>=70%): {(df2['skill_band']=='High').sum()} students "
                        f"({(df2['skill_band']=='High').mean()*100:.1f}%).",
                        f"Medium band (40-69%): {(df2['skill_band']=='Medium').sum()} students.",
                        f"Low band (<40%): {(df2['skill_band']=='Low').sum()} students.",
                    ]),
                    ("Category Analysis — Strongest and Weakest", [
                        f"Strongest category across all schools: {CAT_LABELS.get(sc[-1][0],sc[-1][0])} "
                        f"(avg {sc[-1][1]:.2f}/5.0).",
                        f"Weakest category across all schools: {CAT_LABELS.get(sc[0][0],sc[0][0])} "
                        f"(avg {sc[0][1]:.2f}/5.0). This area requires focused instructional attention.",
                    ]),
                    ("Recommendations", [
                        f"Recommendation 1: Conduct targeted workshops on {CAT_LABELS.get(sc[0][0],sc[0][0])} "
                        "for all schools, beginning with the two lowest-scoring schools.",
                        "Recommendation 2: Introduce differentiated instruction protocols for low-band students, "
                        "including individual teacher feedback sessions.",
                        "Recommendation 3: Implement bi-weekly score tracking shared with teachers "
                        "to create accountability and flag plateau patterns early.",
                    ]),
                ]
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_bytes = generate_insight_pdf(
                            "Framework 2 — Teacher Assessment Analytical Report\nDebate is Cool · Civic Space Nepal",
                            sections
                        )
                        st.download_button("📥 Download Assessment Report PDF", pdf_bytes,
                                           "Framework2_Assessment_Report.pdf","application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF error: {e}")
            else:
                st.warning("Upload and process Teacher Score data first.")

    with c3:
        st.markdown("**Framework 3 — Student Survey Report (4 pages)**")
        if st.button("Generate Survey Report PDF", use_container_width=True):
            if "student_survey_df" in st.session_state:
                df3    = st.session_state["student_survey_df"]
                stats3 = school_survey_stats(df3)
                ot3    = df3["teacher_effectiveness_avg"].mean()
                og3    = df3["self_growth_avg"].mean()
                gap3   = abs(ot3-og3)
                sections = [
                    ("Teacher Effectiveness Scores per School", [
                        f"System-wide average teacher effectiveness: {ot3:.2f}/5.0.",
                        f"Highest rated: {stats3.sort_values('avg_teacher_score').iloc[-1]['school']} "
                        f"({stats3['avg_teacher_score'].max():.2f}/5).",
                        f"Lowest rated: {stats3.sort_values('avg_teacher_score').iloc[0]['school']} "
                        f"({stats3['avg_teacher_score'].min():.2f}/5).",
                    ]),
                    ("Self-Perceived Growth per School", [
                        f"System-wide average self-growth score: {og3:.2f}/5.0.",
                        f"Schools above 3.5 self-growth: "
                        f"{(stats3['avg_growth_score']>=3.5).sum()} of {len(stats3)}.",
                    ]),
                    ("Perception Gaps and Anomaly Detection", [
                        f"System-level perception gap (teacher rating vs self-growth): {gap3:.2f}.",
                        f"{'A gap >0.5 indicates a meaningful disconnect between how students perceive teaching and their own growth.' if gap3>0.5 else 'Perception is well-aligned across the system.'}",
                        f"Anomalous responses detected by Isolation Forest: "
                        f"{int(df3['is_anomaly'].sum()) if 'is_anomaly' in df3.columns else 0}.",
                    ]),
                    ("Recommendations", [
                        "Recommendation 1: Audit schools with high teacher ratings but low skill outcomes "
                        "through structured observation visits.",
                        f"Recommendation 2: Introduce session-end reflection prompts to close the {gap3:.2f}-point "
                        "perception gap between teacher quality and student growth attribution.",
                        "Recommendation 3: Administer this survey at the start and end of each program phase "
                        "to track whether perceptions and performance move together over time.",
                    ]),
                ]
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_bytes = generate_insight_pdf(
                            "Framework 3 — Student Survey Analytical Report\nDebate is Cool · Civic Space Nepal",
                            sections
                        )
                        st.download_button("📥 Download Survey Report PDF", pdf_bytes,
                                           "Framework3_Survey_Report.pdf","application/pdf",
                                           use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF error: {e}")
            else:
                st.warning("Upload and process Survey data first.")
