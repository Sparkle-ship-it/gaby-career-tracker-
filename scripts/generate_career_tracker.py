#!/usr/bin/env python3
import os, json, argparse, pathlib, datetime as dt
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from openai import OpenAI, APIError, APIStatusError, APIConnectionError

MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OUT_DIR = pathlib.Path("outputs")
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today()

# ---------- prompt builders (unchanged) ----------
def prompt_new_artist_signing(subject: str) -> str: ...
def prompt_festivals(subject: str) -> str: ...
def prompt_business_strategy(subject: str) -> str: ...
def prompt_contacts(subject: str) -> str: ...
def prompt_content_calendar(subject: str) -> str: ...
def prompt_general_calendar(subject: str) -> str: ...

# ---------- fallback seeds ----------
SEED_FEST_VALS = [
    {"Festival/Event":"A3C Festival & Conference","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Atlanta, GA","Application Link":"","Requirements":"EPK; live video; links","Due Date":"","Status":"Not Started","Notes":"find booking contact on website"},
    {"Festival/Event":"Savannah Stopover","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Savannah, GA","Application Link":"","Requirements":"Press kit; 2-3 live clips","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"AthFest Music & Arts Festival","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Athens, GA","Application Link":"","Requirements":"EPK; stage plot; links","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"Bragg Jam","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Macon, GA","Application Link":"","Requirements":"EPK; live performance video","Due Date":"","Status":"Not Started","Notes":""},
]

# ---------- safe LLM call ----------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type((APIError, APIStatusError, APIConnectionError)),
)
def llm_raw(prompt: str, model: str = MODEL_DEFAULT) -> str:
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type":"input_text","text": prompt}]}],
        response_format={"type": "json_object"},
        instructions="Return strictly valid JSON for spreadsheet ingestion. Do not include commentary."
    )
    return resp.output_text

def llm_json_safe(prompt: str, model: str, seed=None) -> dict:
    try:
        raw = llm_raw(prompt, model)
        return json.loads(raw)
    except Exception as e:
        print(f"[WARN] Falling back due to error: {e}")
        return {"rows": seed or []}

# ---------- helpers ----------
def rows_or_seed(d: dict, seed=None):
    rows = d.get("rows", [])
    return rows if rows else (seed or [])

def enforce_defaults(df: pd.DataFrame, status_col="Status", due_col="Due Date"):
    if status_col in df.columns:
        df[status_col] = df[status_col].fillna("Not Started").replace("", "Not Started")
    if due_col in df.columns:
        df[due_col] = df[due_col].fillna("")
    return df

# ---------- main generate ----------
def generate(subject: str, model: str = MODEL_DEFAULT):
    # New Artist Signing
    nas = llm_json_safe(prompt_new_artist_signing(subject), model)
    nas_df = pd.DataFrame(rows_or_seed(nas))
    nas_df = enforce_defaults(nas_df)

    # Festival & Live Planning
    flp = llm_json_safe(prompt_festivals(subject), model, seed=SEED_FEST_VALS)
    flp_df = pd.DataFrame(rows_or_seed(flp, seed=SEED_FEST_VALS))
    flp_df = enforce_defaults(flp_df)

    # Business & Strategy
    bas = llm_json_safe(prompt_business_strategy(subject), model)
    bas_df = pd.DataFrame(rows_or_seed(bas))
    bas_df = enforce_defaults(bas_df)

    # Contacts
    contacts = llm_json_safe(prompt_contacts(subject), model)
    contacts_df = pd.DataFrame(rows_or_seed(contacts))

    # Content Calendar
    cc = llm_json_safe(prompt_content_calendar(subject), model)
    cc_df = pd.DataFrame(rows_or_seed(cc))
    cc_df = enforce_defaults(cc_df)

    # General Calendar
    gc = llm_json_safe(prompt_general_calendar(subject), model)
    gc_df = pd.DataFrame(rows_or_seed(gc))

    # --- Excel ---
    xlsx_path = OUT_DIR / "career_action_tracker.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        nas_df.to_excel(xw, sheet_name="New Artist Signing", index=False)
        flp_df.to_excel(xw, sheet_name="Festival & Live Planning", index=False)
        bas_df.to_excel(xw, sheet_name="Business & Strategy", index=False)
        contacts_df.to_excel(xw, sheet_name="Contacts Directory", index=False)
        cc_df.to_excel(xw, sheet_name="Content Calendar", index=False)
        gc_df.to_excel(xw, sheet_name="General Calendar", index=False)

    # --- CSVs ---
    nas_df.to_csv(CSV_DIR / "new_artist_signing.csv", index=False)
    flp_df.to_csv(CSV_DIR / "festival_live_planning.csv", index=False)
    bas_df.to_csv(CSV_DIR / "business_strategy.csv", index=False)
    contacts_df.to_csv(CSV_DIR / "contacts_directory.csv", index=False)
    cc_df.to_csv(CSV_DIR / "content_calendar.csv", index=False)
    gc_df.to_csv(CSV_DIR / "general_calendar.csv", index=False)

    meta = {
        "subject": subject,
        "model": model,
        "generated_at": dt.datetime.now().isoformat(),
        "files": {"excel": str(xlsx_path), "csv_dir": str(CSV_DIR)}
    }
    (OUT_DIR / "run_meta.json").write_text(jso_
