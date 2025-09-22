#!/usr/bin/env python3
import os, json, argparse, pathlib, datetime as dt, sys, traceback
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from openai import OpenAI, APIError, APIStatusError, APIConnectionError

# ---------------- Setup ----------------
MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OUT_DIR = pathlib.Path("outputs")
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today()

# ---------------- Error Reader ----------------
def log_error(e, fatal=False):
    print("\n[ERROR CAUGHT]" if not fatal else "\n[FATAL ERROR]")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {e}")
    print("[TRACEBACK]")
    traceback.print_exc(file=sys.stdout)
    print("[/TRACEBACK]\n")

def safe_run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log_error(e)
        return {"rows": []}  # fallback to empty rows

# ---------------- Prompt Builders ----------------
def prompt_new_artist_signing(subject: str) -> str:
    return f"""
SHEET: New Artist Signing
COLUMNS: ["Task","Category","Contact","Email/Phone","Requirements","Due Date","Status","Notes"]

Rules:
- Georgia-first (Atlanta, Athens, Savannah, Macon, Augusta).
- Status default "Not Started".
- Contact is a role type (A&R Rep, Entertainment Lawyer, Brand Designer).
- Include micro-steps like "Collect high-res photos".
Strict JSON: {{"rows": [{{"Task":"...","Category":"...","Contact":"...","Email/Phone":"","Requirements":"...","Due Date":"","Status":"Not Started","Notes":"..."}}]}}
"""

def prompt_festivals(subject: str) -> str:
    return f"""
SHEET: Festival & Live Planning
COLUMNS: ["Festival/Event","Category","Contact","Email/Phone","City/State","Application Link","Requirements","Due Date","Status","Notes"]

Rules:
- Group by Festival, Venue, Karaoke/Open Mic, Showcase.
- Georgia + Southeast events.
Strict JSON with rows.
"""

def prompt_business_strategy(subject: str) -> str:
    return f"""
SHEET: Business & Strategy
COLUMNS: ["Project","Category","Contact","Link","Requirements","Due Date","Status","Notes"]

Include merch launch steps, Georgia brand partnerships, market research.
Strict JSON with rows.
"""

def prompt_contacts(subject: str) -> str:
    return f"""
SHEET: Contacts Directory
COLUMNS: ["Name","Role","Organization","Email","Phone","City/State","Notes"]

Georgia-first (A&Rs, venue managers, festival coordinators, producers).
Leave blank if unknown; use Notes: "find on site".
Strict JSON with rows.
"""

def prompt_content_calendar(subject: str) -> str:
    three_months = (TODAY + relativedelta(months=+3)).isoformat()
    return f"""
SHEET: Content Calendar
COLUMNS: ["Date","Platform","Content Type","Notes","Status"]

3-month weekly plan from {TODAY.isoformat()} to {three_months}.
Status default "Not Started".
Strict JSON with rows.
"""

def prompt_general_calendar(subject: str) -> str:
    six_months = (TODAY + relativedelta(months=+6)).isoformat()
    return f"""
SHEET: General Calendar
COLUMNS: ["Date","Event","Type","Location","Notes"]

6-month Georgia-first events (deadlines, showcases, gigs).
Strict JSON with rows.
"""

# ---------------- Fallback Seeds ----------------
SEED_FEST_VALS = [
    {"Festival/Event":"A3C Festival & Conference","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Atlanta, GA","Application Link":"","Requirements":"EPK; live video; links","Due Date":"","Status":"Not Started","Notes":"find booking contact on website"},
    {"Festival/Event":"Savannah Stopover","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Savannah, GA","Application Link":"","Requirements":"Press kit; 2-3 live clips","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"AthFest Music & Arts Festival","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Athens, GA","Application Link":"","Requirements":"EPK; stage plot; links","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"Bragg Jam","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Macon, GA","Application Link":"","Requirements":"EPK; live performance video","Due Date":"","Status":"Not Started","Notes":""},
]

# ---------------- LLM Call ----------------
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
        instructions="Return strictly valid JSON with only spreadsheet rows."
    )
    return resp.output_text

def llm_json_safe(prompt: str, model: str, seed=None) -> dict:
    try:
        raw = llm_raw(prompt, model)
        return json.loads(raw)
    except Exception as e:
        log_error(e)
        return {"rows": seed or []}

# ---------------- Helpers ----------------
def rows_or_seed(d: dict, seed=None):
    rows = d.get("rows", [])
    return rows if rows else (seed or [])

def enforce_defaults(df: pd.DataFrame, status_col="Status", due_col="Due Date"):
    if status_col in df.columns:
        df[status_col] = df[status_col].fillna("Not Started").replace("", "Not Started")
    if due_col in df.columns:
        df[due_col] = df[due_col].fillna("")
    return df

def ensure_headers(df: pd.DataFrame, headers: list) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=headers)
    return df

# ---------------- Main Generate ----------------
def generate(subject: str, model: str = MODEL_DEFAULT):
    # Each sheet
    nas = safe_run(llm_json_safe, prompt_new_artist_signing(subject), model)
    nas_df = enforce_defaults(pd.DataFrame(rows_or_seed(nas)))
    nas_df = ensure_headers(nas_df, ["Task","Category","Contact","Email/Phone","Requirements","Due Date","Status","Notes"])

    flp = safe_run(llm_json_safe, prompt_festivals(subject), model, seed=SEED_FEST_VALS)
    flp_df = enforce_defaults(pd.DataFrame(rows_or_seed(flp, seed=SEED_FEST_VALS)))
    flp_df = ensure_headers(flp_df, ["Festival/Event","Category","Contact","Email/Phone","City/State","Application Link","Requirements","Due Date","Status","Notes"])

    bas = safe_run(llm_json_safe, prompt_business_strategy(subject), model)
    bas_df = enforce_defaults(pd.DataFrame(rows_or_seed(bas)))
    bas_df = ensure_headers(bas_df, ["Project","Category","Contact","Link","Requirements","Due Date","Status","Notes"])

    contacts = safe_run(llm_json_safe, prompt_contacts(subject), model)
    contacts_df = pd.DataFrame(rows_or_seed(contacts))
    contacts_df = ensure_headers(contacts_df, ["Name","Role","Organization","Email","Phone","City/State","Notes"])

    cc = safe_run(llm_json_safe, prompt_content_calendar(subject), model)
    cc_df = enforce_defaults(pd.DataFrame(rows_or_seed(cc)))
    cc_df = ensure_headers(cc_df, ["Date","Platform","Content Type","Notes","Status"])

    gc = safe_run(llm_json_safe, prompt_general_calendar(subject), model)
    gc_df = pd.DataFrame(rows_or_seed(gc))
    gc_df = ensure_headers(gc_df, ["Date","Event","Type","Location","Notes"])

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
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

# ---------------- Entry ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()
    generate(args.subject, args.model)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(e, fatal=True)
        sys.exit(0)  # prevent exit code 1
