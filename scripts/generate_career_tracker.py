#!/usr/bin/env python3
import os, json, argparse, pathlib, datetime as dt
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd

# --- OpenAI client (Responses API) ---
# Docs: https://platform.openai.com/docs/api-reference/responses
from openai import OpenAI, APIError, APIStatusError, APIConnectionError

MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # set to gpt-5-mini if available to you
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

OUT_DIR = pathlib.Path("outputs")
CSV_DIR = OUT_DIR / "csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

TODAY = dt.date.today()

# --------------------------
# Prompt helpers (Georgia-first)
# --------------------------
def prompt_new_artist_signing(subject: str) -> str:
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: New Artist Signing (Georgia-first; Atlanta/Athens/Savannah/Macon/Augusta)
COLUMNS (exact keys): ["Task","Category","Contact","Email/Phone","Requirements","Due Date","Status","Notes"]

Rules:
- Status default "Not Started" if not provided.
- Due Date may be "" (empty) unless a concrete deadline is typical.
- Contact is a role type (e.g., "A&R Rep", "Entertainment Lawyer", "Brand Designer").
- Requirements are concrete materials (e.g., "2 live performance clips", "Press Kit PDF", "1-page EPK").
- Include micro-steps like "Collect high-res photos", "Write 150-word bio".
- Keep it specific to an Atlanta-based emerging R&B artist, with Georgia/Southeast networking pathways (A&R meetups, showcases, conferences).
- 18–28 rows is ideal.

Answer as strict JSON with this schema:
{{
  "rows": [
    {{"Task": "...", "Category": "...", "Contact": "...", "Email/Phone": "", "Requirements": "...", "Due Date": "", "Status": "Not Started", "Notes": "..."}}
  ]
}}
"""

def prompt_festivals(subject: str) -> str:
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: Festival & Live Planning (Georgia & Southeast U.S.)
COLUMNS: ["Festival/Event","Category","Contact","Email/Phone","City/State","Application Link","Requirements","Due Date","Status","Notes"]

Rules:
- Group items by Category in the data itself: Festival, Venue, Karaoke/Open Mic, Showcase.
- 20+ entries with Georgia-first; then nearby Southeast.
- Fill Requirements with concrete items (EPK, live video links, typical app fee if common).
- Use Notes for "find booking contact on website" if Email/Phone unknown.
- Due Date can be a month/season estimate if typical; otherwise "".
- Status default "Not Started".

Answer as strict JSON:
{{
  "rows": [
    {{
      "Festival/Event":"...", "Category":"Festival|Venue|Karaoke/Open Mic|Showcase",
      "Contact":"", "Email/Phone":"", "City/State":"City, ST",
      "Application Link":"", "Requirements":"...", "Due Date":"", "Status":"Not Started", "Notes":"..."
    }}
  ]
}}
"""

def prompt_business_strategy(subject: str) -> str:
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: Business & Strategy (Georgia markets)
COLUMNS: ["Project","Category","Contact","Link","Requirements","Due Date","Status","Notes"]

Include:
- A merch launch checklist tailored to Atlanta-based R&B fans.
- Georgia brand partnership ideas (fashion, coffee shops, lifestyle brands).
- Market research steps for Atlanta/Georgia audiences (venues, demographics, platforms).
- Practical Georgia-local distribution & PR steps.
- 12–18 rows.

Answer as strict JSON:
{{
  "rows": [
    {{
      "Project":"...", "Category":"Merch|Partnership|Research|PR|Distribution",
      "Contact":"", "Link":"", "Requirements":"...", "Due Date":"", "Status":"Not Started", "Notes":"..."
    }}
  ]
}}
"""

def prompt_contacts(subject: str) -> str:
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: Contacts Directory (Georgia-first roles)
COLUMNS: ["Name","Role","Organization","Email","Phone","City/State","Notes"]

Rules:
- Provide sample contacts and role types relevant to Atlanta/Georgia: A&Rs, venue managers, festival coordinators, producers.
- If exact Email/Phone is unknown, leave blank and add "find on site" in Notes.
- 15–25 entries.

Strict JSON:
{{ "rows": [ {{"Name":"...", "Role":"...", "Organization":"...", "Email":"", "Phone":"", "City/State":"...", "Notes":"..."}} ] }}
"""

def prompt_content_calendar(subject: str) -> str:
    three_months = (TODAY + relativedelta(months=+3)).isoformat()
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: Content Calendar (3 months)
COLUMNS: ["Date","Platform","Content Type","Notes","Status"]

Rules:
- Build a 3-month weekly plan starting from {TODAY.isoformat()} to approx {three_months}.
- Georgia-aware cadence: monthly local event highlight (e.g., open mic, venue show, festival push).
- Platform examples: Instagram, TikTok, YouTube Shorts, Email list.
- Status default "Not Started".

Strict JSON:
{{ "rows": [ {{"Date":"YYYY-MM-DD","Platform":"...","Content Type":"...","Notes":"...","Status":"Not Started"}} ] }}
"""

def prompt_general_calendar(subject: str) -> str:
    six_months = (TODAY + relativedelta(months=+6)).isoformat()
    return f"""
You are filling a spreadsheet for "{subject}".

SHEET: General Calendar (6 months; Georgia-first)
COLUMNS: ["Date","Event","Type","Location","Notes"]

Rules:
- Deadlines and major events next 6 months (festival due dates, showcases, networking).
- Georgia first; include Southeast if helpful.
- Types might be "Deadline", "Showcase", "Gig", "Networking".

Strict JSON:
{{ "rows": [ {{"Date":"YYYY-MM-DD","Event":"...","Type":"...","Location":"City, ST","Notes":"..."}} ] }}
"""

# --------------------------
# Call OpenAI with retries
# --------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=12),
    retry=retry_if_exception_type((APIError, APIStatusError, APIConnectionError)),
)
def llm_json(prompt: str, model: str = MODEL_DEFAULT) -> dict:
    """
    Calls Responses API and returns parsed JSON with a top-level {"rows": [...]}
    """
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type":"input_text","text": prompt}]}],
        response_format={"type": "json_object"},
        instructions="Return strictly valid JSON for spreadsheet ingestion. Do not include commentary."
    )
    # The SDK exposes output_text for JSON too
    txt = resp.output_text
    return json.loads(txt)

# --------------------------
# Minimal Georgia seed (fallback if API ever returns empty)
# --------------------------
SEED_FEST_VALS = [
    # Festival examples (Georgia-first)
    {"Festival/Event":"A3C Festival & Conference","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Atlanta, GA","Application Link":"","Requirements":"EPK; live video; links","Due Date":"","Status":"Not Started","Notes":"find booking contact on website"},
    {"Festival/Event":"Savannah Stopover","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Savannah, GA","Application Link":"","Requirements":"Press kit; 2-3 live clips","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"AthFest Music & Arts Festival","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Athens, GA","Application Link":"","Requirements":"EPK; stage plot; links","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"Bragg Jam","Category":"Festival","Contact":"","Email/Phone":"","City/State":"Macon, GA","Application Link":"","Requirements":"EPK; live performance video","Due Date":"","Status":"Not Started","Notes":""},
    # Venues / Open Mics (Atlanta-first)
    {"Festival/Event":"Eddie's Attic (Open Mic)","Category":"Karaoke/Open Mic","Contact":"","Email/Phone":"","City/State":"Decatur, GA","Application Link":"","Requirements":"2 live clips; short bio","Due Date":"","Status":"Not Started","Notes":"find booking contact on website"},
    {"Festival/Event":"Aisle 5","Category":"Venue","Contact":"","Email/Phone":"","City/State":"Atlanta, GA","Application Link":"","Requirements":"EPK; links; availability","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"Smith's Olde Bar","Category":"Venue","Contact":"","Email/Phone":"","City/State":"Atlanta, GA","Application Link":"","Requirements":"EPK; live video","Due Date":"","Status":"Not Started","Notes":""},
    {"Festival/Event":"40 Watt Club","Category":"Venue","Contact":"","Email/Phone":"","City/State":"Athens, GA","Application Link":"","Requirements":"Press kit; links","Due Date":"","Status":"Not Started","Notes":""},
]

# --------------------------
# Build each sheet via LLM
# --------------------------
def rows_or_seed(d: dict, key="rows", seed=None):
    if not isinstance(d, dict): return seed or []
    rows = d.get(key, [])
    if isinstance(rows, list) and rows: return rows
    return seed or []

def enforce_defaults(df: pd.DataFrame, status_col="Status", due_col="Due Date"):
    if status_col in df.columns:
        df[status_col] = df[status_col].fillna("Not Started").replace("", "Not Started")
    if due_col in df.columns:
        df[due_col] = df[due_col].fillna("")
    return df

def generate(subject: str, model: str = MODEL_DEFAULT):
    # New Artist Signing
    nas = llm_json(prompt_new_artist_signing(subject), model)
    nas_df = pd.DataFrame(rows_or_seed(nas))
    nas_df = enforce_defaults(nas_df)

    # Festival & Live Planning
    flp = llm_json(prompt_festivals(subject), model)
    flp_rows = rows_or_seed(flp, seed=SEED_FEST_VALS)
    flp_df = pd.DataFrame(flp_rows)
    flp_df = enforce_defaults(flp_df)

    # Business & Strategy
    bas = llm_json(prompt_business_strategy(subject), model)
    bas_df = pd.DataFrame(rows_or_seed(bas))
    bas_df = enforce_defaults(bas_df)

    # Optional: Contacts Directory
    contacts = llm_json(prompt_contacts(subject), model)
    contacts_df = pd.DataFrame(rows_or_seed(contacts))

    # Optional: Content Calendar
    cc = llm_json(prompt_content_calendar(subject), model)
    cc_df = pd.DataFrame(rows_or_seed(cc))
    cc_df = enforce_defaults(cc_df)

    # Optional: General Calendar
    gc = llm_json(prompt_general_calendar(subject), model)
    gc_df = pd.DataFrame(rows_or_seed(gc))

    # --- write Excel
    xlsx_path = OUT_DIR / "career_action_tracker.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        nas_df.to_excel(xw, sheet_name="New Artist Signing", index=False)
        flp_df.to_excel(xw, sheet_name="Festival & Live Planning", index=False)
        bas_df.to_excel(xw, sheet_name="Business & Strategy", index=False)
        contacts_df.to_excel(xw, sheet_name="Contacts Directory", index=False)
        cc_df.to_excel(xw, sheet_name="Content Calendar", index=False)
        gc_df.to_excel(xw, sheet_name="General Calendar", index=False)

    # --- write CSVs (per sheet)
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
        "files": {
            "excel": str(xlsx_path),
            "csv_dir": str(CSV_DIR),
        }
    }
    (OUT_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help='e.g., "Gaby — Georgia-based R&B artist — 2025 roadmap"')
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()
    generate(args.subject, args.model)

if __name__ == "__main__":
    main()

