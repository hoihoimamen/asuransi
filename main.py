"""
Health Benefit Utilization Dashboard
=====================================
Streamlit rebuild of the "Health Benefit Utilization Dashboard - Historical
Analysis 2023-2025" report.

Run with:
    streamlit run app.py
"""

import io
from google import genai
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# --------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Benefit Utilization Dashboard",
    page_icon="🩺",
    layout="wide",
)

NAVY = "#1b3a5c"
BLUE = "#2f6fb0"
GREEN = "#3fa66a"
PURPLE = "#7b5bb0"
ORANGE = "#e8871e"
RED = "#d64545"
YELLOW = "#e6b800"
LIGHT_BG = "#f4f7fb"

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 1.2rem; }}
        div[data-testid="stMetric"] {{
            background: white !important;
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}
        div[data-testid="stMetric"] * {{ color: #1a1a1a !important; }}
        div[data-testid="stMetricLabel"] * {{ color: #333 !important; }}
        .section-header {{
            background-color: {NAVY};
            color: white !important;
            padding: 8px 14px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 10px;
            letter-spacing: .3px;
        }}
        .section-header * {{ color: white !important; }}
        .kpi-card {{
            background: white;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            height: 100%;
        }}
        .kpi-title {{ font-size: 0.78rem; font-weight: 700; color: #333 !important; }}
        .kpi-value {{ font-size: 1.5rem; font-weight: 800; margin: 2px 0; color: #1a1a1a !important; }}
        .kpi-sub {{ font-size: 0.72rem; color: #777 !important; }}
        h1, h2, h3 {{ color: {NAVY}; }}
        div[data-testid="stDataFrame"] {{ background: white; border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_COLS = [
    "NIK", "Member Name", "Year", "Tipe Pegawai", "Transaction Type",
    "Benefit Plan", "Benefit Note", "Beneficiary", "Claim Amount", "Benefit Limit",
]

BENEFIT_NOTE_MAP = {
    "RJ-I": "Rawat Jalan",
    "RI-I": "Rawat Inap",
    "RG-I": "Rawat Gigi",
    "BT-I": "Alat Kacamata / Rehabilitasi",
}


# --------------------------------------------------------------------------
# DATA LOADING & CLEANING
# --------------------------------------------------------------------------
def parse_id_number(val):
    """Parse Indonesian-formatted numbers ('2.296.500,00' or ' 75.184.000,00 ')
    as well as plain numeric values, into a float. Returns np.nan on failure."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return np.nan
    # Remove currency symbols / spaces
    s = re.sub(r"[Rr]p\.?", "", s).strip()
    # Indonesian format: '.' thousands, ',' decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        # could be decimal comma only
        s = s.replace(",", ".")
    else:
        # only dots -> could be thousands separators
        # if more than one dot, treat as thousands separators
        if s.count(".") > 1:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


# @st.cache_data(show_spinner=False)
def load_csv_or_first_sheet(file_bytes, filename):
    """Used for simple single-table files (CSV, or a salary mapping xlsx)."""
    if filename.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes))
    return df


# @st.cache_data(show_spinner=False)
def list_excel_sheets(file_bytes):
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


# @st.cache_data(show_spinner=False)
def read_excel_sheet(file_bytes, sheet_name):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def find_sheet(sheet_names, keyword):
    """Return the first sheet name containing `keyword` (case-insensitive), or None."""
    for name in sheet_names:
        if keyword.lower() in name.lower():
            return name
    return None


def build_salary_df_from_master(master_df):
    """Convert a 'Master Employee Value'-style sheet (Pers.no. + Monthly Salary)
    into a standard NIK / Monthly Salary table. Returns None if columns don't match."""
    m = master_df.copy()
    m.columns = [str(c).strip() for c in m.columns]
    nik_col = "Pers.no." if "Pers.no." in m.columns else ("NIK" if "NIK" in m.columns else None)
    if nik_col is None or "Monthly Salary" not in m.columns:
        return None
    m["NIK"] = m[nik_col].astype(str).str.strip()
    m["Monthly Salary"] = m["Monthly Salary"].apply(parse_id_number)
    return m[["NIK", "Monthly Salary"]].dropna().drop_duplicates(subset="NIK")


BENEFIT_PREFIXES = ["RJ", "RI", "RG", "BT"]
TXN_TYPES = ["Provider", "Reimburse"]
PREFIX_NOTE = {
    "RJ": "Rawat Jalan",
    "RI": "Rawat Inap",
    "RG": "Rawat Gigi",
    "BT": "Alat Kacamata / Rehabilitasi",
}

# --------------------------------------------------------------------------
# WIDE-FORMAT INGESTION
# --------------------------------------------------------------------------
# The claims source can now arrive "wide": one row per NIK+Tahun, with every
# benefit x transaction-type x beneficiary combination as its own column
# (e.g. "RJ_Provider_Pegawai", "BT_Reimburse_Keluarga", ...), plus per-row
# entitlement limits ("Limit_Provider_RJ", ..., "Limit_Reimburse") and a
# "Status Employee" flag. This section detects that shape and melts it back
# into the long, one-row-per-transaction-type layout (REQUIRED_COLS) that the
# rest of the dashboard is built around, so nothing downstream needs to change.

WIDE_FORMAT_MARKER_COLS = {"Tahun", "Nama", "Band"}
WIDE_BENEFICIARIES = ["Pegawai", "Keluarga"]


def is_wide_format(df: pd.DataFrame) -> bool:
    cols = {str(c).strip() for c in df.columns}
    return WIDE_FORMAT_MARKER_COLS.issubset(cols)


def melt_wide_claims(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide per-NIK-per-Year claims table into the long transaction-
    level format the dashboard expects. Every benefit/transaction/beneficiary
    cell becomes its own row, even when the claim amount is 0 — that's what
    keeps employees who didn't claim anything counted correctly as Active
    Employees with a full entitlement (Total Limit)."""
    w = raw_df.copy()
    w.columns = [str(c).strip() for c in w.columns]
    w["NIK"] = w["NIK"].astype(str).str.strip()
    w["Year"] = pd.to_numeric(w["Tahun"], errors="coerce").astype("Int64")
    w["Member Name"] = w["Nama"].astype(str).str.strip()
    w["Tipe Pegawai"] = w.get("Tipe Pegawai", "").astype(str).str.strip()
    w["Band"] = w["Band"].astype(str).str.strip()
    status_col = w["Status Employee"] if "Status Employee" in w.columns else ""

    rows = []
    for idx, r in w.iterrows():
        band = r["Band"]
        status = r.get("Status Employee", None) if "Status Employee" in w.columns else None
        for prefix in BENEFIT_PREFIXES:
            plan_code = f"{prefix}-{band}"
            note = PREFIX_NOTE.get(prefix, prefix)
            limit_provider = parse_id_number(r.get(f"Limit_Provider_{prefix}"))
            limit_reimburse = parse_id_number(r.get("Limit_Reimburse"))
            for txn in TXN_TYPES:
                limit = limit_provider if txn == "Provider" else limit_reimburse
                for benef in WIDE_BENEFICIARIES:
                    col = f"{prefix}_{txn}_{benef}"
                    if col not in w.columns:
                        continue
                    claim = parse_id_number(r.get(col))
                    if pd.isna(claim):
                        claim = 0.0
                    row = {
                        "NIK": r["NIK"],
                        "Member Name": r["Member Name"],
                        "Year": r["Year"],
                        "Tipe Pegawai": r["Tipe Pegawai"],
                        "Transaction Type": txn,
                        "Benefit Plan": plan_code,
                        "Benefit Note": note,
                        "Beneficiary": benef,
                        "Claim Amount": claim,
                        "Benefit Limit": limit,
                    }
                    if status is not None:
                        row["Status Employee"] = status
                    rows.append(row)

    long_df = pd.DataFrame(rows)
    extra_cols = REQUIRED_COLS + (["Status Employee"] if "Status Employee" in w.columns else [])
    return long_df[extra_cols]


def build_salary_from_wide(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Limit_Reimburse in the wide file IS each employee's reimbursement
    ceiling (1x Monthly Salary) for that year — so it can be used directly as
    the salary mapping for Sections 4 & 5, per NIK **and year** (more precise
    than a single NIK-only salary file, since salary changes year to year)."""
    if "Limit_Reimburse" not in raw_df.columns:
        return pd.DataFrame(columns=["NIK", "Year", "Monthly Salary"])
    s = raw_df[["NIK", "Tahun", "Limit_Reimburse"]].copy()
    s.columns = ["NIK", "Year", "Monthly Salary"]
    s["NIK"] = s["NIK"].astype(str).str.strip()
    s["Year"] = pd.to_numeric(s["Year"], errors="coerce").astype("Int64")
    s["Monthly Salary"] = s["Monthly Salary"].apply(parse_id_number)
    return s.dropna(subset=["Monthly Salary"]).drop_duplicates(subset=["NIK", "Year"])


def clean_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(
            "The uploaded file is missing required column(s): "
            + ", ".join(missing)
            + f"\n\nExpected columns: {', '.join(REQUIRED_COLS)}"
        )
        st.stop()

    df["Claim Amount"] = df["Claim Amount"].apply(parse_id_number)
    df["Benefit Limit"] = df["Benefit Limit"].apply(parse_id_number)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["NIK"] = df["NIK"].astype(str).str.strip()
    for c in ["Member Name", "Tipe Pegawai", "Transaction Type", "Benefit Plan", "Beneficiary"]:
        df[c] = df[c].astype(str).str.strip()

    n_before = len(df)
    df = df.dropna(subset=["Claim Amount", "Year"])
    n_after = len(df)
    if n_after < n_before:
        st.warning(f"{n_before - n_after} row(s) were dropped due to invalid Claim Amount / Year values.")

    return df

def fmt_rp(value, in_millions_threshold=1e9):
    """Format Rupiah into 'Rp x,xx M' (miliar) or 'Rp x,xx Jt' (juta) like the source dashboard."""
    if pd.isna(value):
        return "Rp 0"
    if abs(value) >= 1e9:
        return f"Rp {value/1e9:,.2f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    elif abs(value) >= 1e6:
        return f"Rp {value/1e6:,.2f} Jt".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        return f"Rp {value:,.0f}".replace(",", ".")

def fmt_pct(value):
    if pd.isna(value):
        return "0,0%"
    return f"{value:.1f}%".replace(".", ",")


# --------------------------------------------------------------------------
# DATA — always load straight from the bundled data.xlsx
# --------------------------------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload Data Transaksi",
    type=["xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Silakan upload file Excel terlebih dahulu.")
    st.stop()

file_bytes = uploaded_file.getvalue()

sheet_names = list_excel_sheets(file_bytes)
claims_sheet = find_sheet(sheet_names, "working") or sheet_names[0]
raw_df = read_excel_sheet(file_bytes, claims_sheet)
raw_df.columns = [str(c).strip() for c in raw_df.columns]

salary_df = None
master_sheet = find_sheet(sheet_names, "employee")
if master_sheet:
    master_df = read_excel_sheet(file_bytes, master_sheet)
    salary_df = build_salary_df_from_master(master_df)

wide_mode = is_wide_format(raw_df)
if wide_mode:
    n_source_rows = len(raw_df)
    long_df = melt_wide_claims(raw_df)
    df = clean_claims(long_df)

    # Limit_Reimburse in the wide file is a direct, per-year salary/entitlement
    # figure — prefer it over any separately-uploaded (NIK-only) salary mapping.
    wide_salary_df = build_salary_from_wide(raw_df)
    if not wide_salary_df.empty:
        salary_df = wide_salary_df

    st.sidebar.caption(
        f"📐 Wide-format data terdeteksi: {n_source_rows:,} baris NIK×Tahun "
        f"diperluas jadi {len(df):,} baris transaksi.".replace(",", ".")
    )
else:
    df = clean_claims(raw_df)

# --------------------------------------------------------------------------
# AI ASSISTANT — data + prompt setup (UI is rendered as a floating overlay
# further below, once the full dataset and filters exist)
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

SYSTEM_PROMPT = """
    You are an AI Health Benefit Data Analyst.

    You answer questions ONLY using the uploaded Health Benefit Excel dataset.

    Never use outside knowledge.
    Never guess.
    If the answer cannot be calculated from the uploaded data, say so.

    =====================================================
    DATA DEFINITIONS
    =====================================================

    Claim Amount
    - Amount claimed by an employee.

    Benefit Limit
    - Annual claim limit.

    Transaction Type
    - Provider
    - Reimburse

    Benefit Plan
    Example:
    RJ-IV
    RI-IV
    RG-IV
    BT-IV

    Band
    Employee band extracted from Benefit Plan.

    =====================================================
    IMPORTANT CALCULATION RULES
    =====================================================

    Provider Limit

    The Provider Benefit Limit is counted ONLY ONCE for each:

    - Employee (NIK)
    - Year
    - Benefit Plan

    Do NOT sum duplicated Provider limits that appear on multiple transaction rows.

    Equivalent logic:

    drop_duplicates(
        subset=["NIK","Year","Benefit Plan"]
    )

    then sum Benefit Limit.


    -----------------------------------------------------

    Reimbursement Limit

    The Reimbursement Benefit Limit represents
    ONE MONTH SALARY.

    It is counted ONLY ONCE for each

    - Employee (NIK)
    - Year

    Do NOT multiply by Benefit Plan.

    Equivalent logic:

    drop_duplicates(
        subset=["NIK","Year"]
    )

    then sum Benefit Limit.

    -----------------------------------------------------

    Total Annual Limit

    Total Limit =

    Provider Limit
    +
    Reimbursement Limit

    -----------------------------------------------------

    Over Reimbursement

    Group by

    NIK
    Year

    sum Claim Amount where
    Transaction Type == Reimburse

    Compare against the yearly Reimbursement Limit.

    -----------------------------------------------------

    Employee Over Any Benefit

    Group by

    NIK
    Year
    Benefit Plan
    Transaction Type

    Claim =
    SUM(Claim Amount)

    Limit =
    MAX(Benefit Limit)

    Employee is Over Limit if

    Claim > Limit

    -----------------------------------------------------

    Average

    Average Claim =
    Average of Claim Amount

    unless user specifies otherwise.

    -----------------------------------------------------

    Ranking

    When user asks:

    Top
    Highest
    Largest
    Lowest
    Bottom

    Always sort correctly.

    -----------------------------------------------------

    Comparison

    If user compares years,
    calculate each year independently before comparing.

    -----------------------------------------------------

    Currency

    Always use Indonesian Rupiah formatting.

    Example

    Rp 1.245.000

    -----------------------------------------------------

    When possible answer using

    1. Short explanation
    2. Table
    3. Final conclusion

    =====================================================
    EXAMPLE QUESTIONS
    =====================================================

    Top 10 claim tahun 2025

    Top reimbursement

    Band dengan claim terbesar

    Provider terbesar

    Utilization per benefit plan

    Top 5 employee claim

    Average claim per employee

    Claim per beneficiary

    Provider vs Reimburse

    Potential saving

    Employee over reimbursement limit

    Employee over any benefit

    Compare 2024 vs 2025

    Trend claim

    Claim by transaction type

    Claim by benefit plan

    Claim by band

    Claim by employee type
"""

if "gemini_file" not in st.session_state:

    df.to_json(
        "claims.json",
        orient="records",
        force_ascii=False
    )

    st.session_state.gemini_file = client.files.upload(
        file="claims.json"
    )

# --------------------------------------------------------------------------
# SIDEBAR — FILTERS
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("🔎 Filters")

df['Band'] = df['Benefit Plan'].str[3:]
df["Band"] = df["Band"].replace("VIP", "I")
df['jenis_claim'] = df['Benefit Plan'].str[:2]

years = sorted(df["Year"].dropna().unique().tolist())
sel_years = st.sidebar.multiselect("Year", years, default=years)

band_opts = sorted(df["Band"].unique().tolist())
sel_band = st.sidebar.multiselect("Band", band_opts, default=band_opts)

tipe_opts = sorted(df["Tipe Pegawai"].unique().tolist())
sel_tipe = st.sidebar.multiselect("Tipe Pegawai", tipe_opts, default=tipe_opts)

trx_opts = sorted(df["Transaction Type"].unique().tolist())
sel_trx = st.sidebar.multiselect("Transaction Type", trx_opts, default=trx_opts)

plan_opts = sorted(df["jenis_claim"].unique().tolist())
sel_plan = st.sidebar.multiselect("jenis_claim", plan_opts, default=plan_opts)

benef_opts = sorted(df["Beneficiary"].unique().tolist())
sel_benef = st.sidebar.multiselect("Beneficiary", benef_opts, default=benef_opts)

if "Status Employee" in df.columns:
    status_opts = sorted(df["Status Employee"].dropna().unique().tolist())
    sel_status = st.sidebar.multiselect("Status Employee", status_opts, default=status_opts)
else:
    sel_status = None

name_option = sorted(df["Member Name"].unique().tolist())
option_name = st.sidebar.selectbox(
    "Cari Nama",
    name_option,
    index=None,
    placeholder="Cari Nama",
)

if st.sidebar.button("↺ Reset filters"):
    st.rerun()

fdf = df[
    df["Year"].isin(sel_years)
    & df["Tipe Pegawai"].isin(sel_tipe)
    & df["Transaction Type"].isin(sel_trx)
    & df["jenis_claim"].isin(sel_plan)
    & df["Beneficiary"].isin(sel_benef)
    & df['Band'].isin(sel_band)
].copy()

if sel_status is not None:
    fdf = fdf[fdf["Status Employee"].isin(sel_status)]

if option_name:
    fdf = fdf[fdf['Member Name'] == option_name]

employee_per_type = (
    fdf.groupby("Tipe Pegawai")["NIK"]
       .nunique()
       .reset_index(name="Jumlah Karyawan")
)

if fdf.empty:
    st.warning("No data matches the current filters.")
    st.stop()
# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------

st.markdown("## 🩺 HEALTH BENEFIT UTILIZATION DASHBOARD")
yr_label = f"{min(sel_years)} – {max(sel_years)}" if len(sel_years) > 1 else f"{sel_years[0]}"
st.caption(f"Historical Analysis 2023-2026 Q1 (Ongoing)")


# --------------------------------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------------------------------
total_employees_all = df["NIK"].nunique()

active_employees = fdf["NIK"].nunique()

total_claim = fdf["Claim Amount"].sum()
total_claim_all = df["Claim Amount"].sum()

provider_claim = fdf.loc[fdf["Transaction Type"] == "Provider", "Claim Amount"].sum()
reimburse_claim = fdf.loc[fdf["Transaction Type"] == "Reimburse", "Claim Amount"].sum()

pct_of_total = lambda part, whole: (part / whole * 100) if whole else 0

# Unique limits per NIK+Year+Benefit Plan+Transaction Type.
# The limit value itself repeats on every transaction row within a given
# (NIK, Year, Benefit Plan, Transaction Type) — that repetition must be deduped away.
# BUT the limit is an ANNUAL ceiling (RJ/RI/RG/BT plans reset every calendar year,
# and Reimburse = 1x that year's Monthly Salary), so an employee with claims in
# 2023, 2024 AND 2025 legitimately had three separate annual entitlements — Year
# must stay in the key or multi-year totals get understated (confirmed: 270 of 624
# NIK+Plan+TxnType combos span more than one year).
# Transaction Type must also stay in the key, because 'Reimburse' rows carry a
# different number than 'Provider' rows for the very same NIK+Benefit Plan
# (Reimburse limit = 1x Monthly Salary ceiling, Provider limit = the plan-specific
# RJ/RI/RG/BT ceiling) — dropping it would collapse those two distinct entitlements
# into one and silently drop the other.
limit_provider_unique = fdf[fdf['Transaction Type']=='Provider'].drop_duplicates(subset=["NIK", "Year", "Benefit Plan"])
limit_reimburse_unique = fdf[fdf['Transaction Type']=='Reimburse'].drop_duplicates(subset=["NIK", "Year"])

total_limit = limit_provider_unique["Benefit Limit"].sum() +limit_reimburse_unique["Benefit Limit"].sum()

# Employees over limit on ANY benefit (claim per NIK+Year+Benefit Plan+Transaction Type
# vs that limit — must split by Transaction Type too, otherwise a Provider claim could
# get compared against the much larger Reimburse/salary limit, or vice versa).
grp_over = fdf.groupby(["NIK", "Year", "Benefit Plan", "Transaction Type"], as_index=False).agg(
    claim=("Claim Amount", "sum"), limit=("Benefit Limit", "max")
)
fig = px.bar(
    employee_per_type,
    x="Tipe Pegawai",
    y="Jumlah Karyawan",
    text="Jumlah Karyawan",
    color="Tipe Pegawai",
    title="Jumlah Karyawan Unik per Tipe Pegawai"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    xaxis_title="Tipe Pegawai",
    yaxis_title="Jumlah Karyawan",
    showlegend=False,
    height=450
)


grp_over["over"] = grp_over["claim"] > grp_over["limit"]
employees_over_any = grp_over.loc[grp_over["over"], "NIK"].nunique()

# Reimbursement-policy potential saving.
# The per-row "Benefit Limit" on Reimburse transactions already IS the 1x
# Monthly Salary ceiling (carried straight from the source file), so this no
# longer needs a separately-uploaded salary mapping. An uploaded salary_df
# (e.g. from an "employee" sheet in the same workbook) is only used to
# backfill NIK+Year combos where that limit happens to be missing.
reimb_by_emp = (
    fdf.loc[fdf["Transaction Type"] == "Reimburse"]
    .groupby(["NIK", "Year"], as_index=False)
    .agg(**{"Reimbursement Claim": ("Claim Amount", "sum"), "Monthly Salary": ("Benefit Limit", "max")})
)

if salary_df is not None and not salary_df.empty:
    merge_keys = ["NIK", "Year"] if "Year" in salary_df.columns else ["NIK"]
    fallback = salary_df.rename(columns={"Monthly Salary": "Monthly Salary_fallback"})
    reimb_by_emp = reimb_by_emp.merge(fallback, on=merge_keys, how="left")
    reimb_by_emp["Monthly Salary"] = reimb_by_emp["Monthly Salary"].fillna(reimb_by_emp["Monthly Salary_fallback"])
    reimb_by_emp = reimb_by_emp.drop(columns=["Monthly Salary_fallback"])

reimb_by_emp["Monthly Salary"] = reimb_by_emp["Monthly Salary"].fillna(0)
reimb_by_emp["Over Limit Amount"] = (
    reimb_by_emp["Reimbursement Claim"] - reimb_by_emp["Monthly Salary"]
).clip(lower=0)
reimb_by_emp["Over Limit"] = reimb_by_emp["Over Limit Amount"] > 0

has_salary = bool(reimb_by_emp["Monthly Salary"].gt(0).any())
n_over_reimb = int(reimb_by_emp["Over Limit"].sum())
potential_saving = reimb_by_emp["Over Limit Amount"].sum()

# --------------------------------------------------------------------------
# KPI CARDS
# --------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)


def kpi_card(col, icon, color, title, value, sub):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="background:{color};width:30px;height:30px;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;color:white;font-size:0.9rem;">{icon}</div>
                    <div class="kpi-title" style="color:{color};">{title}</div>
                </div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


kpi_card(k1, "👥", NAVY, "Active Employees", f"{active_employees:,}".replace(",", "."),
          f"{pct_of_total(active_employees, total_employees_all):.0f}% dari total")
kpi_card(k2, "🧾", NAVY, "Total Claim Amount", fmt_rp(total_claim),
          f"{min(sel_years) if sel_years else ''}–{max(sel_years) if sel_years else ''}")
kpi_card(k3, "🛡️", BLUE, "Total Provider Claim", fmt_rp(provider_claim),
          f"{fmt_pct(pct_of_total(provider_claim, total_claim))} dari total")
kpi_card(k4, "💵", PURPLE, "Total Reimbursement", fmt_rp(reimburse_claim),
          f"{fmt_pct(pct_of_total(reimburse_claim, total_claim))} dari total")
# kpi_card(k5, "🧑‍🤝‍🧑", ORANGE, "# Employees Over Limit (Any Benefit)", f"{employees_over_any:,}".replace(",", "."),
#           f"{fmt_pct(pct_of_total(employees_over_any, active_employees))} dari total")
if has_salary:
    kpi_card(k5, "🪙", GREEN, "Potential Saving (Reimbursement Policy)", fmt_rp(potential_saving),
              "Jika kebijakan diterapkan")
else:
    kpi_card(k5, "🪙", GREEN, "Potential Saving (Reimbursement Policy)", "—",
              "Upload salary mapping to enable")

st.write("")
total_per_type = fdf.groupby("Tipe Pegawai")["NIK"].nunique()
 
aktif_niks = set(fdf.loc[fdf["Claim Amount"] > 0, "NIK"].unique())
aktif_per_type = (
    fdf[fdf["NIK"].isin(aktif_niks)]
    .groupby("Tipe Pegawai")["NIK"]
    .nunique()
    .reindex(total_per_type.index, fill_value=0)
)
tidak_aktif_per_type = total_per_type - aktif_per_type
 
colors = {
    "Digital Talent": "#00A86B",
    "TPPBW": "#E74C3C",
    "Talent Mobility": "#3498DB",
}
 
cols = st.columns(len(total_per_type))
 
for i, tipe in enumerate(total_per_type.index):
    total_n = int(total_per_type[tipe])
    aktif_n = int(aktif_per_type[tipe])
    tidak_aktif_n = int(tidak_aktif_per_type[tipe])
    pct_aktif = (aktif_n / total_n * 100) if total_n else 0
    bg = colors.get(tipe, "#6C757D")
 
    with cols[i]:
        st.markdown(f"""
            <div style="
            background:{bg};
            padding:18px 20px;
            margin-bottom: 20px;
            border-radius:12px;
            color:white;
            text-align:center;
            ">
            <h3 style="margin:0 0 6px 0;">{tipe}</h3>
            <h1 style="font-size:40px;margin:6px 0;">{total_n}</h1>
            <div style="font-size:13px;opacity:0.9;margin-bottom:8px;">Total Karyawan</div>
            <div style="display:flex;justify-content:space-between;gap:8px;
                        background:rgba(255,255,255,0.15);border-radius:8px;padding:8px 12px;">
                <div style="text-align:left;">
                    <div style="font-size:20px;font-weight:700;">{aktif_n}</div>
                    <div style="font-size:11px;opacity:0.9;">Aktif Transaksi</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:20px;font-weight:700;">{tidak_aktif_n}</div>
                    <div style="font-size:11px;opacity:0.9;">Tidak Aktif Transaksi</div>
                </div>
            </div>
            <div style="font-size:11px;opacity:0.85;margin-top:6px;">{pct_aktif:.1f}% aktif dari total</div>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# TREND CLAIM PER TAHUN
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">📈 TREND CLAIM PER TAHUN</div>', unsafe_allow_html=True)

trend_total = fdf.groupby("Year", as_index=False)["Claim Amount"].sum().sort_values("Year")
trend_by_txn = (
    fdf.groupby(["Year", "Transaction Type"], as_index=False)["Claim Amount"]
    .sum()
    .sort_values("Year")
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=trend_total["Year"], y=trend_total["Claim Amount"],
    mode="lines+markers+text",
    name="Total Claim",
    line=dict(color=NAVY, width=3),
    marker=dict(size=8),
    text=[fmt_rp(v) for v in trend_total["Claim Amount"]],
    textposition="top center",
))
for txn, color in [("Provider", BLUE), ("Reimburse", PURPLE)]:
    sub = trend_by_txn[trend_by_txn["Transaction Type"] == txn]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(
        x=sub["Year"], y=sub["Claim Amount"],
        mode="lines+markers",
        name=txn,
        line=dict(color=color, width=2, dash="dot"),
        marker=dict(size=6),
    ))

fig.update_layout(
    height=320,
    xaxis=dict(title="Tahun", dtick=1),
    yaxis=dict(title="Total Claim (Rp)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# SECTION 1 — BERAPA UTILISASI MANFAAT KESEHATAN?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">① BERAPA UTILISASI MANFAAT KESEHATAN?</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Provider vs Reimbursement (% dari Total Claim)**")
    pv = pd.DataFrame({
        "Type": ["Provider", "Reimbursement"],
        "Amount": [provider_claim, reimburse_claim],
    })
    fig = go.Figure(go.Pie(labels=pv["Type"], values=pv["Amount"], hole=0.6,
                             marker_colors=[BLUE, GREEN], textinfo="percent"))
    fig.update_layout(height=300, margin=dict(t=20, b=10),
                        annotations=[dict(text=fmt_rp(total_claim), x=0.5, y=0.5, font_size=16, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)
used = total_claim
total = total_limit
remaining = max(total - used, 0)
utilization = (used / total * 100) if total else 0
with c2:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[""],
        x=[utilization],
        orientation="h",
        marker_color=BLUE,
        text=[f"{utilization:.1f}%"],
        textposition="inside",
        insidetextanchor="start",
        hovertemplate="Used: %{x:.1f}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        y=[""],
        x=[100 - min(utilization, 100)],
        orientation="h",
        marker_color="#E5E7EB",
        hoverinfo="skip"
    ))

    fig.update_layout(
        barmode="stack",
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            range=[0, 100],
            ticksuffix="%",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    st.markdown("""
    <style>
    /* Label */
    div[data-testid="stMetricLabel"] p {
        font-size: 12px !important;
    }

    /* Angka */
    div[data-testid="stMetricValue"] {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with c1:
        c1.metric("💰 Remaining", f"Rp {remaining:,.0f}".replace(",", "."))
    with c2:
        c2.metric("📦 Total Limit", f"Rp {total:,.0f}".replace(",", "."))

st.markdown("---")
st.markdown("**Total Claim vs 1 Month Salary Limit**")
used = provider_claim + reimburse_claim
total_reimburse_limit = limit_reimburse_unique["Benefit Limit"].sum()
remaining_reimburse_limit = max(total_reimburse_limit - used, 0)
utilization_reimburse = (used / total_reimburse_limit * 100) if total_reimburse_limit else 0

fig = go.Figure()
fig.add_trace(go.Bar(
    y=[""],
    x=[utilization_reimburse],
    orientation="h",
    marker_color=BLUE,
    text=[f"{utilization_reimburse:.1f}%"],
    textposition="inside",
    insidetextanchor="start",
    hovertemplate="Used: %{x:.1f}%<extra></extra>"
))
fig.add_trace(go.Bar(
    y=[""],
    x=[100 - min(utilization_reimburse, 100)],
    orientation="h",
    marker_color="#E5E7EB",
    hoverinfo="skip"
))
fig.update_layout(
    barmode="stack",
    height=100,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

pc1, pc2, pc3 = st.columns(3)
with pc1:
    pc1.metric("💰 Used", f"Rp {used:,.0f}".replace(",", "."))
with pc2:
    pc2.metric("💰 Remaining Limit", f"Rp {remaining_reimburse_limit:,.0f}".replace(",", "."))
with pc3:
    pc3.metric("📦 Total Limit", f"Rp {total_reimburse_limit:,.0f}".replace(",", "."))

# KPI di bawah progress bar

# --------------------------------------------------------------------------
# SECTION 2 — SIAPA YANG MEMANFAATKAN MANFAAT KESEHATAN?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">② SIAPA YANG MEMANFAATKAN MANFAAT KESEHATAN?</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    st.markdown("**Berdasarkan Beneficiary (% dari Total Claim)**")
    bb = fdf.groupby("Beneficiary", as_index=False)["Claim Amount"].sum()
    fig = go.Figure(go.Pie(labels=bb["Beneficiary"], values=bb["Claim Amount"], hole=0.6,
                             marker_colors=[NAVY, GREEN, BLUE, ORANGE], textinfo="percent"))
    fig.update_layout(height=300, margin=dict(t=20, b=10),
                        annotations=[dict(text=fmt_rp(bb["Claim Amount"].sum()), x=0.5, y=0.5, font_size=16, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)



with c2:
    st.markdown("**Berdasarkan Tipe Pegawai (% dari Total Claim)**")

    colors = {
        "Digital Talent": "#00A86B",
        "TPPBW": "#E74C3C",
        "Talent Mobility": "#3498DB",
    }

    tp = (
        fdf.groupby("Tipe Pegawai", as_index=False)["Claim Amount"]
        .sum()
    )

    fig = px.pie(
        tp,
        names="Tipe Pegawai",
        values="Claim Amount",
        color="Tipe Pegawai",
        color_discrete_map=colors,
        hole=0.5,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Claim: Rp %{value:,.0f}<br>%{percent}<extra></extra>"
    )

    fig.update_layout(
        height=300,
        margin=dict(t=20, b=10),
        legend_title=None
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Rata-rata claim per jenis pegawai**")

# Average claim per tipe pegawai (claim > 0)
avg_claim_type = (
    fdf[fdf["Claim Amount"] > 0]
    .groupby("Tipe Pegawai", as_index=False)["Claim Amount"]
    .mean()
    .rename(columns={"Claim Amount": "Average Claim"})
)
    # Overall average
overall_avg = fdf.loc[
    fdf["Claim Amount"] > 0,
    "Claim Amount"
].mean()

# Tambahkan baris Overall
avg_claim_type.loc[len(avg_claim_type)] = [
    "Overall",
    overall_avg
]

# Card
cols = st.columns(len(avg_claim_type))

colors = {
    "Digital Talent": "#00A86B",
    "TPPBW": "#E74C3C",
    "Talent Mobility": "#3498DB",
    "Overall": "#6C757D",
}

for col, (_, row) in zip(cols, avg_claim_type.iterrows()):
    with col:
        st.markdown(f"""
            <div style="
            background:{colors.get(row['Tipe Pegawai'], '#666')};
            padding:20px;   /* atas kanan bawah kiri */
            border-radius:12px;
            color:white;
            margin-bottom:20px;
            text-align:center;
            ">
            <h3>{row['Tipe Pegawai']}</h3>

            <h1 style="
            font-size:28px;
            margin:15px 0;
            font-weight:bold;
            ">
            Rp{row['Average Claim']:,.2f}
            </h1>

            </div>
            """.replace(",", "X").replace(".", ",").replace("X", "."), unsafe_allow_html=True)
        
st.markdown("**Rata-rata claim per jenis claim**")
fdf["jenis_klaim"] = fdf["Benefit Plan"].str[:2]

avg_claim_jenis = (
    fdf[fdf["Claim Amount"] > 0]
    .groupby("jenis_klaim", as_index=False)["Claim Amount"]
    .mean()
    .rename(columns={"Claim Amount": "Average Claim"})
)

cols = st.columns(max(1, len(avg_claim_jenis)))
colors = {
    "RJ": "#3498DB",      # Rawat Jalan
    "RI": "#E74C3C",      # Rawat Inap
    "RG": "#2ECC71",      # Rawat Gigi
    "MC": "#F39C12",      # Medical Check Up (ubah kalau beda)
    "Overall": "#6C757D",
}

MAP_JENIS = {
    "BT":  "Kacamata / Rehabilitasi",
    "RI": "Rawat Inap",
    "RJ": "Rawat Jalan",
    "RG": "Rawat Gigi"
}

for col, (_, row) in zip(cols, avg_claim_jenis.iterrows()):
    nilai = f"Rp{row['Average Claim']:,.2f}"
    nilai = nilai.replace(",", "X").replace(".", ",").replace("X", ".")

    with col:
        st.markdown(f"""
            <div style="
            background:{colors.get(row['jenis_klaim'], '#6C757D')};
            padding:20px;
            border-radius:12px;
            color:white;
            text-align:center;
            margin-bottom:20px;
            ">
            <h3>{MAP_JENIS[row['jenis_klaim']]}</h3>

            <h1 style="font-size:30px;margin:15px 0;">
            {nilai}
            </h1>
            </div>
            """, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# SECTION 3 — BENEFIT APA YANG PALING BANYAK DIMANFAATKAN?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">③ BENEFIT APA YANG PALING BANYAK DIMANFAATKAN?</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

plan_claim = fdf.groupby("Benefit Plan", as_index=False)["Claim Amount"].sum()
plan_claim["pct"] = plan_claim["Claim Amount"] / plan_claim["Claim Amount"].sum() * 100
plan_claim = plan_claim.sort_values("Claim Amount")

plan_limit = fdf.drop_duplicates(["NIK", "Year", "Benefit Plan", "Transaction Type"]).groupby("Benefit Plan", as_index=False)["Benefit Limit"].sum()
plan_util = plan_claim.merge(plan_limit, on="Benefit Plan", how="left")
plan_util["utilization"] = plan_util["Claim Amount"] / plan_util["Benefit Limit"] * 100
plan_util = plan_util.sort_values("utilization")


def util_color(v):
    if v >= 70:
        return RED
    elif v >= 45:
        return YELLOW
    return GREEN


with c1:

    st.markdown("**Total Claim per Benefit Plan (Rp)**")

    # Top 3 berdasarkan total claim
    top3_idx = set(plan_claim["Claim Amount"].nlargest(3).index)

    bar_colors = [
        BLUE if idx in top3_idx else "#D3D3D3"
        for idx in plan_claim.index
    ]

    fig = go.Figure(
        go.Bar(
            x=plan_claim["Claim Amount"],
            y=plan_claim["Benefit Plan"],
            orientation="h",
            marker_color=bar_colors,
            text=plan_claim.apply(
                lambda r: f"{fmt_rp(r['Claim Amount'])} ({r['pct']:.1f}%)",
                axis=1
            ),
            textposition="outside",
        )
    )

    fig.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=20, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

with c2:

    st.markdown("**Utilization per Benefit Plan (% dari Limit)**")

    top3_idx = set(plan_util["utilization"].nlargest(3).index)

    bar_colors = [
        BLUE if idx in top3_idx else "#D3D3D3"
        for idx in plan_util.index
    ]

    fig = go.Figure(
        go.Bar(
            x=plan_util["utilization"],
            y=plan_util["Benefit Plan"],
            orientation="h",
            marker_color=bar_colors,
            text=plan_util["utilization"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )

    fig.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=20, b=10),
        xaxis=dict(
            range=[
                0,
                max(
                    100,
                    plan_util["utilization"].max() * 1.15 if len(plan_util) else 100,
                ),
            ]
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    if plan_util["Benefit Limit"].isna().any():
        st.caption("Some Benefit Plans are missing limit data under the current filters.")

st.markdown("**Total Claim digunakan**")

summary_jenis = (
    fdf.groupby("jenis_klaim", as_index=False)
    .agg(
        Claim=("Claim Amount", "sum"),
        Limit=("Benefit Limit", "sum")
    )
)

summary_jenis["Utilization"] = (
    summary_jenis["Claim"] / summary_jenis["Limit"] * 100
)

cols = st.columns(len(summary_jenis))

for col, (_, row) in zip(cols, summary_jenis.iterrows()):

    claim = f"Rp{row['Claim']:,.0f}".replace(",", ".")
    limit = f"Rp{row['Limit']:,.0f}".replace(",", ".")

    with col:
        st.markdown(f"""
        <div style="
        background:{colors.get(row['jenis_klaim'], '#6C757D')};
        padding:20px;
        border-radius:12px;
        margin-bottom: 20px;
        color:white;
        text-align:center;
        ">

        <h3>{MAP_JENIS[row['jenis_klaim']]}</h3>

        <h2>
            {claim}
        </h2>

        </div>

        </div>
        """, unsafe_allow_html=True)
# --------------------------------------------------------------------------
# SECTION 4 — SIAPA YANG MELEBIHI LIMIT REIMBURSEMENT (1x MONTHLY SALARY)?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">④ SIAPA YANG MELEBIHI LIMIT REIMBURSEMENT (1x MONTHLY SALARY)?</div>', unsafe_allow_html=True)

if not has_salary:
    st.warning(
        "No Reimburse limit (1x Monthly Salary) is available for the current filter, "
        "so this section can't be computed."
    )
else:
    over_df = reimb_by_emp[reimb_by_emp["Over Limit"]].copy()

    emp_info = (
        fdf[["NIK", "Member Name", "Band", "Tipe Pegawai"]]
        .drop_duplicates("NIK")
    )

    over_df = (
        over_df.merge(emp_info, on="NIK", how="left")
            .sort_values("Over Limit Amount", ascending=False)
    )

    m1, m2 = st.columns(2)
    with m1:
        employees_over = over_df["NIK"].nunique()
        st.metric(
            "Employees Over Reimbursement Limit",
            f"{employees_over:,}".replace(",", "."),
            f"{pct_of_total(employees_over, active_employees):.1f}% dari total"
        )
    with m2:
        st.metric("Total Over Limit Amount", fmt_rp(over_df["Over Limit Amount"].sum()))

    top10 = over_df.copy()
    top10.insert(0, "Rank", range(1, len(top10) + 1))

    top10_display = top10[
        [
            "Rank",
            "NIK",
            "Member Name",
            "Band",
            "Year",
            "Tipe Pegawai",
            "Monthly Salary",
            "Reimbursement Claim",
            "Over Limit Amount",
        ]
    ].rename(columns={
        "Band": "Band",
        "Tipe Pegawai": "Tipe Pegawai",
        "Monthly Salary": "Monthly Salary (Rp)",
        "Reimbursement Claim": "Reimbursement Claim (Rp)",
        "Over Limit Amount": "Over Limit Amount (Rp)",
    })

    st.dataframe(top10_display)

st.markdown("---")
st.markdown("**🧪 Simulasi: Total Claim (Provider + Reimburse) vs Limit Reimburse**")
st.caption(
    "Skenario: seluruh klaim setahun (Provider + Reimburse digabung) dibandingkan "
    "terhadap satu batas tunggal, yaitu Limit Reimburse (1x Monthly Salary) tahun itu — "
    "bukan terhadap limit masing-masing plan. Ini murni simulasi, bukan aturan aktual."
)

if not has_salary:
    st.warning(
        "No Reimburse limit (1x Monthly Salary) is available for the current filter, "
        "so this simulation can't be run."
    )
else:
    total_claim_by_emp_year = (
        fdf.groupby(["NIK", "Year"], as_index=False)["Claim Amount"]
        .sum()
        .rename(columns={"Claim Amount": "Total Claim (Provider+Reimburse)"})
    )

    reimburse_limit_map = limit_reimburse_unique[["NIK", "Year", "Benefit Limit"]].rename(
        columns={"Benefit Limit": "Limit Reimburse"}
    )

    sim_all = total_claim_by_emp_year.merge(reimburse_limit_map, on=["NIK", "Year"], how="left")
    sim_all["Over Limit Amount"] = (
        sim_all["Total Claim (Provider+Reimburse)"] - sim_all["Limit Reimburse"]
    ).clip(lower=0)
    sim_all["Over Limit"] = sim_all["Over Limit Amount"] > 0

    emp_info = fdf[["NIK", "Member Name", "Band", "Tipe Pegawai"]].drop_duplicates("NIK")
    sim_over = (
        sim_all.loc[sim_all["Over Limit"]]
        .merge(emp_info, on="NIK", how="left")
        .sort_values("Over Limit Amount", ascending=False)
    )

    sm1, sm2 = st.columns(2)
    with sm1:
        n_sim_over = sim_over["NIK"].nunique()
        st.metric(
            "Employees Over Limit Reimburse (Total Claim)",
            f"{n_sim_over:,}".replace(",", "."),
            f"{pct_of_total(n_sim_over, active_employees):.1f}% dari total"
        )
    with sm2:
        st.metric("Total Over Limit Amount (Simulasi)", fmt_rp(sim_over["Over Limit Amount"].sum()))

    if sim_over.empty:
        st.info("Tidak ada NIK yang melebihi Limit Reimburse pada skenario ini.")
    else:
        sim_display = sim_over.copy()
        sim_display.insert(0, "Rank", range(1, len(sim_display) + 1))
        sim_display = sim_display[
            [
                "Rank", "NIK", "Member Name", "Band", "Year", "Tipe Pegawai",
                "Total Claim (Provider+Reimburse)", "Limit Reimburse", "Over Limit Amount",
            ]
        ].rename(columns={
            "Total Claim (Provider+Reimburse)": "Total Claim (Rp)",
            "Limit Reimburse": "Limit Reimburse (Rp)",
            "Over Limit Amount": "Over Limit Amount (Rp)",
        })
        st.dataframe(sim_display, use_container_width=True)

# --------------------------------------------------------------------------
# SECTION 5 — DI MANA TERDAPAT POTENSI EFISIENSI BIAYA?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">⑤ APA DAMPAK POTENSIAL DARI SKENARIO KEBIJAKAN?</div>', unsafe_allow_html=True)

if not has_salary:
    st.warning(
        "No Reimburse limit (1x Monthly Salary) is available for the current filter, "
        "so this simulation can't be run."
    )
else:
    # c1, c2 = st.columns(2)

    reimb_total_before = reimb_by_emp["Reimbursement Claim"].sum()
    reimb_total_after = (reimb_by_emp["Reimbursement Claim"] - reimb_by_emp["Over Limit Amount"]).sum()
    saving_total = reimb_by_emp["Over Limit Amount"].sum()
    n_over_before = int(reimb_by_emp["Over Limit"].sum())
    pct_over_before = pct_of_total(n_over_before, active_employees)

  
    st.markdown("**Simulasi Dampak Kebijakan**")
    sim_table = pd.DataFrame({
            "": ["Total Reimbursement Claim", "# Employees Over Limit", "% Employees Over Limit"],
            "Existing": [fmt_rp(reimb_total_before), f"{n_over_before}", f"{pct_over_before:.1f}%"],
            "Skenario Simulasi": [fmt_rp(reimb_total_after), "0", "0,0%"],
            "Potensi Dampak": [fmt_rp(saving_total), f"{n_over_before}", f"{pct_over_before:.1f}%"],
    })
    st.dataframe(sim_table, use_container_width=True, hide_index=True)


    st.markdown("**Potensi Efisiensi per Tipe Pegawai (Rp)**")
    tp_map = fdf.drop_duplicates("NIK")[["NIK", "Tipe Pegawai"]]
    eff_tp = reimb_by_emp.merge(tp_map, on="NIK", how="left")
    eff_tp = eff_tp.groupby("Tipe Pegawai", as_index=False)["Over Limit Amount"].sum().sort_values("Over Limit Amount")
    fig = px.bar(eff_tp, x="Over Limit Amount", y="Tipe Pegawai", orientation="h",
                      text=eff_tp["Over Limit Amount"].apply(fmt_rp))
    fig.update_traces(marker_color=GREEN)
    fig.update_layout(height=260, xaxis_title=None, yaxis_title=None, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


st.markdown("---")
st.caption("Catatan: Data berdasarkan transaksi pada rentang tahun terfilter dan hanya mencakup pegawai aktif dalam file yang diunggah.")

with st.expander("🔍 View filtered raw data"):
    st.dataframe(fdf, use_container_width=True)

# --------------------------------------------------------------------------
# AI ASSISTANT — FLOATING CHAT BUTTON + OVERLAY
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Floating action button (bottom-right) that toggles the chat overlay */
    div.element-container:has(> div#chat-fab-anchor)
        + div.element-container div[data-testid="stButton"] {
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 10000;
        width: auto;
    }
    div.element-container:has(> div#chat-fab-anchor)
        + div.element-container div[data-testid="stButton"] button {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2f6fb0, #1b3a5c);
        color: white !important;
        font-size: 24px;
        border: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        padding: 0;
        line-height: 1;
    }
    div.element-container:has(> div#chat-fab-anchor)
        + div.element-container div[data-testid="stButton"] button:hover {
        transform: scale(1.06);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    div.element-container:has(> div#chat-fab-anchor)
        + div.element-container div[data-testid="stButton"] button p {
        font-size: 24px !important;
    }

    /* Overlay chat panel (bottom-right, above the FAB) */
    div.element-container:has(> div#chat-panel-anchor)
        + div.element-container div[data-testid="stVerticalBlockBorderWrapper"] {
        position: fixed;
        bottom: 96px;
        right: 24px;
        width: 380px;
        max-width: 92vw;
        z-index: 9999;
        background: white;
        border-radius: 16px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
        padding: 6px 6px 2px 6px;
    }
    @media (max-width: 480px) {
        div.element-container:has(> div#chat-panel-anchor)
            + div.element-container div[data-testid="stVerticalBlockBorderWrapper"] {
            right: 12px;
            bottom: 88px;
            width: 92vw;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div id="chat-fab-anchor"></div>', unsafe_allow_html=True)
fab_icon = "✖️" if st.session_state.chat_open else "💬"
if st.button(fab_icon, key="chat_fab_toggle", help="Tanya AI soal data ini"):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

if st.session_state.chat_open:
    st.markdown('<div id="chat-panel-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**🤖 AI Health Benefit Assistant**")
        st.caption("Tanya apa saja soal data yang sudah diupload.")

        chat_box = st.container(height=360)
        with chat_box:
            if not st.session_state.messages:
                st.caption("Belum ada percakapan. Coba tanya: “Top 5 employee claim tahun 2025”.")
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        prompt = st.chat_input("Ask anything about the data...", key="chat_overlay_input")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Analyzing data..."):
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=[
                        SYSTEM_PROMPT,
                        st.session_state.gemini_file,
                        prompt,
                    ],
                )
                answer = response.text

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()