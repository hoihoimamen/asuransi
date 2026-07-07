"""
Health Benefit Utilization Dashboard
=====================================
Streamlit rebuild of the "Health Benefit Utilization Dashboard - Historical
Analysis 2023-2025" report.

Run with:
    streamlit run app.py
"""

import io
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

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
PAREA_TIPE_MAP = {
    "AA": "TPPBW",
    "AD": "Talent Mobility",
    "AG": "Digital Talent",
}
# Manual fallback for Provider limits that can't be learned from any existing
# transaction (no employee at that tier ever claimed that plan via Provider).
# Add more "PLAN-TIER": amount entries here as they're confirmed.
MANUAL_LIMIT_OVERRIDES = {
    "BT-I": 3_500_000,
}


def build_employee_master(karyawan_df: pd.DataFrame) -> pd.DataFrame:
    """Turn a 'karyawan.xlsx'-style wage table (one row per wage component per
    employee) into one row per employee, with Monthly Salary = sum of all their
    wage-component Amounts (Gaji, Insentif, Tunjangan, etc.), plus their earliest
    Start date (join year) used to avoid zero-filling years before they joined."""
    k = karyawan_df.copy()
    k.columns = [str(c).strip() for c in k.columns]
    required = ["Pers.no.", "Personnel Number", "PArea", "PS group", "Amount"]
    missing = [c for c in required if c not in k.columns]
    if missing:
        st.sidebar.error(f"karyawan.xlsx is missing column(s): {', '.join(missing)} — skipping zero-fill.")
        return None
    k["NIK"] = k["Pers.no."].astype(str).str.strip()
    k["Amount"] = k["Amount"].apply(parse_id_number)

    if "Start date" in k.columns:
        k["_start_dt"] = pd.to_datetime(k["Start date"], format="%d.%m.%Y", errors="coerce")
        if k["_start_dt"].isna().all():
            # fallback: let pandas guess the format if the fixed format didn't match
            k["_start_dt"] = pd.to_datetime(k["Start date"], dayfirst=True, errors="coerce")
    else:
        k["_start_dt"] = pd.NaT

    emp = k.groupby("NIK").agg(
        Monthly_Salary=("Amount", "sum"),
        Personnel_Number=("Personnel Number", "first"),
        PS_group=("PS group", "first"),
        PArea=("PArea", "first"),
        Start_Date=("_start_dt", "min"),
    ).reset_index()
    emp["Member Name"] = emp["Personnel_Number"].astype(str).str.replace(
        r"^(Bpk\.|Ibu)\s*", "", regex=True
    ).str.strip()
    emp["Tier"] = emp["PS_group"].astype(str).str.strip()
    emp["Start_Year"] = emp["Start_Date"].dt.year
    emp["Tipe Pegawai"] = emp["PArea"].map(PAREA_TIPE_MAP).fillna("Tidak Diketahui")
    return emp


def build_zero_fill_rows(claims_df: pd.DataFrame, employee_master: pd.DataFrame):
    """For every employee in employee_master, for every Year already present in
    claims_df AND >= that employee's join year (from 'Start date' — an employee
    who joined in 2024 gets no 2023 rows; one who joined in 2026 gets none at all
    since the study period is 2023-2025), for the 4 main benefit categories at
    that employee's tier (from PS group), for both Provider and Reimburse: if
    that exact (NIK, Year, Benefit Plan, Transaction Type) combination has no
    real transaction, add a synthetic Claim Amount = 0 row carrying the correct
    entitlement Limit, so employees who never claimed anything (or only claimed
    some benefits) still count fully toward Active Employees / Total Limit /
    Utilization.

    Returns (zero_df, info dict) — info has diagnostics for the sidebar.
    """
    years_in_scope = sorted(claims_df["Year"].dropna().unique().tolist())
    existing_keys = set(zip(claims_df["NIK"], claims_df["Year"], claims_df["Benefit Plan"], claims_df["Transaction Type"]))

    provider_limit_lookup = (
        claims_df.loc[claims_df["Transaction Type"] == "Provider"]
        .drop_duplicates("Benefit Plan")
        .set_index("Benefit Plan")["Benefit Limit"]
        .to_dict()
    )
    # Fill in any plan+tier combo that has no learnable limit with a manual override.
    for plan_code, amount in MANUAL_LIMIT_OVERRIDES.items():
        if plan_code not in provider_limit_lookup or pd.isna(provider_limit_lookup.get(plan_code)):
            provider_limit_lookup[plan_code] = amount

    emp = employee_master
    n_no_start_date = int(emp["Start_Year"].isna().sum())
    n_joined_after_scope = int((emp["Start_Year"] > max(years_in_scope)).sum()) if years_in_scope else 0

    rows = []
    missing_limit_combos = set()
    used_overrides = set()
    skipped_no_tier = 0
    for _, e in emp.iterrows():
        tier = e["Tier"]
        if pd.isna(tier) or tier == "" or tier == "nan":
            skipped_no_tier += 1
            continue
        start_year = e["Start_Year"]  # NaN if unknown -> don't restrict (assume always eligible)
        for year in years_in_scope:
            if pd.notna(start_year) and year < start_year:
                continue  # not employed yet that year
            for prefix in BENEFIT_PREFIXES:
                plan_code = f"{prefix}-{tier}"
                for txn in TXN_TYPES:
                    key = (e["NIK"], year, plan_code, txn)
                    if key in existing_keys:
                        continue
                    if txn == "Provider":
                        limit = provider_limit_lookup.get(plan_code, np.nan)
                        if pd.isna(limit):
                            missing_limit_combos.add(plan_code)
                        elif plan_code in MANUAL_LIMIT_OVERRIDES:
                            used_overrides.add(plan_code)
                    else:
                        limit = e["Monthly_Salary"]
                    rows.append({
                        "NIK": e["NIK"],
                        "Member Name": e["Member Name"],
                        "Year": year,
                        "Tipe Pegawai": e["Tipe Pegawai"],
                        "Transaction Type": txn,
                        "Benefit Plan": plan_code,
                        "Benefit Note": PREFIX_NOTE[prefix],
                        "Beneficiary": "Pegawai",
                        "Claim Amount": 0.0,
                        "Benefit Limit": limit,
                    })

    zero_df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    n_unmapped_parea = int((emp["Tipe Pegawai"] == "Tidak Diketahui").sum())
    info = {
        "n_employees_master": len(employee_master),
        "n_zero_rows": len(zero_df),
        "n_unmapped_parea": n_unmapped_parea,
        "n_no_start_date": n_no_start_date,
        "n_joined_after_scope": n_joined_after_scope,
        "missing_limit_combos": sorted(missing_limit_combos),
        "used_overrides": sorted(used_overrides),
        "skipped_no_tier": skipped_no_tier,
    }
    return zero_df, info

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
# BUNDLED_PATH = "data.xlsx"

# if not os.path.exists(BUNDLED_PATH):
#     st.error(f"`{BUNDLED_PATH}` was not found next to app.py. Place your workbook there and rerun.")
#     st.stop()

# with open(BUNDLED_PATH, "rb") as f:
#     file_bytes = f.read()

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

salary_df = None
master_sheet = find_sheet(sheet_names, "employee")
if master_sheet:
    master_df = read_excel_sheet(file_bytes, master_sheet)
    salary_df = build_salary_df_from_master(master_df)

df = clean_claims(raw_df)

# st.sidebar.title("📁 Data")
# st.sidebar.caption(f"Loaded from `{BUNDLED_PATH}` — claims: sheet '{claims_sheet}'"
#                      + (f", salary: sheet '{master_sheet}'" if salary_df is not None else ""))
# if salary_df is None:
#     st.sidebar.warning("No employee salary sheet detected — Sections 4 & 5 will be unavailable.")

# --- Zero-fill: employees with no (or partial) transactions still count fully ---
uploaded_file_karyawan = st.sidebar.file_uploader(
    "Upload karyawan.xlsx",
    type=["xlsx"],
    key="karyawan"
)

if uploaded_file_karyawan is not None:
    karyawan_bytes = uploaded_file_karyawan.getvalue()
    karyawan_sheets = list_excel_sheets(karyawan_bytes)
    karyawan_sheet = karyawan_sheets[0]
    karyawan_raw = read_excel_sheet(karyawan_bytes, karyawan_sheet)
    employee_master = build_employee_master(karyawan_raw)

    if employee_master is not None:
        zero_df, info = build_zero_fill_rows(df, employee_master)

        df = pd.concat([df, zero_df], ignore_index=True, sort=False)
        
        if info["n_unmapped_parea"]:
            st.sidebar.caption(
                f"⚠️ {info['n_unmapped_parea']} employees have a PArea outside "
                f"AA/AD/AG (mapped to TPPBW/Talent Mobility/Digital Talent) — labeled 'Tidak Diketahui'."
            )
        if info["n_no_start_date"]:
            st.sidebar.caption(
                f"⚠️ {info['n_no_start_date']} employees have no readable Start date — "
                f"treated as eligible for all years (2023-2025) since join year is unknown."
            )
        # if info["n_joined_after_scope"]:
        #     st.sidebar.caption(
        #         f"ℹ️ {info['n_joined_after_scope']} employees joined after 2025 — correctly excluded from all zero-fill rows."
        #     )
        if info["missing_limit_combos"]:
            st.sidebar.caption(
                f"⚠️ No Provider limit could be inferred for: {', '.join(info['missing_limit_combos'])} "
                f"(no existing claim at that tier to learn the limit from) — those entitlements are excluded from Total Limit."
            )
        # if info["used_overrides"]:
        #     st.sidebar.caption(
        #         f"✅ Manual limit override used for: {', '.join(info['used_overrides'])}."
        #     )
        if info["skipped_no_tier"]:
            st.sidebar.caption(f"⚠️ {info['skipped_no_tier']} employees skipped (no PS group / tier).")

        # Also fold newly-discovered employees' salary into salary_df for Sections 4 & 5
        emp_salary = employee_master.rename(columns={"Monthly_Salary": "Monthly Salary"})[["NIK", "Monthly Salary"]]
        if salary_df is None:
            salary_df = emp_salary.dropna()
        else:
            salary_df = pd.concat([salary_df, emp_salary]).dropna().drop_duplicates(subset="NIK", keep="first")



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

# Reimbursement-policy potential saving (needs salary)
has_salary = salary_df is not None and not salary_df.empty
if has_salary:
    reimb_by_emp = (
        fdf.loc[fdf["Transaction Type"] == "Reimburse"]
        .groupby(["NIK", "Year"], as_index=False)["Claim Amount"].sum()
        .rename(columns={"Claim Amount": "Reimbursement Claim"})
    )
    reimb_by_emp = reimb_by_emp.merge(salary_df, on="NIK", how="left")
    reimb_by_emp["Monthly Salary"] = reimb_by_emp["Monthly Salary"].fillna(0)
    reimb_by_emp["Over Limit Amount"] = (
        reimb_by_emp["Reimbursement Claim"] - reimb_by_emp["Monthly Salary"]
    ).clip(lower=0)
    reimb_by_emp["Over Limit"] = reimb_by_emp["Over Limit Amount"] > 0
    n_over_reimb = int(reimb_by_emp["Over Limit"].sum())
    potential_saving = reimb_by_emp["Over Limit Amount"].sum()
else:
    n_over_reimb = None
    potential_saving = None

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

    tp = (
        fdf.groupby("Tipe Pegawai", as_index=False)["Claim Amount"]
        .sum()
    )

    fig = px.pie(
        tp,
        names="Tipe Pegawai",
        values="Claim Amount",
        hole=0.5,  # Donut chart
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
    "Talent Mobility": "#3498DB",
    "TPPBW": "#F39C12",
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

cols = st.columns(len(avg_claim_jenis))
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
        "This section needs each employee's **Monthly Salary**, which isn't part of the claims "
        "data. Upload a salary mapping file (columns: `NIK`, `Monthly Salary`) in the sidebar to unlock it."
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

# --------------------------------------------------------------------------
# SECTION 5 — DI MANA TERDAPAT POTENSI EFISIENSI BIAYA?
# --------------------------------------------------------------------------
st.markdown('<div class="section-header">⑤ APA DAMPAK POTENSIAL DARI SKENARIO KEBIJAKAN?</div>', unsafe_allow_html=True)

if not has_salary:
    st.warning("Upload the salary mapping file in the sidebar to run this simulation.")
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