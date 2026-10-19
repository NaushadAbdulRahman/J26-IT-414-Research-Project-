"""
views.py  --  the screens of the Component 1 tool.

Used in two ways:
    run("sidebar")  the standalone app (src/app.py), pages chosen in the sidebar
    run("tabs")     one page inside the group's shared app (app/pages/1_...py),
                    with the five screens as tabs
How it looks lives in ui.py; how it decides lives in screening.py.
"""
import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import shap
import ui
import validate
from screening import (Screener, NAMES, OUTPUTS, fib4, fib4_zone, shown_value, parse_upload)

S = None          # the trained screener, set by run()
FEATURES = None


def K(name):
    """Prefix this component's session keys so they never clash with other components' pages."""
    return f"c1_{name}"


@st.cache_resource
def get_screener():
    return Screener()


# Example patients: real, de-identified NHANES test-set patients
EXAMPLES = {
    "Enter values myself": None,
    "Example A: low risk": dict(age=31, sex="Male", bmi=25.3, waist=94.9, diabetes="No", ast=23.0,
                                alt=26.0, ggt=25.0, platelets=416.0, albumin=4.1, hba1c=5.8, triglycerides=203.0),
    "Example B: tool uncertain": dict(age=72, sex="Male", bmi=27.6, waist=111.6, diabetes="No", ast=17.0,
                                      alt=13.0, ggt=12.0, platelets=239.0, albumin=4.1, hba1c=6.0, triglycerides=129.0),
    "Example C: FIB-4 uncertain, tool refers": dict(age=46, sex="Male", bmi=29.9, waist=104.1, diabetes="No", ast=62.0,
                                                    alt=99.0, ggt=72.0, platelets=140.0, albumin=4.5, hba1c=5.6, triglycerides=92.0),
}
DEFAULT = dict(age=55, sex="Male", bmi=29.0, waist=98.0, diabetes="No", ast=38.0, alt=42.0,
               ggt=45.0, platelets=210.0, albumin=4.2, hba1c=6.0, triglycerides=150.0)

def load_example():
    """Fill the form with an example patient, or empty it for 'Enter values myself'."""
    ex = EXAMPLES[st.session_state[K("example")]]
    for k in DEFAULT:
        st.session_state[K(k)] = ex[k] if ex else None
    st.session_state[K("pid")] = ""
    st.session_state[K("check")] = None          # forget the last check


def check_form():
    """Runs when 'Screen patient' is pressed, before the page redraws, so errors can sit under each box."""
    s = st.session_state
    st.session_state[K("check")] = validate.check({k: s[K(k)] for k in DEFAULT}, s.get(K("pid"), ""))


def field_error(col, field):
    """Show this box's error underneath it (and outline the box, see ui.mark_errors)."""
    chk = st.session_state.get(K("check"))
    if chk and field in chk[0]:
        msg = chk[0][field]
        if msg.startswith("Required"):
            msg = "Required"                              # the full reason is in the list on the right
        elif "looks like" in msg:
            msg = msg.split(":")[0] + "."                 # e.g. "42 looks like g/L. This box takes g/dL."
        col.markdown(f'<p class="fx-err">{msg}</p>', unsafe_allow_html=True)


def report_html(pid, x, label, p, f4, top, sv):
    rows = "".join(f"<tr><td>{NAMES[f]}</td><td>{shown_value(f, x.iloc[0][f])}</td></tr>" for f in FEATURES)
    why = "".join(f"<li>{NAMES[FEATURES[j]]} {shown_value(FEATURES[j], x.iloc[0, j])}: "
                  f"{'raises' if sv[j] > 0 else 'lowers'} risk</li>" for j in top)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Screening report {pid}</title>
<style>body{{font-family:Arial,sans-serif;max-width:680px;margin:40px auto;color:{ui.INK};line-height:1.5}}
h1{{font-size:22px;margin:0}} .d{{border-left:6px solid {ui.ZONE[label]};background:{ui.ZONE_TINT[label]};
padding:12px 16px;margin:18px 0}} .d b{{font-size:22px;color:{ui.ZONE[label]}}}
td{{padding:4px 14px 4px 0;border-bottom:1px solid #ddd}} small{{color:{ui.MUTED}}}</style></head><body>
<h1>Liver fibrosis screening report</h1>
<p>Patient: <b>{pid or "not given"}</b><br>Screened: {dt.datetime.now():%d %B %Y, %H:%M}</p>
<div class="d"><b>{ui.HEADLINE[label]}</b><br>{ui.DETAIL[label]}</div>
<p>Estimated risk of significant fibrosis: <b>{p:.0%}</b><br>
FIB-4: <b>{f4:.2f}</b> (on its own, FIB-4 would say: {fib4_zone(np.array([f4]))[0].lower()})</p>
<h2 style="font-size:17px">Main reasons (SHAP)</h2><ul>{why}</ul>
<h2 style="font-size:17px">Values entered</h2><table>{rows}</table>
<p><small>Research prototype, not for clinical use. Model: {S.model_name}, trained on
{S.data_source.upper()} data; not yet validated on local patients.</small></p></body></html>"""


def page_screen():
    ui.title("Does this patient need a FibroScan?",
             "Enter routine results from the patient's lab report. The tool compares its decision "
             "with FIB-4 and shows what drove it.")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.selectbox("Load an example patient", list(EXAMPLES), key=K("example"), on_change=load_example)
        # enter_to_submit=False: no "Press Enter to submit form" hint inside the boxes
        chk = st.session_state.get(K("check"))
        ui.mark_errors([K(f) for f in chk[0]] if chk else [])

        def number(col, field, label, decimals):
            """A number box with no hidden limits: every value is checked in validate.py instead,
            so a typing slip is reported, never silently changed."""
            need = "Required" if field in validate.REQUIRED else "Optional"
            col.number_input(label, value=None, step=1 if decimals == 0 and field == "age" else 10.0 ** -decimals,
                             format="%d" if field == "age" else f"%.{decimals}f",
                             placeholder=need, key=K(field))
            field_error(col, field)

        def choice(col, field, label, options):
            col.selectbox(label, options, index=None, placeholder="Required", key=K(field))
            field_error(col, field)

        with st.form(K("patient"), enter_to_submit=False, border=False):
            st.text_input("Patient ID (optional)", key=K("pid"), max_chars=20,
                          placeholder="Clinic number, not a name")
            field_error(st, "pid")
            ui.group("Patient")
            # one st.columns call per row, so every row lines up
            a, b, c = st.columns(3)
            number(a, "age", "Age, years", 0)
            choice(b, "sex", "Sex", ["Male", "Female"])
            choice(c, "diabetes", "Diabetes", ["No", "Yes"])
            a, b, _ = st.columns(3)
            number(a, "bmi", "BMI, kg/m²", 1)
            number(b, "waist", "Waist, cm", 1)
            ui.group("Liver tests")
            a, b, _ = st.columns(3)
            number(a, "ast", "AST, U/L", 0)
            number(b, "alt", "ALT, U/L", 0)
            a, b, _ = st.columns(3)
            number(a, "ggt", "GGT, U/L", 0)
            number(b, "albumin", "Albumin, g/dL", 1)
            ui.group("Blood count and metabolic")
            a, b, c = st.columns(3)
            number(a, "platelets", "Platelets, ×10⁹/L", 0)
            number(b, "hba1c", "HbA1c, %", 1)
            number(c, "triglycerides", "Triglycerides, mg/dL", 0)
            st.caption("Age, sex, diabetes, AST, ALT and platelets are required. "
                       "The others can be left blank if not measured.")
            submitted = st.form_submit_button("Screen patient", type="primary", width="stretch",
                                              on_click=check_form)

    with right:
        chk = st.session_state.get(K("check"))
        if not submitted or chk is None:
            ui.empty("The decision appears here.<br>Enter the patient's results and choose "
                     "<b>Screen patient</b>, or load one of the three example patients.")
            return
        errors, warnings, filled = chk
        if errors:
            n = len(errors)
            ui.problems(f"{n} {'box needs' if n == 1 else 'boxes need'} fixing before this patient can be screened",
                        [f"<b>{validate.LABEL[f]}</b>: {m}" for f, m in errors.items()], "SCAN")
            return

        s = st.session_state
        v = {k: s[K(k)] for k in DEFAULT}
        v["sex"] = 1 if v["sex"] == "Male" else 0
        v["diabetes"] = 1 if v["diabetes"] == "Yes" else 0
        x = pd.DataFrame([v])[FEATURES].astype(float)          # blanks become NaN; the model fills them in
        p = float(S.risk(x)[0])
        label = S.decide([p])[0]
        f4 = float(fib4(x).iloc[0])
        pid = s.get(K("pid"), "").strip()

        if warnings:
            ui.problems("Check these results", warnings, "UNSURE")
        if filled:
            st.caption(f"Not entered: {', '.join(filled)}. The model used a typical value for "
                       "these, so the risk is less precise.")
        ui.decision(label, pid)
        ui.risk(p)
        ui.group("Where this patient falls")
        ui.scales(f4, p, S.low, S.high)

        ui.group("Why: SHAP explanation")
        sv = S.reasons(x)[0]
        top = list(np.argsort(-np.abs(sv))[:5])
        ui.reasons([(NAMES[FEATURES[j]], shown_value(FEATURES[j], x.iloc[0, j]), sv[j]) for j in top])
        st.caption("Each row is a SHAP value: how far that result moved this patient's risk "
                   "compared with an average patient.")

        with st.expander("SHAP waterfall plot and all 12 values"):
            plt.figure()                     # fresh canvas for SHAP to draw on
            exp = shap.Explanation(values=sv, base_values=S.shap_base, data=x.iloc[0].values,
                                   feature_names=[NAMES[f] for f in FEATURES])
            shap.plots.waterfall(exp, max_display=12, show=False)
            st.pyplot(plt.gcf())
            plt.close("all")
            st.caption("Starts from the average patient, E[f(x)], and adds each result's SHAP "
                       "value to reach this patient's score, f(x). Scale: the model's log-odds "
                       "before calibration; the risk above is after calibration.")
            st.dataframe(pd.DataFrame({
                "Result": [NAMES[f] for f in FEATURES],
                "Value": [shown_value(f, x.iloc[0][f]) for f in FEATURES],
                "SHAP value": np.round(sv, 3),
                "Effect": ["raises risk" if v > 0 else "lowers risk" for v in sv],
            }).sort_values("SHAP value", key=abs, ascending=False), hide_index=True, width="stretch")

        st.download_button("Download report", report_html(pid, x, label, p, f4, top, sv),
                           file_name=f"screening_report_{pid or 'patient'}.html",
                           mime="text/html", on_click="ignore")
        st.session_state[K("history")].append({
            "Time": dt.datetime.now().strftime("%H:%M"), "Patient ID": pid or "-",
            "Decision": ui.HEADLINE[label], "Risk %": round(100 * p, 1), "FIB-4": round(f4, 2),
            "FIB-4 alone": str(fib4_zone(np.array([f4]))[0]).capitalize()})


def page_batch():
    ui.title("Screen a clinic list",
             "Upload a spreadsheet with one row per patient. You get a referral list with the "
             "patients who need a scan first.")
    template = pd.DataFrame([dict(patient_id="P001", **DEFAULT),
                             dict(patient_id="P002", **{**DEFAULT, "age": 64, "ast": 58.0, "platelets": 150.0})])
    c1, c2, _ = st.columns([1, 1.3, 1.2])
    c1.download_button("Download template", template.to_csv(index=False),
                       file_name="screening_template.csv", mime="text/csv", on_click="ignore", width="stretch")
    if c2.button("Use 25 sample NHANES patients", width="stretch"):
        st.session_state[K("use_sample")] = True
    up = st.file_uploader("Spreadsheet (CSV)", type="csv")

    df = None
    if up is not None:
        df = pd.read_csv(up)
    elif st.session_state.get(K("use_sample")):
        df = S.background.sample(25, random_state=1).reset_index(drop=True)
        df.insert(0, "patient_id", [f"S{i + 1:03d}" for i in range(len(df))])
        st.caption("Sample patients come from the training data, so they show how the page works, "
                   "not how accurate the tool is.")

    if df is None:
        ui.empty("No patients yet. Download the template, fill in one row per patient, and upload it.")
    else:
        try:
            data, _ = parse_upload(df, FEATURES)
        except ValueError as e:
            st.error(f"{e}. Download the template to see the expected columns.")
            return
        ids = (data["patient_id"].astype(str) if "patient_id" in data.columns
               else pd.Series([f"Row {i + 1}" for i in range(len(data))], index=data.index))
        # same checks as the single-patient form; rows with an error are not screened
        row_errors, row_warnings = validate.check_table(data)
        bad = pd.Series([e != "" for e in row_errors], index=data.index)
        if bad.any():
            ui.problems(f"{int(bad.sum())} of {len(data)} rows not screened. Fix them in the spreadsheet "
                        "and upload it again.",
                        [f"<b>{ids[i]}</b> (row {n + 1}): {row_errors[n]}"
                         for n, i in enumerate(data.index) if bad[i]][:20], "SCAN")
        if bad.all():
            return
        ok = data[~bad]
        result = S.screen_table(ok)
        result.insert(0, "Patient ID", ids[~bad].values)
        result["Check"] = [w for w, b in zip(row_warnings, bad) if not b]
        order = {"SCAN": 0, "UNSURE": 1, "NO SCAN": 2}
        result = result.sort_values(by=["Decision", "Risk %"],
                                    key=lambda c: c.map(order) if c.name == "Decision" else -c)
        n = len(result)
        ui.stack("This tool", {z: int((result["Decision"] == z).sum()) for z in order}, n)
        ui.stack("FIB-4 alone", {z: int((result["FIB-4 would say"] == z).sum()) for z in order}, n)

        shown = result.copy()
        shown["Decision"] = shown["Decision"].map(ui.HEADLINE)
        shown["FIB-4 would say"] = shown["FIB-4 would say"].map(ui.HEADLINE)

        def colour(val):
            for z, h in ui.HEADLINE.items():
                if val == h:
                    return f"color:{ui.ZONE[z]};font-weight:700"
            return ""
        styler = shown.style.format({"Risk %": "{:.1f}", "FIB-4": "{:.2f}"})
        styler = (styler.map if hasattr(styler, "map") else styler.applymap)(colour, subset=["Decision"])
        st.dataframe(styler, width="stretch", hide_index=True)
        st.download_button("Download referral list", result.to_csv(index=False),
                           file_name="referral_list.csv", mime="text/csv", on_click="ignore")


def page_history():
    ui.title("Patients screened today", "Kept only while the app is open. Nothing is saved to disk.")
    hist = pd.DataFrame(st.session_state[K("history")])
    if hist.empty:
        ui.empty("No patients screened yet. Screened patients appear here.")
    else:
        st.dataframe(hist, width="stretch", hide_index=True)
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("Download history", hist.to_csv(index=False),
                           file_name="session_history.csv", mime="text/csv", on_click="ignore", width="stretch")
        if c2.button("Clear history", width="stretch"):
            st.session_state[K("history")] = []
            st.rerun()


def page_performance():
    path = os.path.join(OUTPUTS, "results.json")
    if not os.path.exists(path):
        ui.title("Model performance")
        ui.empty("No results yet. Run <code>python src/run_all.py</code> to produce them.")
        return
    r = json.load(open(path))
    fc = r.get("fair_comparison", {})
    sick = round(r["fib4"]["sick_patients_missed"] / r["fib4"]["miss_rate_pct"] * 100) if r["fib4"]["miss_rate_pct"] else 0
    same = fc.get("same_scans_as_guideline_fib4")
    lede = (f"Tested on {r['n_test']:,} patients from {r['data_source'].upper()} that the model never saw, "
            f"{sick} of them with significant fibrosis. ")
    if same:
        lede += (f"With the same number of scans as FIB-4, FIB-4 missed {same['fib4']['miss_rate_pct']:.0f}% "
                 f"of them and this tool missed {same['model']['miss_rate_pct']:.0f}%.")
    ui.title("How it compares with FIB-4", lede)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        fp = os.path.join(OUTPUTS, "tradeoff_curve.png")
        if os.path.exists(fp):
            st.image(fp, width="stretch")
            st.caption("Every possible cut-off for each method. Lower and further left is better: "
                       "fewer sick patients missed for fewer scans.")
    with right:
        ui.group("Side by side")
        rows = [("AUROC", f"{r['auroc']['fib4']:.3f}", f"{r['auroc']['model']:.3f}")]
        if fc:
            tgt = fc["same_miss_target_10pct"]
            rows += [("Missed, same number of scans", f"{same['fib4']['miss_rate_pct']:.1f}%", f"{same['model']['miss_rate_pct']:.1f}%"),
                     ("Scanned, at most 10% missed", f"{tgt['fib4']['scans_pct']:.1f}%", f"{tgt['model']['scans_pct']:.1f}%"),
                     ("Missed, at most 10% target", f"{tgt['fib4']['miss_rate_pct']:.1f}%", f"{tgt['model']['miss_rate_pct']:.1f}%")]
        st.dataframe(pd.DataFrame(rows, columns=["Measure", "FIB-4", "This tool"]), hide_index=True, width="stretch")
        ui.group("Chosen model")
        cv = r.get("cross_validated_auroc_on_training_data", {})
        st.write(f"{r['chosen_model']}, picked from {len(cv)} models by cross-validation on the "
                 "training data. The simpler model is kept when scores are within 0.01.")
        st.dataframe(pd.DataFrame({"Model": list(cv), "Cross-validated AUROC": list(cv.values())}),
                     hide_index=True, width="stretch")

    for f, title, note in [
        ("calibration_curve.png", "Calibration: does a 20% risk mean 20%?",
         "Points on the dotted line mean the predicted risk matches what was observed."),
        ("decision_curve.png", "Decision curve: is it worth using?",
         "Higher is better. Net benefit weighs sick patients caught against unnecessary scans."),
        ("shap_summary.png", "Global SHAP: what drives the risk across all test patients",
         "One dot per patient. Further right means that result raised the risk more. "
         "Red is a high value, blue a low value.")]:
        fp = os.path.join(OUTPUTS, f)
        if os.path.exists(fp):
            with st.expander(title):
                st.image(fp, width="stretch")
                st.caption(note)


def page_about():
    ui.title("About this tool")
    ui.prose(f"""
<p><b>What it does.</b> Decides whether a patient with fatty liver disease should be referred for a
FibroScan, using 12 routine results. It gives a decision, an estimated risk, FIB-4 for comparison,
and the results that drove the decision.</p>
<p><b>How it decides.</b> A {S.model_name.lower()} model, chosen from three by cross-validation,
estimates the risk. The risk is calibrated, so 30% means about 30 in 100 such patients have
fibrosis. A conformal decision layer sets two cut-offs so that at most about 10% of sick patients
are cleared: no scan below {S.low:.1%} risk, scan above {S.high:.1%}, uncertain in between.</p>
<p><b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) splits one patient's prediction into
a contribution from each result. Starting from the average patient, each result pushes the score up
or down, and the pushes add up exactly to that patient's score.</p>
<p><b>Data.</b> Developed on NHANES 2017 to March 2020 (US CDC). Significant fibrosis means liver
stiffness of 8.0 kPa or more on FibroScan. To be recalibrated and validated on local patients.</p>
<p><b>Limits.</b> US general-population data, not Sri Lankan clinic patients. Stiffness is not
biopsy. Obesity affects both the model and the FibroScan reading. Not yet validated locally.</p>
<p><b>Research prototype. Not a medical device and not for clinical use.</b></p>
<p style="color:{ui.MUTED}">Component 1, project J26-IT-414, IT23229884</p>
""")


PAGES = [("Screen a patient", page_screen), ("Screen many patients", page_batch),
         ("Session history", page_history), ("Model performance", page_performance), ("About", page_about)]


def run(mode="sidebar"):
    global S, FEATURES
    ui.style()
    try:
        S = get_screener()
    except FileNotFoundError:
        st.error("No trained model found. Run `python src/run_all.py` in "
                 "component-1-fibrosis-screening, then reload this page.")
        return
    FEATURES = S.features
    st.session_state.setdefault(K("history"), [])
    for k in DEFAULT:                       # the form starts empty: nothing is pre-filled for a real patient
        st.session_state.setdefault(K(k), None)

    if mode == "sidebar":
        with st.sidebar:
            ui.html('<p style="font-size:20px;font-weight:700;margin:0 0 2px">Fibrosis referral</p>'
                    f'<p style="color:{ui.MUTED};font-size:14px;margin:0 0 18px">Who needs a FibroScan?</p>')
            choice = st.radio("Page", [name for name, _ in PAGES], label_visibility="collapsed")
            ui.html(f'<p style="color:{ui.MUTED};font-size:13px;line-height:1.5;margin-top:28px">'
                    f'Model: {S.model_name}, trained on {S.data_source.upper()}.<br>'
                    f'No scan below {S.low:.1%} risk, scan above {S.high:.1%}.<br><br>'
                    'Research prototype. Not for clinical use.</p>')
        dict(PAGES)[choice]()
    else:
        for tab, (_, page) in zip(st.tabs([name for name, _ in PAGES]), PAGES):
            with tab:
                page()
