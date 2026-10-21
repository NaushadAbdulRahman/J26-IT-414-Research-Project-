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
    "Example A: low risk": dict(age=23, sex="Male", bmi=27.6, waist=89.8, diabetes="No", ast=29.0,
                                alt=85.0, ggt=95.0, platelets=299.0, albumin=4.7, hba1c=5.4, triglycerides=94.0),
    "Example B: FIB-4 clears, tool unsure": dict(age=59, sex="Female", bmi=41.8, waist=116.8, diabetes="No",
                                                 ast=17.0, alt=22.0, ggt=28.0, platelets=284.0, albumin=4.4,
                                                 hba1c=5.6, triglycerides=115.0),
    "Example C: FIB-4 uncertain, tool refers": dict(age=70, sex="Male", bmi=34.6, waist=121.3, diabetes="Yes",
                                                    ast=26.0, alt=25.0, ggt=79.0, platelets=224.0, albumin=3.7,
                                                    hba1c=7.0, triglycerides=63.0),
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

        st.download_button("Download report", report_html(pid, x, label, p, f4, top, sv),
                           file_name=f"screening_report_{pid or 'patient'}.html",
                           mime="text/html", on_click="ignore")
        st.session_state[K("history")].append({
            "Time": dt.datetime.now().strftime("%H:%M"), "Patient ID": pid or "-",
            "Decision": ui.HEADLINE[label], "Risk %": round(100 * p, 1), "FIB-4": round(f4, 2),
            "FIB-4 alone": str(fib4_zone(np.array([f4]))[0]).capitalize()})

    # ---------- full-width section under the form: the complete SHAP explanation ----------
    with st.expander("Full SHAP explanation: waterfall and all 12 values", expanded=False):
        chart, table = st.columns([1.25, 1], gap="large")
        with chart:
            fig = ui.waterfall(sv, S.shap_base, [f"{NAMES[f]}  {shown_value(f, x.iloc[0][f])}" for f in FEATURES])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            st.caption("Starts from the average patient and adds each result's SHAP value to reach this "
                       "patient's score. Scale: the model's log-odds before calibration; the risk above "
                       "is after calibration.")
        with table:
            st.dataframe(pd.DataFrame({
                "Result": [NAMES[f] for f in FEATURES],
                "Value": [shown_value(f, x.iloc[0][f]) for f in FEATURES],
                "SHAP value": np.round(sv, 3),
                "Effect": ["raises risk" if v > 0 else "lowers risk" for v in sv],
            }).sort_values("SHAP value", key=abs, ascending=False), hide_index=True, width="stretch",
                height=35 * 12 + 38)


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
                    return f"color:{ui.zone_hex(z)};font-weight:700"
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
    if "guarantee_stress_test" not in r:
        ui.title("Model performance")
        ui.empty("These results are from an older run. Run <code>python src/run_all.py</code> again.")
        return
    ci = r["uncertainty_95ci"]
    who = "patients with fatty liver" if r.get("cohort") == "masld" else "adults"
    ui.title("How it compares with FIB-4",
             f"Tested on {r['n_test']:,} {who} from {r['data_source'].upper()} that the model never saw, "
             f"{r['n_test_with_fibrosis']} of them with significant fibrosis (FibroScan 8 kPa or more).")

    ui.stats([
        ("Sick patients missed", f"{r['fib4']['miss_rate_pct']:.0f}%", f"{r['model']['miss_rate_pct']:.0f}%",
         f"tool 95% CI {ci['model_miss_rate_pct_95ci'][0]:.0f} to {ci['model_miss_rate_pct_95ci'][1]:.0f}%"),
        ("Sent for a FibroScan", f"{r['fib4']['scans_needed_pct']:.0f}%", f"{r['model']['scans_needed_pct']:.0f}%",
         "tool includes the uncertain group"),
        ("AUROC (1.0 = perfect)", f"{r['auroc']['fib4']:.2f}", f"{r['auroc']['model']:.2f}",
         f"difference 95% CI {ci['auroc_difference_95ci'][0]:.2f} to {ci['auroc_difference_95ci'][1]:.2f}"),
    ])

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        fp = os.path.join(OUTPUTS, "tradeoff_curve.png")
        if os.path.exists(fp):
            st.image(fp, width="stretch")
            st.caption("Every possible cut-off for each method. Lower and further left is better: "
                       "fewer sick patients missed for fewer scans.")
    with right:
        fc = r["fair_comparison"]
        same, tgt = fc["same_scans_as_guideline_fib4"], fc["same_miss_target_10pct"]
        ui.group("Fair comparisons")
        st.dataframe(pd.DataFrame([
            ("Missed, same number of scans", f"{same['fib4']['miss_rate_pct']:.0f}%", f"{same['model']['miss_rate_pct']:.0f}%"),
            ("Scans needed, at most 10% missed", f"{tgt['fib4']['scans_pct']:.0f}%", f"{tgt['model']['scans_pct']:.0f}%"),
            ("Missed, FIB-4 age-adjusted rule", f"{fc['fib4_age_adjusted']['miss_rate_pct']:.0f}%", "-"),
        ], columns=["Comparison", "FIB-4", "This tool"]), hide_index=True, width="stretch")
        st.caption("Cut-offs are set on separate calibration patients, never on the test patients.")
        g = fc["fib4_grey_zone"]
        ui.group("FIB-4's uncertain zone")
        st.write(f"FIB-4 was unsure about {g['patients']} patients. The tool cleared {g['model_cleared']} of them "
                 f"({g['sick_among_model_cleared']} sick) and referred {g['model_referred']} "
                 f"({g['sick_among_model_referred']} sick).")
        ui.group("Chosen model")
        st.write(f"{r['chosen_model']}, picked from 3 by cross-validation on the training data "
                 f"(simplest within 0.01). Calibration: {r.get('calibration_method', 'isotonic')}, also chosen "
                 "on training data.")

    ui.group("Research questions")
    stress = r["guarantee_stress_test"]
    cvd = r["composite_vs_decomposed"]
    sa = r["shap_alignment"]
    top = ", ".join(NAMES[x["feature"]] for x in sa["ranking"][:sa["top_k"]])
    budget = {row["budget_pct"]: row for row in r["scan_budget"]}
    b40 = budget.get(40) or r["scan_budget"][len(r["scan_budget"]) // 2]
    answers = [
        ("RQ1. Is the safety bound kept?",
         f"Yes, on average. Over {stress['repeats']} new splits the tool missed {stress['mean_miss_rate_pct']:.1f}% "
         f"of sick patients on average (allowed {stress['allowed_miss_rate_pct']:.0f}%). Single splits vary "
         f"({stress['miss_rate_pct_range_5_to_95'][0]:.0f} to {stress['miss_rate_pct_range_5_to_95'][1]:.0f}%) "
         "because each holds only about 100 sick patients.", "guarantee_stress_test.png"),
        ("RQ2. At that safety level, fewer scans than FIB-4?",
         f"Not fewer than guideline FIB-4, which scans only {r['fib4']['scans_needed_pct']:.0f}% but misses "
         f"{r['fib4']['miss_rate_pct']:.0f}% of sick patients. To miss at most 10%, FIB-4 would need "
         f"{tgt['fib4']['scans_pct']:.0f}% scans; the tool needs {tgt['model']['scans_pct']:.0f}%. "
         f"With scans limited to {b40['budget_pct']}%, the tool misses {b40['model']['miss_rate_pct']:.0f}% "
         f"against FIB-4's {b40['fib4']['miss_rate_pct']:.0f}%.", "scan_budget.png"),
        ("RQ3. Do FIB-4's parts work better than the score?",
         "No clear difference. Test AUROC: " + "; ".join(f"{k.split(':')[0]} {v['test_auroc']:.2f}" for k, v in cvd.items())
         + ". The gain comes from adding the other results, not from splitting FIB-4.", None),
        ("RQ4. Do the explanations make clinical sense?",
         f"Yes. The top {sa['top_k']} drivers by SHAP are {top}: {sa['recognised_in_top_k']} of {sa['top_k']} are "
         "recognised fibrosis risk factors.", "shap_summary.png"),
    ]
    for q, a_text, img in answers:
        with st.expander(q, expanded=False):
            st.write(a_text)
            if q.startswith("RQ3"):
                st.dataframe(pd.DataFrame([{"Inputs": k, "CV AUROC (training)": v["cv_auroc_training"],
                                            "Test AUROC": v["test_auroc"], "Test Brier": v["test_brier"]}
                                           for k, v in cvd.items()]), hide_index=True, width="stretch")
            if img and os.path.exists(os.path.join(OUTPUTS, img)):
                st.image(os.path.join(OUTPUTS, img), width=620)

    ui.group("Benchmarks and checks")
    ub, cal, sens, ext = r["uncalibrated_benchmark"], r["calibration"], r.get("sensitivity_all_adults"), r.get("external_validation")
    rows = [("Uncalibrated model, usual 50% cut-off", f"{ub['default_cutoff_0_5']['scans_pct']:.0f}%",
             f"{ub['default_cutoff_0_5']['miss_rate_pct']:.0f}%"),
            ("Uncalibrated model + decision layer", f"{ub['with_decision_layer']['scans_pct']:.0f}%",
             f"{ub['with_decision_layer']['miss_rate_pct']:.0f}%"),
            ("This tool (calibrated + decision layer)", f"{r['model']['scans_needed_pct']:.0f}%",
             f"{r['model']['miss_rate_pct']:.0f}%")]
    if sens:
        rows.append((f"All adults, not only fatty liver ({sens['patients']:,})",
                     f"{sens['model_three_zone']['scans_pct']:.0f}%", f"{sens['model_three_zone']['miss_rate_pct']:.0f}%"))
    if ext:
        rows.append((f"Second NHANES dataset 2021-2023 ({ext['patients']:,})",
                     f"{ext['model_three_zone']['scans_pct']:.0f}%", f"{ext['model_three_zone']['miss_rate_pct']:.0f}%"))
    st.dataframe(pd.DataFrame(rows, columns=["Setting", "Scanned", "Sick missed"]), hide_index=True, width="stretch")
    if not ext:
        st.caption("External check on the second NHANES dataset (2021-2023): ready to run once its files are in "
                   "data/nhanes_2021/.")
    st.dataframe(pd.DataFrame([
        {"": "Before calibration", **{k.replace("_", " ").capitalize(): v for k, v in cal["uncalibrated"].items()}},
        {"": "After calibration", **{k.replace("_", " ").capitalize(): v for k, v in cal["calibrated"].items()}},
    ]), hide_index=True, width="stretch")
    st.caption("Brier score and ECE: lower is better. Calibration slope: 1 is ideal; intercept: 0 is ideal.")

    for f, title, note in [
        ("calibration_curve.png", "Calibration: does a 20% risk mean 20%?",
         "Points on the dotted line mean the predicted risk matches what was observed."),
        ("decision_curve.png", "Decision curve: is it worth using?",
         "Higher is better. Net benefit weighs sick patients caught against unnecessary scans.")]:
        fp = os.path.join(OUTPUTS, f)
        if os.path.exists(fp):
            with st.expander(title):
                st.image(fp, width=620)
                st.caption(note)


def page_about():
    r = {}
    path = os.path.join(OUTPUTS, "results.json")
    if os.path.exists(path):
        r = json.load(open(path))
    ui.title("About this tool")
    ui.prose(f"""
<p><b>What it does.</b> Decides whether a patient with fatty liver disease (MASLD) should be referred
for a FibroScan, using 12 routine results. It gives a decision, an estimated risk, FIB-4 for comparison,
and the results that drove the decision.</p>
<p><b>How it decides.</b> A {S.model_name.lower()} model, chosen from three by cross-validation,
estimates the risk, and is then calibrated so that 30% means about 30 in 100 such patients have
fibrosis. A conformal decision layer sets two cut-offs so that on average at most 10% of sick patients
are cleared: no scan below {S.low:.1%} risk, scan above {S.high:.1%}, uncertain in between.
Uncertain patients are referred, to be safe.</p>
<p><b>What is SHAP?</b> SHAP (SHapley Additive exPlanations) splits one patient's prediction into
a contribution from each result. Starting from the average patient, each result pushes the score up
or down, and the pushes add up exactly to that patient's score.</p>
<p><b>Data.</b> NHANES 2017 to March 2020 (US CDC): {r.get('n_patients', 0):,} adults with fatty liver
(CAP 275 dB/m or more) and at least one metabolic risk factor, with a reliable FibroScan. Significant
fibrosis means liver stiffness of 8.0 kPa or more. Split 60/20/20 into training, calibration and test.</p>
<p><b>Checks on what you type.</b> Required results, possible ranges, common unit mix-ups
(g/L vs g/dL, mmol/mol vs %, mmol/L vs mg/dL) and a few cross-checks run before every screening.</p>
<p><b>Limits.</b> US survey data, not Sri Lankan clinic patients. MASLD is approximated (alcohol and
viral hepatitis are not checked). Stiffness is not biopsy. Obesity affects both the model and the
FibroScan reading. To be recalibrated and validated on local patients.</p>
<p><b>Research prototype. Not a medical device and not for clinical use.</b></p>
<p style="color:{ui.V_MUTED}">Component 1, project J26-IT-414, IT23229884</p>
""")


PAGES = [("Screen a patient", page_screen), ("Screen many patients", page_batch),
         ("Session history", page_history), ("Model performance", page_performance), ("About", page_about)]


def run(mode="sidebar"):
    global S, FEATURES
    ui.style(own_sidebar=(mode == "sidebar"))
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
                    f'<p style="color:{ui.V_MUTED};font-size:14px;margin:0 0 18px">Who needs a FibroScan?</p>')
            choice = st.radio("Page", [name for name, _ in PAGES], label_visibility="collapsed")
            ui.html(f'<p style="color:{ui.V_MUTED};font-size:13px;line-height:1.5;margin-top:28px">'
                    f'Model: {S.model_name}, trained on {S.data_source.upper()}.<br>'
                    f'No scan below {S.low:.1%} risk, scan above {S.high:.1%}.<br><br>'
                    'Research prototype. Not for clinical use.</p>')
        dict(PAGES)[choice]()
    else:
        for tab, (_, page) in zip(st.tabs([name for name, _ in PAGES]), PAGES):
            with tab:
                page()
