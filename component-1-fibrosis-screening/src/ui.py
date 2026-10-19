"""
ui.py  --  how the app looks. No model logic in here.

Colour carries meaning and nothing else:
    green  = no scan needed     amber = uncertain     crimson = refer for scan
Everything else is ink on clinical white.
"""
import os
import base64
import streamlit as st

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _font_face(weight):
    """Embed the bundled Atkinson Hyperlegible file so the app looks the same offline."""
    path = os.path.join(FONT_DIR, f"atkinson-hyperlegible-latin-{weight}-normal.woff2")
    if not os.path.exists(path):
        return ""
    data = base64.b64encode(open(path, "rb").read()).decode()
    return (f"@font-face{{font-family:'Atkinson Hyperlegible';font-weight:{weight};font-style:normal;"
            f"src:url(data:font/woff2;base64,{data}) format('woff2');}}")

INK, MUTED, RULE, PAPER = "#18252F", "#56656D", "#D5DDDA", "#F2F5F4"
ZONE = {"NO SCAN": "#2E6B4E", "UNSURE": "#8F5B00", "SCAN": "#A12A4E"}
ZONE_TINT = {"NO SCAN": "#E3EFE8", "UNSURE": "#F6ECD9", "SCAN": "#F6E1E7"}
HEADLINE = {"NO SCAN": "No scan needed now",
            "UNSURE": "Uncertain: refer to be safe",
            "SCAN": "Refer for FibroScan"}
DETAIL = {"NO SCAN": "Risk is below the rule-out threshold. Re-screen at the next routine visit.",
          "UNSURE": "The tool cannot confidently rule fibrosis in or out for this patient.",
          "SCAN": "Risk is above the referral threshold."}

CSS = f"""
<style>
FONTFACES
.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea,
.stApp li, .stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"] *, [data-baseweb="select"] div {{
  font-family: 'Atkinson Hyperlegible', system-ui, -apple-system, sans-serif;
}}
.stApp {{ color: {INK}; }}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stDecoration"], #MainMenu, footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}

/* sidebar: a quiet navigation column */
[data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 6px 0; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-size: 16px; }}

/* inputs read like a lab request form */
[data-testid="stForm"] {{ border: 0; padding: 0; background: transparent; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ display: none; }}
/* no "Press Enter to submit form" hint drawn over the value */
[data-testid="InputInstructions"] {{ display: none !important; }}
/* every field the same: one height, one inset, one text size, so values line up in a column */
.stApp [data-testid="stNumberInputContainer"], .stApp [data-testid="stTextInputRootElement"],
.stApp .stSelectbox .react-aria-ComboBox > div {{ height: 44px; min-height: 44px; border-radius: 6px; }}
.stApp [data-testid="stNumberInputField"], .stApp [data-testid="stTextInputField"],
.stApp .stSelectbox input {{ height: 44px; padding: 0 12px !important; font-size: 16px !important;
  line-height: 44px; font-variant-numeric: tabular-nums; }}
.stApp [data-testid="stTextInputField"]::placeholder {{ color: #8A979D; }}
[data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-size: 14px; }}
[data-testid="stWidgetLabel"] {{ min-height: 0; margin-bottom: 2px; }}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {{
  border-radius: 6px; font-weight: 700; min-height: 46px; }}
.stButton button p, .stFormSubmitButton button p, .stDownloadButton button p {{ font-size: 16px; font-weight: 700; }}
/* primary button in ink, also when the shared app's theme is used */
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {{
  background: {INK}; border-color: {INK}; color: #FFFFFF; }}
/* tabs (when shown inside the shared app) */
.stApp [data-baseweb="tab-highlight"], .stApp .react-aria-SelectionIndicator {{ background-color: {INK} !important; }}
.stApp [role="tab"][aria-selected="true"] p {{ color: {INK} !important; font-weight: 700; }}
.stApp [role="tab"] p {{ font-size: 16px !important; }}

/* our own blocks */
.stApp .fx-title {{ font-size: 34px !important; font-weight: 700; line-height: 1.15 !important; margin: 0 0 8px; color: {INK}; letter-spacing: -0.01em; }}
.stApp .fx-lede {{ font-size: 17px !important; color: {MUTED}; margin: 0 0 22px; max-width: 62ch; }}
.fx-group {{ font-size: 15px; font-weight: 700; color: {INK}; margin: 18px 0 2px;
  padding-bottom: 4px; border-bottom: 1px solid {RULE}; }}
.fx-empty {{ border: 1px dashed {RULE}; border-radius: 8px; padding: 28px; color: {MUTED};
  font-size: 16px; line-height: 1.5; }}
.fx-decision {{ border-left: 6px solid var(--z); background: var(--t); padding: 18px 22px;
  border-radius: 0 8px 8px 0; }}
.fx-decision h2 {{ font-size: 26px; line-height: 1.15; margin: 0; color: var(--z); font-weight: 700; padding: 0; }}
.fx-decision p {{ margin: 6px 0 0; font-size: 16px; color: {INK}; }}
.fx-risk {{ display: flex; align-items: baseline; gap: 14px; margin: 22px 0 18px; }}
.fx-risk b {{ font-size: 56px; line-height: 1; font-variant-numeric: tabular-nums; }}
.fx-risk span {{ color: {MUTED}; font-size: 16px; max-width: 30ch; line-height: 1.35; }}
.fx-scales {{ margin: 18px 0 8px; }}
.fx-row {{ display: grid; grid-template-columns: 112px 1fr; align-items: center; gap: 14px; margin: 0 0 26px; }}
.fx-row .who {{ font-size: 15px; font-weight: 700; }}
.fx-row .who small {{ display: block; font-weight: 400; color: {MUTED}; font-size: 13px; }}
.fx-track {{ position: relative; display: grid; grid-template-columns: 1fr 1fr 1fr; height: 14px; }}
.fx-track div {{ height: 14px; }}
.fx-mark {{ position: absolute; top: -9px; width: 4px; height: 32px; background: {INK}; border-radius: 2px;
  transform: translateX(-2px); }}
.fx-mark span {{ position: absolute; top: -24px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums; }}
.fx-zones {{ display: grid; grid-template-columns: 112px 1fr 1fr 1fr; gap: 0 14px; font-size: 13px; color: {MUTED}; }}
.fx-zones span:not(:first-child) {{ padding-left: 2px; }}
.fx-why {{ margin-top: 10px; }}
.fx-why-row {{ display: grid; grid-template-columns: 170px 1fr 92px; gap: 12px; align-items: center;
  padding: 7px 0; border-bottom: 1px solid {RULE}; font-size: 15px; }}
.fx-why-row .bar {{ height: 10px; border-radius: 2px; }}
.fx-why-row .eff {{ font-size: 14px; text-align: right; }}
.fx-stack {{ display: flex; height: 26px; border-radius: 4px; overflow: hidden; margin: 6px 0 4px; }}
.fx-stack div {{ height: 26px; }}
.fx-key {{ font-size: 14px; color: {MUTED}; margin-bottom: 16px; }}
.fx-key b {{ color: {INK}; }}
.fx-prose {{ max-width: 68ch; font-size: 17px; line-height: 1.6; }}
</style>
"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def style():
    html(CSS.replace("FONTFACES", _font_face(400) + _font_face(700)))


def title(text, lede=None):
    html(f'<p class="fx-title">{text}</p>' + (f'<p class="fx-lede">{lede}</p>' if lede else ""))


def group(text):
    html(f'<p class="fx-group">{text}</p>')


def decision(label, pid=""):
    who = f"{pid}: " if pid else ""
    html(f'<div class="fx-decision" style="--z:{ZONE[label]};--t:{ZONE_TINT[label]}">'
         f'<h2>{who}{HEADLINE[label]}</h2><p>{DETAIL[label]}</p></div>')


def risk(p):
    html(f'<div class="fx-risk"><b>{p:.0%}</b><span>estimated risk of significant liver '
         f'fibrosis (stage F2 or worse)</span></div>')


def _pos(v, a, b, hi):
    """Place a value on a three-zone scale where each zone gets one third of the width."""
    if v < a:
        return v / a / 3
    if v <= b:
        return 1 / 3 + (v - a) / (b - a) / 3
    return 2 / 3 + min(v - b, hi - b) / (hi - b) / 3


def scales(f4, p, low, high):
    """Two aligned scales: FIB-4 on top, this tool below, same three zones."""
    def row(who, sub, x, text):
        zones = "".join(f'<div style="background:{ZONE_TINT[z]};border-top:3px solid {ZONE[z]}"></div>'
                        for z in ("NO SCAN", "UNSURE", "SCAN"))
        return (f'<div class="fx-row"><div class="who">{who}<small>{sub}</small></div>'
                f'<div class="fx-track">{zones}<div class="fx-mark" style="left:{100 * x:.1f}%">'
                f'<span>{text}</span></div></div></div>')
    html('<div class="fx-scales">'
         + row("FIB-4", "current test", _pos(f4, 1.30, 2.67, 5.0), f"{f4:.2f}")
         + row("This tool", "risk", _pos(p, low, high, 0.6), f"{p:.0%}")
         + f'<div class="fx-zones"><span></span><span>No scan</span><span>Uncertain</span><span>Scan</span></div>'
         + f'<div class="fx-zones"><span></span><span>FIB-4 &lt; 1.30<br>risk &lt; {low:.1%}</span>'
           f'<span>1.30 to 2.67<br>{low:.1%} to {high:.1%}</span>'
           f'<span>FIB-4 &gt; 2.67<br>risk &gt; {high:.1%}</span></div></div>')


def reasons(rows):
    """rows: list of (name, shown value, shap value). Bars drawn to scale, words say the direction."""
    top = max(abs(v) for _, _, v in rows) or 1
    body = "".join(
        f'<div class="fx-why-row"><span><b>{n}</b> {val}</span>'
        f'<div><div class="bar" style="width:{max(3, 100 * abs(v) / top):.0f}%;'
        f'background:{ZONE["SCAN"] if v > 0 else ZONE["NO SCAN"]}"></div></div>'
        f'<span class="eff" style="color:{ZONE["SCAN"] if v > 0 else ZONE["NO SCAN"]}">'
        f'{"raises risk" if v > 0 else "lowers risk"}</span></div>' for n, val, v in rows)
    html(f'<div class="fx-why">{body}</div>')


def stack(who, counts, total):
    """A single bar split into no scan / uncertain / scan, with the counts written out."""
    parts = "".join(f'<div style="width:{100 * counts[z] / total:.2f}%;background:{ZONE[z]}"></div>'
                    for z in ("NO SCAN", "UNSURE", "SCAN") if counts[z])
    sent = counts["UNSURE"] + counts["SCAN"]
    html(f'<p class="fx-group" style="border:0;margin-top:14px">{who}</p>'
         f'<div class="fx-stack">{parts}</div>'
         f'<p class="fx-key"><b>{counts["NO SCAN"]}</b> no scan, <b>{counts["UNSURE"]}</b> uncertain, '
         f'<b>{counts["SCAN"]}</b> refer. Sent for FibroScan: <b>{sent} of {total}</b>.</p>')


def empty(text):
    html(f'<div class="fx-empty">{text}</div>')


def prose(md):
    html(f'<div class="fx-prose">{md}</div>')
