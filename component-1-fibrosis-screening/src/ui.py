"""
ui.py  --  how the app looks. No model logic in here.

Colour carries meaning and nothing else:
    green  = no scan needed     amber = uncertain     crimson = refer for scan
Everything else is ink on clinical white.
"""
import os
import base64
import numpy as np
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

# Light colours: also used for the downloadable report, which is always printed on white
INK, MUTED, RULE, PAPER = "#18252F", "#56656D", "#D5DDDA", "#F2F5F4"
ZONE = {"NO SCAN": "#2E6B4E", "UNSURE": "#8F5B00", "SCAN": "#A12A4E"}
ZONE_TINT = {"NO SCAN": "#E3EFE8", "UNSURE": "#F6ECD9", "SCAN": "#F6E1E7"}

# The app follows Streamlit's light or dark theme. Every colour on screen is a CSS variable,
# set from one of these two palettes, so the same green/amber/crimson meaning holds in both.
PALETTE = {
    "light": {"ink": INK, "muted": MUTED, "faint": "#8A979D", "rule": RULE, "paper": "#EEF0EB", "on-ink": "#FFFFFF",
              "bg": "#F7F7F3", "field": "#FFFFFF", "field-line": "#D9DDD6",
              "accent": "#1E6E64", "accent-hover": "#175A52", "on-accent": "#FFFFFF", "accent-soft": "#E3EFEC",
              "z-noscan": ZONE["NO SCAN"], "z-unsure": ZONE["UNSURE"], "z-scan": ZONE["SCAN"],
              "t-noscan": ZONE_TINT["NO SCAN"], "t-unsure": ZONE_TINT["UNSURE"], "t-scan": ZONE_TINT["SCAN"]},
    "dark":  {"ink": "#E7ECEA", "muted": "#A3AEB4", "faint": "#6F7B81", "rule": "#2F3940", "paper": "#171D23",
              "on-ink": "#0E1117",
              "bg": "#0E1117", "field": "#1B2129", "field-line": "#2F3940",
              "accent": "#3FA592", "accent-hover": "#56B8A5", "on-accent": "#08201C", "accent-soft": "#13302B",
              "z-noscan": "#6CC497", "z-unsure": "#E5A93B", "z-scan": "#F07C9A",
              "t-noscan": "#152A20", "t-unsure": "#2F2513", "t-scan": "#35171F"},
}
_KEY = {"NO SCAN": "noscan", "UNSURE": "unsure", "SCAN": "scan"}
ZV = {z: f"var(--c1-z-{k})" for z, k in _KEY.items()}       # zone colour, for inline styles
TV = {z: f"var(--c1-t-{k})" for z, k in _KEY.items()}       # zone tint
V_MUTED = "var(--c1-muted)"


def theme():
    """'dark' or 'light': whichever theme Streamlit is showing (follows the computer's setting)."""
    try:
        fixed = st.get_option("theme.base")       # a theme pinned in .streamlit/config.toml wins
        if fixed in ("light", "dark"):
            return fixed
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:            # older Streamlit without st.context.theme
        return "light"


def zone_hex(z):
    """A real colour (not a CSS variable) for places that cannot read variables, like st.dataframe."""
    return PALETTE[theme()]["z-" + _KEY[z]]
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
.stApp {{ color: var(--c1-ink); background: var(--c1-bg); }}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"], footer {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}

/* inputs read like a lab request form */
[data-testid="stForm"] {{ border: 0; padding: 0; background: transparent; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ display: none; }}
/* no "Press Enter to submit form" hint drawn over the value */
[data-testid="InputInstructions"], .stApp [class*="InputInstructions"] {{ display: none !important; }}
/* every field the same: one height, one inset, one text size, so values line up in a column */
.stApp [data-testid="stNumberInputContainer"], .stApp [data-testid="stTextInputRootElement"],
.stApp .stSelectbox .react-aria-ComboBox > div {{ height: 44px; min-height: 44px; border-radius: 6px;
  background: var(--c1-field); box-shadow: inset 0 0 0 1px var(--c1-field-line); }}
.stApp [data-testid="stNumberInputContainer"]:focus-within, .stApp [data-testid="stTextInputRootElement"]:focus-within,
.stApp .stSelectbox .react-aria-ComboBox > div:focus-within {{ box-shadow: inset 0 0 0 2px var(--c1-accent); }}
.stApp [data-testid="stNumberInputField"], .stApp [data-testid="stTextInputField"],
.stApp .stSelectbox input {{ height: 44px; padding: 0 12px !important; font-size: 16px !important;
  line-height: 44px; font-variant-numeric: tabular-nums; }}
.stApp [data-testid="stTextInputField"]::placeholder {{ color: var(--c1-faint); }}
[data-testid="stWidgetLabel"] p {{ color: var(--c1-muted); font-size: 14px; }}
[data-testid="stWidgetLabel"] {{ min-height: 0; margin-bottom: 2px; }}
.stButton button, .stFormSubmitButton button, .stDownloadButton button {{
  border-radius: 6px; font-weight: 700; min-height: 46px; }}
.stButton button p, .stFormSubmitButton button p, .stDownloadButton button p {{ font-size: 16px; font-weight: 700; }}
/* buttons in the accent green (a different green from the "No scan" result colour) */
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {{
  background: var(--c1-accent); border-color: var(--c1-accent); color: var(--c1-on-accent); }}
.stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover {{
  background: var(--c1-accent-hover); border-color: var(--c1-accent-hover); color: var(--c1-on-accent); }}
.stApp button[kind="primary"] p, .stApp button[kind="primaryFormSubmit"] p {{ color: var(--c1-on-accent); }}
.stApp button[kind="secondary"], .stApp [data-testid="stDownloadButton"] button,
.stApp [data-testid="stBaseButton-secondary"] {{
  background: var(--c1-field); border: 1px solid var(--c1-accent); color: var(--c1-accent); }}
.stApp button[kind="secondary"] p, .stApp [data-testid="stDownloadButton"] button p {{ color: var(--c1-accent); }}
.stApp button[kind="secondary"]:hover, .stApp [data-testid="stDownloadButton"] button:hover {{
  background: var(--c1-accent-soft); border-color: var(--c1-accent); }}
/* tabs (when shown inside the shared app) */
.stApp [data-baseweb="tab-highlight"], .stApp .react-aria-SelectionIndicator {{ background-color: var(--c1-accent) !important; }}
.stApp [role="tab"][aria-selected="true"] p {{ color: var(--c1-accent) !important; font-weight: 700; }}
/* expanders sit on the page like cards */
.stApp [data-testid="stExpander"] details {{ background: var(--c1-field); border-color: var(--c1-field-line); }}
.stApp [role="tab"] p {{ font-size: 16px !important; }}

/* our own blocks */
.stApp .fx-title {{ font-size: 34px !important; font-weight: 700; line-height: 1.15 !important; margin: 0 0 8px; color: var(--c1-ink); letter-spacing: -0.01em; }}
.stApp .fx-lede {{ font-size: 17px !important; color: var(--c1-muted); margin: 0 0 22px; max-width: 62ch; }}
.fx-group {{ font-size: 15px; font-weight: 700; color: var(--c1-ink); margin: 18px 0 2px;
  padding-bottom: 4px; border-bottom: 1px solid var(--c1-rule); }}
.fx-empty {{ border: 1px dashed var(--c1-rule); border-radius: 8px; padding: 28px; color: var(--c1-muted);
  font-size: 16px; line-height: 1.5; }}
.fx-decision {{ border-left: 6px solid var(--z); background: var(--t); padding: 18px 22px;
  border-radius: 0 8px 8px 0; }}
.fx-decision h2 {{ font-size: 26px; line-height: 1.15; margin: 0; color: var(--z); font-weight: 700; padding: 0; }}
.fx-decision p {{ margin: 6px 0 0; font-size: 16px; color: var(--c1-ink); }}
.fx-risk {{ display: flex; align-items: baseline; gap: 14px; margin: 22px 0 18px; }}
.fx-risk b {{ font-size: 56px; line-height: 1; font-variant-numeric: tabular-nums; }}
.fx-risk span {{ color: var(--c1-muted); font-size: 16px; max-width: 30ch; line-height: 1.35; }}
.fx-scales {{ margin: 18px 0 8px; }}
.fx-row {{ display: grid; grid-template-columns: 112px 1fr; align-items: center; gap: 14px; margin: 0 0 26px; }}
.fx-row .who {{ font-size: 15px; font-weight: 700; }}
.fx-row .who small {{ display: block; font-weight: 400; color: var(--c1-muted); font-size: 13px; }}
.fx-track {{ position: relative; display: grid; grid-template-columns: 1fr 1fr 1fr; height: 14px; }}
.fx-track div {{ height: 14px; }}
.fx-mark {{ position: absolute; top: -9px; width: 4px; height: 32px; background: var(--c1-ink); border-radius: 2px;
  transform: translateX(-2px); }}
.fx-mark span {{ position: absolute; top: -24px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums; }}
.fx-zones {{ display: grid; grid-template-columns: 112px 1fr 1fr 1fr; gap: 0 14px; font-size: 13px; color: var(--c1-muted); }}
.fx-zones span:not(:first-child) {{ padding-left: 2px; }}
.fx-why {{ margin-top: 10px; }}
.fx-why-row {{ display: grid; grid-template-columns: 170px 1fr 92px; gap: 12px; align-items: center;
  padding: 7px 0; border-bottom: 1px solid var(--c1-rule); font-size: 15px; }}
.fx-why-row .bar {{ height: 10px; border-radius: 2px; }}
.fx-why-row .eff {{ font-size: 14px; text-align: right; }}
.fx-stack {{ display: flex; height: 26px; border-radius: 4px; overflow: hidden; margin: 6px 0 4px; }}
.fx-stack div {{ height: 26px; }}
.fx-key {{ font-size: 14px; color: var(--c1-muted); margin-bottom: 16px; }}
.fx-key b {{ color: var(--c1-ink); }}
[data-testid="stElementContainer"]:has(.fx-style) {{ display: none; }}
.stApp p.fx-err {{ color: var(--c1-z-scan); font-size: 13px; line-height: 1.35; margin: -6px 0 4px; }}
.fx-problems {{ border-left: 4px solid var(--z); background: var(--t); padding: 14px 18px;
  border-radius: 0 8px 8px 0; margin: 0 0 18px; }}
.fx-problems b.h {{ display: block; color: var(--z); font-size: 16px; margin-bottom: 6px; }}
.fx-problems ul {{ margin: 0; padding-left: 18px; }}
.fx-problems li {{ font-size: 14px; line-height: 1.45; margin: 2px 0; color: var(--c1-ink); }}
.fx-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 4px 0 26px; }}
.fx-stat {{ border-top: 3px solid var(--c1-ink); padding: 12px 0 0; }}
.fx-stat p.m {{ font-size: 14px; color: var(--c1-muted); margin: 0 0 6px; }}
.fx-stat .v {{ display: flex; gap: 26px; align-items: baseline; }}
.fx-stat .v span {{ font-size: 26px; font-variant-numeric: tabular-nums; color: var(--c1-muted); }}
.fx-stat .v b {{ font-size: 34px; color: var(--c1-ink); }}
.fx-stat small {{ display: block; font-size: 12px; color: var(--c1-muted); }}
.fx-stat p.n {{ font-size: 12.5px; color: var(--c1-muted); margin: 6px 0 0; }}
@media (max-width: 800px) {{ .fx-stats {{ grid-template-columns: 1fr; }} }}
.fx-prose {{ max-width: 68ch; font-size: 17px; line-height: 1.6; }}
</style>
"""
SIDEBAR_CSS = f"""
<style>
/* sidebar: a quiet navigation column */
[data-testid="stSidebar"] {{ background: var(--c1-paper); border-right: 1px solid var(--c1-rule); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: var(--c1-accent); font-weight: 700; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 6px 0; }}
[data-testid="stSidebar"] [role="radiogroup"] p {{ font-size: 16px; }}

</style>
"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def style(own_sidebar=True):
    """Fonts, colours and layout. own_sidebar=False inside the group app: leave its sidebar alone."""
    colours = ".stApp{" + ";".join(f"--c1-{k}:{v}" for k, v in PALETTE[theme()].items()) + "}"
    html(CSS.replace("FONTFACES", _font_face(400) + _font_face(700) + colours))
    if own_sidebar:
        html(SIDEBAR_CSS)


def title(text, lede=None):
    html(f'<p class="fx-title">{text}</p>' + (f'<p class="fx-lede">{lede}</p>' if lede else ""))


def group(text):
    html(f'<p class="fx-group">{text}</p>')


def decision(label, pid=""):
    who = f"{pid}: " if pid else ""
    html(f'<div class="fx-decision" style="--z:{ZV[label]};--t:{TV[label]}">'
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
        zones = "".join(f'<div style="background:{TV[z]};border-top:3px solid {ZV[z]}"></div>'
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
        f'background:{ZV["SCAN"] if v > 0 else ZV["NO SCAN"]}"></div></div>'
        f'<span class="eff" style="color:{ZV["SCAN"] if v > 0 else ZV["NO SCAN"]}">'
        f'{"raises risk" if v > 0 else "lowers risk"}</span></div>' for n, val, v in rows)
    html(f'<div class="fx-why">{body}</div>')


def stack(who, counts, total):
    """A single bar split into no scan / uncertain / scan, with the counts written out."""
    parts = "".join(f'<div style="width:{100 * counts[z] / total:.2f}%;background:{ZV[z]}"></div>'
                    for z in ("NO SCAN", "UNSURE", "SCAN") if counts[z])
    sent = counts["UNSURE"] + counts["SCAN"]
    html(f'<p class="fx-group" style="border:0;margin-top:14px">{who}</p>'
         f'<div class="fx-stack">{parts}</div>'
         f'<p class="fx-key"><b>{counts["NO SCAN"]}</b> no scan, <b>{counts["UNSURE"]}</b> uncertain, '
         f'<b>{counts["SCAN"]}</b> refer. Sent for FibroScan: <b>{sent} of {total}</b>.</p>')


def problems(heading, items, zone):
    """A short list of things to fix (zone SCAN, crimson) or to double-check (zone UNSURE, amber)."""
    lis = "".join(f"<li>{i}</li>" for i in items)
    html(f'<div class="fx-problems" style="--z:{ZV[zone]};--t:{TV[zone]}">'
         f'<b class="h">{heading}</b><ul>{lis}</ul></div>')


def mark_errors(keys):
    """Outline the boxes that have an error in crimson."""
    if keys:
        sel = ", ".join(f".st-key-{k} [data-testid='stNumberInputContainer'], "
                        f".st-key-{k} [data-testid='stTextInputRootElement'], "
                        f".st-key-{k} .react-aria-ComboBox > div" for k in keys)
        html(f"<div class='fx-style'></div><style>{sel} {{ box-shadow: inset 0 0 0 1.5px {ZV['SCAN']}; "
             f"background: {TV['SCAN']} !important; }}</style>")


def stats(rows):
    """Headline numbers: one row per measure, FIB-4 beside this tool."""
    body = "".join(
        f'<div class="fx-stat"><p class="m">{m}</p><div class="v"><span><small>FIB-4</small>{f}</span>'
        f'<span><small>This tool</small><b>{t}</b></span></div><p class="n">{n}</p></div>'
        for m, f, t, n in rows)
    html(f'<div class="fx-stats">{body}</div>')


def waterfall(sv, base, labels, top=12):
    """SHAP waterfall drawn in the app's own colours (follows light/dark theme).
    Bars are walked from the biggest effect to the smallest: each starts where the
    previous one ended, from the average patient's score to this patient's score."""
    import matplotlib.pyplot as plt
    pal = PALETTE[theme()]
    seq = list(np.argsort(-np.abs(sv))[:top])           # biggest effect first (drawn at the top)
    ends = np.cumsum([base] + [sv[j] for j in seq])     # running score after each bar
    span = (ends.max() - ends.min()) or 1.0
    ys = np.arange(len(seq))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(seq) + 1.2), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for y, j, left in zip(ys, seq, ends[:-1]):
        v = sv[j]
        c = pal["z-scan"] if v > 0 else pal["z-noscan"]
        ax.barh(y, v, left=left, height=0.62, color=c, edgecolor="none")
        ax.text(left + v + (0.015 if v >= 0 else -0.015) * span, y, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=c)
    for x0, name in [(ends[0], "average patient"), (ends[-1], "this patient")]:
        ax.axvline(x0, color=pal["muted"], lw=0.8, ls=(0, (3, 3)))
        ax.text(x0, len(seq) - 0.35, name, ha="center", va="bottom", fontsize=8, color=pal["muted"])

    ax.set_yticks(ys)
    ax.set_yticklabels([labels[j] for j in seq], fontsize=10, color=pal["ink"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(pal["rule"])
    ax.tick_params(axis="x", colors=pal["muted"], labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Model score (log-odds)", fontsize=8.5, color=pal["muted"])
    ax.set_xlim(ends.min() - 0.2 * span, ends.max() + 0.2 * span)
    ax.set_ylim(-0.7, len(seq) + 0.25)
    fig.tight_layout()
    return fig


def empty(text):
    html(f'<div class="fx-empty">{text}</div>')


def prose(md):
    html(f'<div class="fx-prose">{md}</div>')
