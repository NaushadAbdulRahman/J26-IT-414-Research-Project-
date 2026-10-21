"""c2_ui.py  --  how the Failure-Finder page looks. No audit logic in here.

Fonts, colours, inputs and buttons come from the group's shared look
(Component 1's ui.py), so every page of the app looks like one product.
This file only adds the blocks the Failure-Finder needs.

Colour carries meaning:
    crimson = sent home but worth a scan (red flag)
    green   = no red flag
    teal    = found by a scan / already sent for a scan
"""
from html import escape

import streamlit as st

try:
    import ui as group_ui        # Component 1's shared look (fonts, colours, buttons)
except Exception:                # the page still works without it, only plainer
    group_ui = None

# Used only if the shared look cannot be loaded (same light colours as Component 1)
FALLBACK = {"ink": "#18252F", "muted": "#56656D", "faint": "#8A979D", "rule": "#D5DDDA",
            "paper": "#EEF0EB", "on-ink": "#FFFFFF", "bg": "#F7F7F3", "field": "#FFFFFF",
            "field-line": "#D9DDD6", "accent": "#1E6E64", "on-accent": "#FFFFFF",
            "accent-soft": "#E3EFEC", "z-noscan": "#2E6B4E", "z-unsure": "#8F5B00",
            "z-scan": "#A12A4E", "t-noscan": "#E3EFE8", "t-unsure": "#F6ECD9", "t-scan": "#F6E1E7"}

TRAIT_ORDER = ["age", "sex", "bmi", "ast", "alt", "platelets"]
TRAIT_NAMES = {"age": "Age", "bmi": "BMI", "ast": "AST", "alt": "ALT", "platelets": "Platelets"}

# (border colour, background tint) for each kind of verdict box
KINDS = {"flag":  ("var(--c1-z-scan)", "var(--c1-t-scan)"),
         "error": ("var(--c1-z-scan)", "var(--c1-t-scan)"),
         "clear": ("var(--c1-z-noscan)", "var(--c1-t-noscan)"),
         "scan":  ("var(--c1-accent)", "var(--c1-accent-soft)")}

CSS = """<style>
.stApp [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child{background-color:var(--c1-accent) !important;border-color:var(--c1-accent) !important}
.ff-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:8px 0 18px}
.ff-tile{border-top:3px solid var(--c1-ink);padding-top:10px}
.ff-tile .k{font-size:14px;color:var(--c1-muted)}
.ff-tile .v{font-size:34px;font-weight:700;line-height:1.15;font-variant-numeric:tabular-nums;color:var(--c1-ink)}
.ff-tile .n{font-size:13px;color:var(--c1-muted);line-height:1.4;margin-top:2px}
.ff-tile.bad{border-top-color:var(--c1-z-scan)}
.ff-tile.bad .v{color:var(--c1-z-scan)}
.ff-split{display:flex;height:30px;border-radius:4px;overflow:hidden;margin:4px 0 6px}
.ff-split div{height:30px;display:flex;align-items:center;padding:0 10px;font-size:13px;font-weight:700;
  color:var(--c1-on-ink);white-space:nowrap;overflow:hidden}
.ff-key{font-size:15px;color:var(--c1-muted);margin:0 0 22px;line-height:1.45}
.ff-key b{color:var(--c1-ink)}
.ff-verdict{border-left:6px solid var(--z);background:var(--t);padding:16px 22px;border-radius:0 8px 8px 0;margin:4px 0 18px}
.ff-verdict .h{font-size:24px;line-height:1.2;font-weight:700;color:var(--z)}
.ff-verdict .p{margin-top:6px;font-size:16px;color:var(--c1-ink);line-height:1.45}
.ff-rows{margin:2px 0 8px}
.ff-row{display:grid;grid-template-columns:minmax(190px,1.2fr) 2fr 130px 110px;gap:16px;align-items:center;
  padding:9px 0;border-bottom:1px solid var(--c1-rule);font-size:15px;color:var(--c1-ink)}
.ff-row.head{font-size:13px;color:var(--c1-muted);border-bottom:1px solid var(--c1-ink);padding:0 0 6px;align-items:end}
.ff-row .g{font-weight:700}
.ff-row .num{text-align:right;font-variant-numeric:tabular-nums}
.ff-row .num small{display:block;color:var(--c1-muted);font-size:12px;font-weight:400}
.ff-rows.compact .ff-row{grid-template-columns:1fr 120px 90px;grid-template-areas:"g n p" "t t t";row-gap:6px}
.ff-rows.compact .g{grid-area:g}.ff-rows.compact .t{grid-area:t}.ff-rows.compact .n{grid-area:n}.ff-rows.compact .p{grid-area:p}
.ff-track{position:relative;height:14px;background:var(--c1-paper);border-radius:3px}
.ff-fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:var(--z)}
.ff-line{position:absolute;top:-6px;bottom:-6px;width:2px;background:var(--c1-ink);transform:translateX(-1px)}
.ff-scale{position:relative;height:34px}
.ff-scale span{position:absolute;bottom:0;transform:translateX(-50%);font-size:12px;color:var(--c1-muted);white-space:nowrap}
.ff-scale span.bar{bottom:16px;color:var(--c1-ink);font-weight:700}
.ff-note{font-size:15px;color:var(--c1-muted);margin:8px 0 14px;line-height:1.45;max-width:90ch}
.ff-note b{color:var(--c1-ink)}
.ff-chips{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 16px}
.ff-chip{font-size:13px;padding:3px 10px;border-radius:999px;border:1px solid var(--c1-field-line);color:var(--c1-ink);background:var(--c1-field)}
.ff-chip.on{border-color:var(--c1-z-scan);color:var(--c1-z-scan);font-weight:700;background:var(--c1-t-scan)}
.ff-label{font-size:14px;color:var(--c1-muted);margin:10px 0 2px}
.ff-score{display:flex;align-items:baseline;gap:14px;margin:6px 0 14px}
.ff-score b{font-size:52px;line-height:1;font-variant-numeric:tabular-nums;color:var(--c1-ink)}
.ff-score span{color:var(--c1-muted);font-size:15px;line-height:1.35}
.ff-empty{border:1px dashed var(--c1-rule);border-radius:8px;padding:28px;color:var(--c1-muted);font-size:16px;line-height:1.5}
.ff-steps{counter-reset:s;margin:8px 0 20px;padding:0;list-style:none;max-width:80ch}
.ff-steps li{counter-increment:s;position:relative;padding:0 0 16px 46px;font-size:16px;line-height:1.45;color:var(--c1-ink);font-weight:700}
.ff-steps li::before{content:counter(s);position:absolute;left:0;top:-2px;width:30px;height:30px;border-radius:50%;
  background:var(--c1-accent);color:var(--c1-on-accent);font-weight:700;font-size:14px;display:flex;align-items:center;justify-content:center}
.ff-steps li small{display:block;color:var(--c1-muted);font-size:15px;font-weight:400}
.ff-evidence{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin:8px 0 18px}
.ff-ev{border:1px solid var(--c1-field-line);border-radius:8px;padding:14px 16px;background:var(--c1-field)}
.ff-ev .k{font-size:13px;color:var(--c1-muted)}
.ff-ev .v{font-size:24px;font-weight:700;color:var(--c1-ink);margin:2px 0}
.ff-ev .n{font-size:14px;color:var(--c1-muted);line-height:1.4}
.ff-h{font-size:18px;font-weight:700;color:var(--c1-ink);margin:18px 0 4px;padding-bottom:4px;border-bottom:1px solid var(--c1-rule)}
@media (max-width:800px){.ff-flow,.ff-evidence{grid-template-columns:1fr 1fr}
  .ff-row{grid-template-columns:1fr;gap:6px}.ff-row .num{text-align:left}.ff-row.head{display:none}}
</style>"""


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def style():
    """The group's shared look, then this page's own blocks."""
    loaded = False
    if group_ui is not None:
        try:
            group_ui.style(own_sidebar=False)
            loaded = True
        except Exception:
            loaded = False
    if not loaded:
        html("<style>.stApp{" + ";".join(f"--c1-{k}:{v}" for k, v in FALLBACK.items()) + "}</style>")
    html(CSS)


def title(text, lede):
    if group_ui is not None:
        group_ui.title(escape(text), escape(lede))
    else:
        st.title(text)
        st.write(lede)


def plain(group):
    """'bmi >=35 AND alt >=40' -> 'BMI ≥ 35 · ALT ≥ 40' (plain text, not yet escaped)."""
    parts = []
    for piece in group.split(" AND "):
        trait, value = piece.split(" ", 1)
        if trait == "sex":
            parts.append(value.capitalize())
            continue
        value = value.replace(">=", "≥ ").replace("<", "< ").replace("-", "–")
        parts.append(f"{TRAIT_NAMES.get(trait, trait)} {value}")
    return " · ".join(parts)


def by_trait(labels):
    """Sort labels like 'bmi >=35' in the order age, sex, BMI, AST, ALT, platelets."""
    return sorted(labels, key=lambda l: TRAIT_ORDER.index(l.split(" ")[0]))


def fmt_p(p):
    if p < 0.0001:
        return "&lt; 0.0001"
    if p < 0.001:
        return f"{p:.4f}"
    return f"{p:.3f}"


def tiles(items):
    """items: (label, value, note, bad). One row of headline numbers."""
    body = "".join(
        f'<div class="ff-tile{" bad" if bad else ""}"><div class="k">{escape(k)}</div>'
        f'<div class="v">{escape(v)}</div><div class="n">{escape(n)}</div></div>'
        for k, v, n, bad in items)
    html(f'<div class="ff-flow">{body}</div>')


def split(found, missed, tool):
    """Where the patients with fibrosis ended up: found by a scan, or sent home."""
    total = found + missed
    if total == 0:
        return
    a = 100 * found / total
    html(f'<div class="ff-split"><div style="width:{a:.2f}%;background:var(--c1-accent)">{found} found by a scan</div>'
         f'<div style="width:{100 - a:.2f}%;background:var(--c1-z-scan)">{missed} sent home</div></div>'
         f'<p class="ff-key">Of the <b>{total}</b> patients who truly have fibrosis, {escape(tool)} sends '
         f'<b>{missed}</b> home without a scan ({missed / total:.0%}). These are the patients the '
         f'Failure-Finder looks for.</p>')


def verdict(kind, head, body_html):
    """A box with a coloured edge. body_html must already be escaped by the caller."""
    z, t = KINDS[kind]
    html(f'<div class="ff-verdict" style="--z:{z};--t:{t}"><div class="h">{escape(head)}</div>'
         f'<div class="p">{body_html}</div></div>')


def group_rows(rows, bar, compact=False):
    """rows: dicts with group, cleared, wrongly_cleared, share, p_adjusted.
    Each group is a bar (its fibrosis rate among patients sent home) against the referral bar line.
    compact=True puts the bar under the label, for narrow columns."""
    pos = 100 * bar
    scale = (f'<div class="ff-scale"><span style="left:0;transform:none">0%</span>'
             f'<span style="left:50%">50%</span><span style="left:100%;transform:translateX(-100%)">100%</span>'
             f'<span class="bar" style="left:{pos:.1f}%">referral bar {bar:.1%}</span></div>')
    head = (f'<div class="ff-row head"><div class="g">Kind of patient</div><div class="t">{scale}</div>'
            f'<div class="num n">Sent home who had fibrosis</div>'
            f'<div class="num p">Luck chance<small>adjusted</small></div></div>')
    body = "".join(
        f'<div class="ff-row"><div class="g" title="{escape(r["group"])}">{escape(plain(r["group"]))}</div>'
        f'<div class="t"><div class="ff-track" style="--z:var(--c1-z-scan)">'
        f'<div class="ff-fill" style="width:{100 * r["share"]:.1f}%"></div>'
        f'<div class="ff-line" style="left:{pos:.1f}%"></div></div></div>'
        f'<div class="num n"><b>{r["wrongly_cleared"]} of {r["cleared"]}</b><small>{r["share"]:.0%}</small></div>'
        f'<div class="num p">{fmt_p(r["p_adjusted"])}</div></div>'
        for r in rows)
    html(f'<div class="ff-rows{" compact" if compact else ""}">{head}{body}</div>')


def chips(labels, highlight):
    """The patient's own groups as pills; the ones that match the red flag are highlighted."""
    html('<p class="ff-label">This patient\'s traits (the matching ones are marked)</p><div class="ff-chips">'
         + "".join(f'<span class="ff-chip{" on" if l in highlight else ""}">{escape(plain(l))}</span>'
                   for l in by_trait(labels))
         + '</div>')


def score(f4, cutoff, cleared, tool):
    where = "sent home" if cleared else "sent for a scan"
    html(f'<div class="ff-score"><b>{f4:.2f}</b><span>FIB-4 score. {escape(tool)} cut-off for this '
         f'patient is {cutoff:.2f}, so they are {where}.</span></div>')


def note(body_html):
    html(f'<p class="ff-note">{body_html}</p>')


def heading(text):
    html(f'<p class="ff-h">{escape(text)}</p>')


def empty(text):
    html(f'<div class="ff-empty">{escape(text)}</div>')


def steps(items):
    """items: (bold line, explanation)."""
    html('<ol class="ff-steps">' + "".join(f"<li>{escape(a)}<small>{escape(b)}</small></li>" for a, b in items)
         + "</ol>")


def evidence(items):
    """items: (what, result, explanation)."""
    html('<div class="ff-evidence">' + "".join(
        f'<div class="ff-ev"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>'
        f'<div class="n">{escape(n)}</div></div>' for k, v, n in items) + "</div>")
