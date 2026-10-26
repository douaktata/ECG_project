"""Holter-style summary report, built from the beats the real-time monitor (ecg_monitor.py) has classified.

It reproduces the first part of a commercial Holter report: a header, a condensed summary in sentences,
and the Summary table (heart rate, heart rate variability, ectopic beats). It can be generated at any
time while the signal is still arriving (the AI server's dashboard does that), or offline for a whole
MIT-BIH recording:

    python ecg_report.py --record 233                # writes report_233.html, open it in a browser

What it can NOT do: the ST-T part (ST depression, T-wave), atrial fibrillation detection, PQ/QT intervals.
The model was never trained for these; the report says so instead of inventing numbers.
"""
import argparse
import datetime as dt
import html
import os

import numpy as np

# ---- Rules of the report (all in one place, so they are easy to change) --------------------------
TACHY_BPM = 100          # sinus heart rate above this = tachycardia (too fast)
BRADY_BPM = 50           # sinus heart rate below this = bradycardia (too slow)
PAUSE_SECONDS = 2.0      # no beat for more than 2 s = pause
SIGNAL_LOST_SECONDS = 3.0  # more than 3 s: probably signal lost (electrode off): not counted as analysed
PREMATURE = 0.88         # rr_pre / rr_local below this = the beat came early (Level 0 rule, Notebook 3)
V_SHAPE_CORR = 0.90      # correlation with the patient's typical beat below this = different shape
IRREGULAR_RHYTHM = 0.15  # share of successive RR changes > 20 %: above this, the rhythm is irregular

# Healthy HRV ranges. 24-hour ranges are the ones printed on commercial Holter reports;
# a short recording (here 30 min) has smaller values, so short-term ranges are used for it.
HRV_RANGES_24H = {"sdnn": (102, 162), "sdann": (91, 145), "pnn50": (1, 19), "rmssd": (20, 42)}
HRV_RANGES_SHORT = {"sdnn": (32, 93), "pnn50": (1, 40), "rmssd": (19, 75)}

LEVELS = ["Lowest", "Low", "Medium", "High"]


# ---- 1. Beat types --------------------------------------------------------------------------------
def beat_type(beat):
    """N (normal), S (supraventricular: early but normal shape) or V (ventricular: different shape).

    The deployed Random Forest only says normal / abnormal. To split the abnormal beats like a Holter
    report, we use what defines each family (see PROJECT_GUIDE.md, section 2):
      - V: the model says abnormal AND the shape differs from the patient's typical beat;
      - S: the beat comes early (Level 0 rule), or the model says abnormal with a normal shape.
    """
    if beat["abnormal"] and beat["corr_template"] < V_SHAPE_CORR:
        return "V"
    if beat["abnormal"] or beat["prematurity"] < PREMATURE:
        return "S"
    return "N"


def runs(types, kind):
    """Groups of consecutive beats of one kind: list of (index of the first beat, length)."""
    found, start = [], None
    for i, t in enumerate(list(types) + ["end"]):
        if t == kind and start is None:
            start = i
        elif t != kind and start is not None:
            found.append((start, i - start))
            start = None
    return found


def max_per_minute(times):
    """Largest number of events in one minute of the recording (the 'Max events/m' column)."""
    if len(times) == 0:
        return 0
    return int(np.bincount((np.asarray(times) / 60).astype(int)).max())


# ---- 2. Heart rate variability ------------------------------------------------------------------
def hrv(nn_s, nn_times):
    """Standard HRV measures from NN intervals (= RR intervals between two normal beats), in seconds."""
    if len(nn_s) < 20:
        return None
    nn = np.asarray(nn_s) * 1000                      # in ms
    diffs = np.diff(nn)
    segments = (np.asarray(nn_times) // 300).astype(int)     # 5-minute segments
    means = [nn[segments == s].mean() for s in np.unique(segments) if (segments == s).sum() >= 50]
    # bins of 1/128 s; always at least one bin (a perfectly regular rhythm has min == max)
    histogram, _ = np.histogram(nn, bins=np.arange(nn.min(), nn.max() + 2 * 7.8125, 7.8125))
    return {
        "sdnn": float(nn.std()),                                   # overall variability
        "sdann": float(np.std(means)) if len(means) >= 2 else None,  # variability of 5-min averages
        "pnn50": float(100 * np.mean(np.abs(diffs) > 50)),         # % of successive changes > 50 ms
        "rmssd": float(np.sqrt(np.mean(diffs ** 2))),              # beat-to-beat variability
        "triangular_index": float(len(nn) / histogram.max()),
    }


# ---- 3. The whole report --------------------------------------------------------------------------
def compute_report(beats, duration_s, start_time=None, alerts=()):
    """All the numbers of the report, from the monitor's beats (list of dicts) and the signal duration.

    start_time: a datetime for the first sample (clock times are shown), or None (times since start).
    """
    beats = [b for b in beats if "prematurity" in b]
    report = {"duration_s": float(duration_s), "start_time": start_time, "n_beats": len(beats),
              "alerts": list(alerts)}
    if len(beats) < 10:
        report["ready"] = False
        return report
    report["ready"] = True

    t = np.array([b["time"] for b in beats])
    rr = np.array([b["rr_pre"] for b in beats])
    hr = np.array([b["heart_rate"] for b in beats])
    types = [beat_type(b) for b in beats]
    is_n = np.array([k == "N" for k in types])
    report["types"] = types

    # Analysed time: everything except the gaps where the signal was probably lost
    lost = rr[rr > SIGNAL_LOST_SECONDS].sum()
    report["analysed_s"] = max(0.0, t[-1] - t[0] + rr[0] - lost)

    # Heart rate, all beats ("BPM") and sinus beats only ("BPM (sinus)")
    ok = rr <= SIGNAL_LOST_SECONDS
    report["bpm"] = {
        "beats": len(beats),
        "min": float(hr.min()), "min_t": float(t[hr.argmin()]),
        "avg": float(60 / rr[ok].mean()),
        "max": float(hr.max()), "max_t": float(t[hr.argmax()]),
        "max_rr_ms": float(rr[ok].max() * 1000), "max_rr_t": float(t[ok][rr[ok].argmax()]),
    }
    # a normal beat after two normal beats: the intervals around an ectopic beat (and its pause) are left out
    sinus = is_n & np.r_[False, is_n[:-1]] & np.r_[False, False, is_n[:-2]] & ok
    if sinus.sum() >= 10:
        s_hr, s_t, s_rr = hr[sinus], t[sinus], rr[sinus]
        report["bpm_sinus"] = {
            "beats": int(is_n.sum()),
            "min": float(s_hr.min()), "min_t": float(s_t[s_hr.argmin()]),
            "avg": float(60 / s_rr.mean()),
            "max": float(s_hr.max()), "max_t": float(s_t[s_hr.argmax()]),
            "max_rr_ms": float(s_rr.max() * 1000), "max_rr_t": float(s_t[s_rr.argmax()]),
            "tachy_s": float(s_rr[s_hr > TACHY_BPM].sum()),
            "brady_s": float(s_rr[s_hr < BRADY_BPM].sum()),
        }
        # Usual cleaning before HRV: drop intervals that jump > 20 % (often a missed or extra beat)
        jump = np.abs(np.diff(s_rr, prepend=s_rr[0])) / s_rr > 0.2
        report["hrv"] = hrv(s_rr[~jump], s_t[~jump])
    else:
        report["bpm_sinus"], report["hrv"] = None, None

    pauses = (rr > PAUSE_SECONDS) & (rr <= SIGNAL_LOST_SECONDS)
    report["pauses"] = {"count": int(pauses.sum()),
                        "longest_ms": float(rr[pauses].max() * 1000) if pauses.any() else None}
    report["signal_lost"] = int((rr > SIGNAL_LOST_SECONDS).sum())

    # Rhythm: how often the RR interval changes by more than 20 % from one normal beat to the next
    nn = rr[sinus]
    irregularity = float(np.mean(np.abs(np.diff(nn)) / nn[1:] > 0.2)) if len(nn) > 20 else 0.0
    report["irregularity"] = irregularity
    avg = report["bpm"]["avg"]
    if irregularity > IRREGULAR_RHYTHM:
        report["rhythm"] = "Irregular rhythm (possible atrial fibrillation: needs review)"
    elif avg > TACHY_BPM:
        report["rhythm"] = "Regular, fast rhythm"
    elif avg < BRADY_BPM:
        report["rhythm"] = "Regular, slow rhythm"
    else:
        report["rhythm"] = "Regular rhythm, normal rate (consistent with sinus rhythm)"

    # Ectopic beats: isolated beats, couplets (2 in a row), triplets (3), runs (4 or more)
    for kind, key in (("S", "supraventricular"), ("V", "ventricular")):
        groups = runs(types, kind)
        by_length = lambda keep: [s for s, n in groups if keep(n)]
        single, couplets = by_length(lambda n: n == 1), by_length(lambda n: n == 2)
        triplets, long_runs = by_length(lambda n: n == 3), by_length(lambda n: n >= 4)
        total = sum(n for _, n in groups)
        report[key] = {
            "total": total, "burden": 100 * total / len(beats),
            "single": len(single), "single_max_m": max_per_minute(t[single]),
            "couplets": len(couplets), "couplets_max_m": max_per_minute(t[couplets]),
            "triplets": len(triplets), "triplets_max_m": max_per_minute(t[triplets]),
            "runs": len(long_runs), "runs_max_m": max_per_minute(t[long_runs]),
            "longest_run": max((n for _, n in groups), default=0),
        }
    # In an irregular rhythm (e.g. atrial fibrillation) "early" beats mean nothing: S counts would be false
    report["supraventricular"]["not_counted"] = irregularity > IRREGULAR_RHYTHM
    report["levels"] = levels(report)
    return report


def levels(r):
    """A simple risk level per section (rules of this prototype, NOT a clinical score)."""
    out = {}
    b, s = r["bpm"], r["bpm_sinus"]
    if b["min"] < 40 or b["max"] > 150 or r["pauses"]["count"] > 0:
        out["automatism"] = 3
    elif not 50 <= b["avg"] <= 100:
        out["automatism"] = 2
    elif s and (s["tachy_s"] + s["brady_s"]) > 0.1 * r["analysed_s"]:
        out["automatism"] = 1
    else:
        out["automatism"] = 0

    h = r["hrv"]
    if h:
        ranges = HRV_RANGES_24H if r["duration_s"] >= 18 * 3600 else HRV_RANGES_SHORT
        outside = sum(1 for k, (lo, hi) in ranges.items() if h.get(k) is not None and not lo <= h[k] <= hi)
        out["hrv"] = min(outside, 2)
    else:
        out["hrv"] = None

    for key, run_level in (("supraventricular", 2), ("ventricular", 3)):
        e = r[key]
        if e.get("not_counted"):
            out[key] = 2          # irregular rhythm: needs review
            continue
        level = 0 if e["burden"] < 1 else 1 if e["burden"] < 5 else 2 if e["burden"] < 10 else 3
        if e["couplets"] or e["triplets"]:
            level = max(level, 1)
        if e["runs"]:
            level = max(level, run_level)
        out[key] = level
    out["overall"] = max(v for v in out.values() if v is not None)
    return out


# ---- 4. HTML page -------------------------------------------------------------------------------
def fmt_time(seconds, start_time):
    """Time of an event: clock time if the start is known, else time since the start."""
    if start_time is not None:
        return (start_time + dt.timedelta(seconds=seconds)).strftime("%H:%M:%S")
    return "+" + str(dt.timedelta(seconds=int(seconds)))


def fmt_duration(seconds):
    seconds = int(round(seconds))
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s" if m else f"{s}s"


LEVEL_ICONS = {   # colour and chevrons, like the commercial report
    0: ("#3a9a4a", "M3 5l5 4 5-4M3 10l5 4 5-4"),
    1: ("#3a9a4a", "M3 7l5 4 5-4"),
    2: ("#e8a33d", "M3 11l5-4 5 4"),
    3: ("#e0524f", "M3 11l5-4 5 4"),
}


def level_html(level):
    if level is None:
        return '<span class="level muted">—</span>'
    colour, path = LEVEL_ICONS[level]
    return (f'<span class="level"><svg width="16" height="16" viewBox="0 0 16 16"><path d="{path}" fill="none" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            f'{LEVELS[level]}</span>')


def at(value, seconds, start_time, digits=0):
    return f'<b>{value:.{digits}f}</b> at<br><u>{fmt_time(seconds, start_time)}</u>'


def render_html(report, patient=None, live=False, truth=None):
    """The report as a standalone HTML page (print it to PDF from the browser)."""
    p = {k: html.escape(str(v)) for k, v in (patient or {}).items() if v not in (None, "")}
    e = html.escape
    start = report.get("start_time")
    r = report

    def info(label, value):
        return f'<div class="info"><div class="label">{label}</div><div>{value or "—"}</div></div>'

    end_txt = fmt_time(r["duration_s"], start) if start else ""
    start_txt = start.strftime("%d %b %Y %H:%M:%S") if start else "start of the recording"
    overall = r["levels"]["overall"] if r.get("ready") else None
    header = f"""
    <section class="header">
      <div class="patient">
        <div class="name">{level_html(overall).replace('class="level"', 'class="level big"')}
          <span>{p.get("name", "Unnamed patient")}</span></div>
        {info("Patient ID", p.get("patient_id"))}
      </div>
      <div class="grid">
        {info("Date of birth", p.get("dob"))}{info("Gender", p.get("gender"))}
        {info("Device ID", p.get("device_id"))}{info("Recording time", fmt_duration(r["duration_s"]))}
        {info("Start", start_txt)}{info("End", end_txt)}
        {info("Lead configuration", p.get("lead", "1 lead"))}
        {info("Analysed (signal lost skipped)", fmt_duration(r.get("analysed_s", 0)))}
        {info("Ordering physician", p.get("physician"))}{info("Generated", dt.datetime.now().strftime("%d %b %Y %H:%M:%S"))}
      </div>
    </section>"""

    if not r.get("ready"):
        body = (f'<p class="waiting">Not enough beats yet ({r["n_beats"]} classified). '
                'The report needs at least 10 beats: wait a few seconds.</p>')
        return page(header + body, live)

    b, s, h = r["bpm"], r["bpm_sinus"], r["hrv"]
    sv, v = r["supraventricular"], r["ventricular"]
    lv = r["levels"]

    # Condensed summary, in sentences
    items = [f"The minimum heart rate was {b['min']:.0f}, the maximum heart rate was {b['max']:.0f}, "
             f"and the average heart rate was {b['avg']:.0f}."]
    for name, ectopy in (("SVPCs", sv), ("PVCs", v)):
        if ectopy.get("not_counted"):
            items.append(f"{name} not counted: the rhythm is irregular (possible atrial fibrillation, needs review).")
            continue
        burden = "&lt;0.1" if 0 < ectopy["burden"] < 0.1 else f"{ectopy['burden']:.1f}"
        verb, name = ("was", name[:-1]) if ectopy["total"] == 1 else ("were", name)
        items.append(f"There {verb} {ectopy['total']} {name} (<i>{burden}%</i> burden).")
    if r["pauses"]["count"]:
        items.append(f"{r['pauses']['count']} pause(s) &gt; {PAUSE_SECONDS:.0f} s, longest "
                     f"{r['pauses']['longest_ms']:.0f} ms.")
    if r["signal_lost"]:
        items.append(f"The signal was lost (no beat &gt; {SIGNAL_LOST_SECONDS:.0f} s) {r['signal_lost']} time(s): "
                     "check the electrodes.")
    summary = f"""
    <section>
      <h2>Condensed summary</h2>
      <p>The monitoring was worn from <b>{e(start_txt)}</b> for <b>{fmt_duration(r["duration_s"])}</b>.</p>
      <p>The predominant rhythm shows <i>{e(r["rhythm"])}</i>.</p>
      <p>The findings of the monitor are detailed below:</p>
      <ul>{"".join(f"<li>{x}</li>" for x in items)}</ul>
    </section>"""

    # Summary table
    ranges = HRV_RANGES_24H if r["duration_s"] >= 18 * 3600 else HRV_RANGES_SHORT

    def hrv_cell(key, unit):
        if not h or h.get(key) is None:
            return "—"
        value = h[key]
        lo, hi = ranges.get(key, (-np.inf, np.inf))
        cls = "" if lo <= value <= hi else ' class="red"'
        return f"<span{cls}>{value:.0f} {unit}</span>".replace(" </span>", "</span>")

    # Each row: (kind, cells). kind "sub" = small grey column titles, "data" = values.
    # Every row has 7 columns after the category: a label, then 6 values (a cell can be (html, colspan)).
    def ectopy_rows(ectopy, prefix, single_name):
        if ectopy.get("not_counted"):
            return [("data", [("Not counted: the rhythm is irregular (possible atrial fibrillation). "
                               "Early beats cannot be told apart in an irregular rhythm.", 7)])]
        rows = [("sub", ["", ("", 4), "Beats", "Max events/m"]),
                ("data", [single_name, ("", 4), ectopy["single"], ectopy["single_max_m"]])]
        episodes = [(f"{prefix} Couplet", "couplets"), (f"{prefix} Triplet", "triplets"),
                    (f"{prefix} Run (4+ beats)", "runs")]
        shown = [(n, k) for n, k in episodes if ectopy[k]]     # like the commercial report: only those found
        if shown:
            rows.append(("sub", ["", ("", 4), "Episodes", "Max events/m"]))
            rows += [("data", [n, ("", 4), ectopy[k], ectopy[k + "_max_m"]]) for n, k in shown]
        return rows

    def cells_html(cells):
        out = []
        for i, cell in enumerate(cells):
            content, span = cell if isinstance(cell, tuple) else (cell, 1)
            attrs = (' class="lbl"' if i == 0 else "") + (f' colspan="{span}"' if span > 1 else "")
            out.append(f"<td{attrs}>{content}</td>")
        return "".join(out)

    def section(group, categories):
        n_rows = sum(len(rows) for _, rows in categories)
        out = []
        for i, (category, rows) in enumerate(categories):
            band = False
            for j, (kind, cells) in enumerate(rows):
                if kind == "data":
                    band = not band                   # data rows alternate grey / white
                classes = [kind] + (["band"] if kind == "data" and band else []) + (["cat-start"] if j == 0 else [])
                lead = ""
                if i == 0 and j == 0:
                    lead += f'<td class="group" rowspan="{n_rows}"><span>{group}</span></td>'
                if j == 0:
                    lead += f'<td class="cat" rowspan="{len(rows)}">{category}</td>'
                out.append(f'<tr class="{" ".join(classes)}">{lead}{cells_html(cells)}</tr>')
        return f'<tbody class="group-body">{"".join(out)}</tbody>'

    if s is None:
        sinus_row = ("data", ["BPM (sinus)", ("not enough normal beats", 6)])
    else:
        sinus_row = ("data", ["BPM (sinus)", f'{s["beats"]:,}', at(s["min"], s["min_t"], start), f'{s["avg"]:.0f}',
                              at(s["max"], s["max_t"], start),
                              fmt_duration(s["tachy_s"]) if s["tachy_s"] else "—",
                              fmt_duration(s["brady_s"]) if s["brady_s"] else "—"])
    automatism = [
        ("sub", ["", "Beats", "Min BPM", "Avg BPM", "Max BPM", ("Max R-R, ms", 2)]),
        ("data", ["BPM", f'{b["beats"]:,}', at(b["min"], b["min_t"], start), f'{b["avg"]:.0f}',
                  at(b["max"], b["max_t"], start), (at(b["max_rr_ms"], b["max_rr_t"], start), 2)]),
        ("sub", ["", "Beats", "Min BPM", "Avg BPM", "Max BPM", "Tachy", "Brady"]),
        sinus_row,
    ]
    if h:
        hrv_rows = [("sub", ["", "", "SDNN", "SDANN", "pNN50", "RMSSD", "Triang Ind"]),
                    ("data", ["HRV (Sinus)", "", hrv_cell("sdnn", "ms"), hrv_cell("sdann", "ms"),
                              hrv_cell("pnn50", "%"), hrv_cell("rmssd", "ms"), f'{h["triangular_index"]:.0f}'])]
    else:
        hrv_rows = [("data", ["HRV (Sinus)", ("not enough normal beats", 6)])]
    st_rows = [("data", [("Not analysed: this model was not trained for ST deviations or T-wave changes "
                          "(it needs a calibrated signal in mV, several leads, and its own training data).", 7)])]
    table = f"""
    <section>
      <h2>Summary</h2>
      <table class="summary">
        <colgroup><col style="width:28px"><col style="width:118px"><col style="width:150px">
          <col><col><col><col><col><col></colgroup>
        {section("General", [(f"Automatism{level_html(lv['automatism'])}", automatism),
                             (f"HRV{level_html(lv['hrv'])}", hrv_rows)])}
        {section("Ectopy", [(f"Supraventricular{level_html(lv['supraventricular'])}",
                             ectopy_rows(sv, "Atrial", "Atrial Premature Contraction")),
                            (f"Ventricular{level_html(lv['ventricular'])}",
                             ectopy_rows(v, "Ventricular", "Ventricular Premature Contraction"))])}
        {section("ST-T", [('ST deviations<span class="level muted">n/a</span>', st_rows)])}
      </table>
      <p class="note">HRV ranges used: {"24-hour" if ranges is HRV_RANGES_24H else "short-term (recording &lt; 18 h)"}
        healthy ranges; values outside are in red. Levels (Lowest &rarr; High) are simple rules of this prototype,
        not a clinical score. Times are {"clock times" if start else "since the start of the recording"}.</p>
    </section>"""

    check = ""
    if truth:
        rows = "".join(f"<tr><td>{e(k)}</td><td>{m}</td><td>{c}</td></tr>" for k, m, c in truth)
        check = f"""
    <section>
      <h2>Check against the cardiologists (MIT-BIH annotations)</h2>
      <table class="check"><tr><th></th><th>This model</th><th>Cardiologists</th></tr>{rows}</table>
    </section>"""

    footer = ('<p class="disclaimer">Generated by the ECG anomaly detection prototype (Random Forest, MIT-BIH). '
              'Decision-support tool: the numbers must be checked by a qualified person. Not a diagnosis.</p>')
    return page(header + summary + table + check + footer, live)


STYLE = """
:root { --ink:#222; --muted:#888; --line:#bbb; --band:#f2f2f2; --red:#c62828; }
* { box-sizing: border-box; }
body { margin:0; background:#e9e9e9; color:var(--ink);
       font-family:"Fira Sans","Segoe UI","Helvetica Neue",Arial,sans-serif; font-size:13px; }
.page { background:#fff; width:210mm; min-height:297mm; margin:16px auto; padding:12mm 12mm; }
.toolbar { text-align:right; max-width:210mm; margin:12px auto 0; }
.toolbar button { font:inherit; padding:6px 14px; cursor:pointer; }
.live { color:#3a9a4a; font-weight:600; margin-right:12px; }
h2 { font-size:18px; margin:22px 0 8px; padding-bottom:5px; border-bottom:2px solid #555; }
section p { margin:6px 0; } ul { margin:4px 0; padding-left:20px; }
.header { display:grid; grid-template-columns: 1fr 1.6fr; border:1px solid var(--line); }
.header .patient { padding:10px 12px; border-right:2px solid #555; }
.header .name { display:flex; align-items:center; gap:10px; font-size:22px; font-weight:600; margin-bottom:10px; }
.grid { display:grid; grid-template-columns:repeat(2,1fr); }
.info { padding:6px 12px; border-bottom:1px solid #eee; }
.info .label { color:var(--muted); font-size:11.5px; }
.level { display:inline-flex; align-items:center; gap:2px; font-weight:400; font-size:12.5px; margin-top:2px; }
.level.big { flex-direction:column; font-size:11px; color:var(--muted); }
.level.big svg { width:26px; height:26px; }
.muted { color:var(--muted); }
table.summary { width:100%; border-collapse:collapse; table-layout:fixed; }
table.summary td { padding:4px 5px; text-align:center; line-height:1.3; }
table.summary td.lbl { text-align:left; white-space:normal; }
table.summary tr.sub td:not(.group):not(.cat) { color:var(--muted); font-size:11.5px; padding-top:7px; }
table.summary tr.band td:not(.group):not(.cat) { background:var(--band); }
table.summary td.group { padding:0; border-right:1px solid var(--line); vertical-align:middle; }
table.summary td.group span { display:block; transform:rotate(-90deg); font-weight:700; font-size:15px; white-space:nowrap; }
table.summary td.cat { text-align:left; vertical-align:middle; font-weight:600; white-space:normal;
                       border-right:1px solid var(--line); }
table.summary td.cat .level { display:flex; }
table.summary tbody.group-body { border-top:2px solid #555; }
table.summary tbody.group-body:last-child { border-bottom:2px solid #555; }
table.summary tr.cat-start td:not(.group) { border-top:1px solid var(--line); }
table.summary td[colspan="7"] { color:var(--muted); white-space:normal; text-align:left; }
.red { color:var(--red); font-weight:700; }
.note, .disclaimer { color:var(--muted); font-size:11px; }
.disclaimer { margin-top:24px; border-top:1px solid #ddd; padding-top:6px; }
.waiting { font-size:15px; padding:30px 0; color:var(--muted); }
table.check { border-collapse:collapse; } table.check td, table.check th { border:1px solid #ddd; padding:4px 12px; }
table.check td:not(:first-child) { text-align:center; }
@media print { body { background:#fff; } .toolbar { display:none; } .page { margin:0; width:auto; min-height:0; padding:0; }
               @page { size:A4; margin:12mm; } }
"""


def page(content, live):
    toolbar = ('<div class="toolbar">' + ('<span class="live">● live (updates automatically)</span>' if live else "")
               + '<button onclick="window.print()">Print / Save as PDF</button></div>')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ECG summary report</title>'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{STYLE}</style></head>'
            f'<body>{toolbar}<div class="page">{content}</div></body></html>')


# ---- 5. Offline use: a whole MIT-BIH recording ------------------------------------------------------
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V", "F": "F", "/": "Q", "f": "Q", "Q": "Q"}


def truth_counts(report, annotations):
    """Our counts next to the cardiologists' (S and V beats, couplets, triplets, runs)."""
    families = [AAMI[s] for s in annotations.symbol if s in AAMI]
    rows = [("Beats", report["n_beats"], len(families))]
    for kind, key, name in (("S", "supraventricular", "S"), ("V", "ventricular", "V")):
        groups = runs(families, kind)
        rows.append((f"{name} beats (total)", report[key]["total"], sum(n for _, n in groups)))
        rows.append((f"{name} isolated", report[key]["single"], sum(1 for _, n in groups if n == 1)))
        rows.append((f"{name} couplets", report[key]["couplets"], sum(1 for _, n in groups if n == 2)))
        rows.append((f"{name} triplets", report[key]["triplets"], sum(1 for _, n in groups if n == 3)))
        rows.append((f"{name} runs (4+)", report[key]["runs"], sum(1 for _, n in groups if n >= 4)))
    return rows


def main():
    import wfdb
    from ecg_monitor import ECGMonitor, load_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", default="233", help="MIT-BIH record name")
    parser.add_argument("--model", default="rf_model.joblib")
    parser.add_argument("--data-dir", default="", help="folder with the MIT-BIH files (empty = PhysioNet)")
    parser.add_argument("--out", default="", help="output file (default report_<record>.html)")
    args = parser.parse_args()

    path = os.path.join(args.data_dir, args.record) if args.data_dir else args.record
    kwargs = {} if args.data_dir else {"pn_dir": "mitdb"}
    record = wfdb.rdrecord(path, **kwargs)
    annotations = wfdb.rdann(path, "atr", **kwargs)
    lead = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    signal, fs = record.p_signal[:, lead], record.fs

    monitor = ECGMonitor(load_model(args.model), fs=fs, powerline_hz=60)
    print(f"Running the monitor on record {args.record} ({len(signal) / fs / 60:.0f} min)...")
    for i in range(0, len(signal), int(fs)):
        monitor.push(signal[i:i + int(fs)])

    report = compute_report(monitor.beats, monitor.now(), alerts=monitor.alerts)
    patient = {"name": f"MIT-BIH record {args.record}", "patient_id": args.record,
               "device_id": "MIT-BIH (replayed)", "lead": record.sig_name[lead]}
    out = args.out or f"report_{args.record}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_html(report, patient, truth=truth_counts(report, annotations)))
    print("Saved:", out, "(open it in a browser; 'Print / Save as PDF' makes a PDF)")


if __name__ == "__main__":
    main()
