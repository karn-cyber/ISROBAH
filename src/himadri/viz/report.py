"""Auto-generated HTML report (dark theme), embedding figures inline as base64
so it is a single portable file."""
from __future__ import annotations

import base64
from pathlib import Path

from jinja2 import Template

_TEMPLATE = Template("""<!doctype html><html><head><meta charset="utf-8">
<title>HIMADRI — Subsurface Ice Report</title>
<style>
 body{background:#0b0f1a;color:#e6ecff;font-family:-apple-system,Segoe UI,Roboto,sans-serif;
      max-width:1100px;margin:0 auto;padding:32px;line-height:1.5}
 h1{font-weight:700;letter-spacing:.5px} h2{color:#33d6a6;margin-top:36px}
 .sub{color:#9fb0d0} .grid{display:flex;gap:16px;flex-wrap:wrap}
 .card{background:#121829;border:1px solid #243049;border-radius:12px;padding:18px;flex:1;min-width:220px}
 .big{font-size:30px;font-weight:700;color:#00e5ff} .ok{color:#33d6a6} .warn{color:#ffd166}
 img{width:100%;border-radius:12px;border:1px solid #243049;margin-top:10px}
 table{border-collapse:collapse;width:100%;margin-top:10px}
 td,th{border:1px solid #243049;padding:8px;text-align:left} th{color:#9fb0d0}
 code{background:#1a2236;padding:2px 6px;border-radius:6px}
</style></head><body>
<h1>HIMADRI</h1>
<div class="sub">Hybrid Ice Mapping And Detection using Radar Intelligence ·
ISRO BAH 2026 · Problem Statement 8 · mode: <code>{{ mode }}</code></div>

<h2>1 · Headline results</h2>
<div class="grid">
 <div class="card"><div class="sub">Ice volume (top {{ depth_m }} m)</div>
   <div class="big">{{ "{:,.0f}".format(vol.total_m3) }} m³</div>
   <div class="sub">90% interval [{{ "{:,.0f}".format(vol.lower_m3) }}, {{ "{:,.0f}".format(vol.upper_m3) }}] m³</div></div>
 <div class="card"><div class="sub">Detection ROC-AUC</div>
   <div class="big">{{ "%.3f"|format(metrics.roc_auc) if metrics else "—" }}</div>
   <div class="sub">IoU {{ "%.2f"|format(metrics.iou) if metrics else "—" }}</div></div>
 <div class="card"><div class="sub">Rim false-positives vs CPR-only</div>
   <div class="big ok">{{ "%.0f%%"|format(metrics.rim_fp_reduction_fraction*100) if metrics and metrics.rim_fp_reduction_fraction==metrics.rim_fp_reduction_fraction else "—" }} fewer</div>
   <div class="sub">{{ metrics.himadri.rim_false_positives if metrics else "—" }} vs {{ metrics.baseline_cpr_only.rim_false_positives if metrics else "—" }}</div></div>
 <div class="card"><div class="sub">Traverse</div>
   <div class="big {{ 'ok' if energy.feasible else 'warn' }}">{{ "FEASIBLE" if energy.feasible else "CHECK" }}</div>
   <div class="sub">{{ energy.total_distance_m }} m · peak {{ energy.peak_energy_Wh }} Wh / {{ energy.battery_Wh }} Wh</div></div>
</div>

<h2>2 · Detection & uncertainty</h2>
<img src="data:image/png;base64,{{ img.detection }}">

<h2>3 · Resolving the CPR ambiguity (the core idea)</h2>
<p>A high circular-polarisation ratio (CPR) is ambiguous: rough rock mimics ice.
HIMADRI adds the degree of polarisation (DOP), the m-chi decomposition, the
L/S-band depth signature and optical roughness, fuses them, and reports a
calibrated probability with uncertainty. Ice and rock collide on CPR but
separate on DOP.</p>
<img src="data:image/png;base64,{{ img.disambiguation }}">

<h2>4 · Landing site & rover traverse</h2>
<img src="data:image/png;base64,{{ img.planning }}">
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total distance</td><td>{{ energy.total_distance_m }} m</td></tr>
<tr><td>Time in shadow</td><td>{{ energy.dark_time_min }} min (limit {{ energy.thermal_limit_min }})</td></tr>
<tr><td>Peak energy draw</td><td>{{ energy.peak_energy_Wh }} Wh (battery {{ energy.battery_Wh }} Wh)</td></tr>
<tr><td>Energy margin</td><td>{{ energy.energy_margin_Wh }} Wh</td></tr></table>

<h2>5 · Ice volume by depth</h2>
<img src="data:image/png;base64,{{ img.volume }}">
<div class="sub">Mean ice fraction {{ "%.3f"|format(vol.mean_ice_fraction) }} ·
effective permittivity ~ {{ "%.2f"|format(vol.meta.mean_eps_eff) }} ·
Monte-Carlo n={{ vol.meta.n_mc }}</div>

{% if metrics and metrics.volume %}
<table><tr><th>Volume validation</th><th></th></tr>
<tr><td>Estimate</td><td>{{ "{:,.0f}".format(metrics.volume.estimate_m3) }} m³</td></tr>
<tr><td>Truth (synthetic)</td><td>{{ "{:,.0f}".format(metrics.volume.truth_m3) }} m³</td></tr>
<tr><td>Relative error</td><td>{{ "%.1f%%"|format(metrics.volume.relative_error*100) }}</td></tr>
<tr><td>Interval contains truth</td><td>{{ metrics.volume.interval_contains_truth }}</td></tr></table>
{% endif %}

{% if importance %}
<h2>6 · What the model relies on (feature importance)</h2>
<table><tr><th>Feature</th><th>Importance</th></tr>
{% for k, v in importance %}<tr><td><code>{{ k }}</code></td><td>{{ "%.3f"|format(v) }}</td></tr>{% endfor %}
</table>{% endif %}

<h2>7 · Method summary</h2>
<p class="sub">Pipeline: DFSAR(L+S) + OHRC + DEM → calibrate / speckle-filter /
co-register → polarimetric features (CPR, DOP, χ, m-χ, L/S) + optical roughness
+ illumination → physics pseudo-labels → gradient-boosted classifier → Bayesian
radar×optical fusion → per-pixel probability + uncertainty → dielectric/abundance
inversion (Monte-Carlo) → volume → multi-criteria landing site → power/thermal
dash-and-return traverse. Backend: {{ backend }}.</p>
</body></html>""")


def render_report(context: dict, outdir) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    img = {}
    for name, path in context["figs"].items():
        img[name] = base64.b64encode(Path(path).read_bytes()).decode()
    html = _TEMPLATE.render(
        mode=context["mode"],
        depth_m=context["depth_m"],
        vol=context["vol"],
        metrics=context.get("metrics"),
        energy=context["energy"],
        img=img,
        importance=context.get("importance"),
        backend=context.get("backend", "—"),
    )
    out = outdir / "report.html"
    out.write_text(html)
    return out
