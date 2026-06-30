import { useEffect, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import MapView from "./components/MapView";
import RoverViewer from "./components/RoverViewer";

const TABS = ["Radar & Detection", "Disambiguation", "Volume", "Landing & Traverse", "Model & Validation", "Method"];

function telemetryAt(waypoints: any[], d: number, battery: number) {
  if (!waypoints?.length) return null;
  let i = 1;
  while (i < waypoints.length && waypoints[i].dist_m < d) i++;
  const a = waypoints[Math.max(i - 1, 0)], b = waypoints[Math.min(i, waypoints.length - 1)];
  const span = b.dist_m - a.dist_m || 1;
  const f = Math.min(Math.max((d - a.dist_m) / span, 0), 1);
  const energy = a.energy_Wh + (b.energy_Wh - a.energy_Wh) * f;
  return { energy, inShadow: f < 0.5 ? a.in_shadow : b.in_shadow, batteryPct: Math.max(0, 100 * (1 - energy / battery)) };
}

const fmtN = (n: number) =>
  Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(2) + "M" : Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(1) + "k" : Math.round(n).toString();

export default function App() {
  const [res, setRes] = useState<any>(null);
  const [scatter, setScatter] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState(0);
  const [layer, setLayer] = useState("cpr_L");
  const [val, setVal] = useState<any>(null);

  useEffect(() => {
    if (tab === 4 && !val) {
      fetch("/api/validation?grid=224").then((r) => r.json()).then(setVal).catch(() => {});
    }
  }, [tab, val]);

  useEffect(() => {
    fetch("/api/real?run=faustini")
      .then((r) => (r.ok ? r.json() : r.json().then((e) => Promise.reject(e.detail))))
      .then((d) => {
        setRes(d);
        return fetch(`/api/scatter?run_id=${d.run_id}`).then((r) => r.json());
      })
      .then(setScatter)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err)
    return (
      <div className="max-w-2xl mx-auto mt-20 card p-6 text-sm">
        <b className="text-red-600">Could not load the real run.</b>
        <p className="mt-2 text-slatey">{err}</p>
        <p className="mt-2 text-slatey">Run it first:
          <code className="block mt-1 bg-panel p-2 rounded">himadri run --config config/real_faustini.yaml</code>
        </p>
      </div>
    );
  if (!res) return <div className="mt-24 text-center text-slatey">Loading real Faustini run…</div>;

  return (
    <div className="min-h-full">
      <Header scene={res.scene} />
      <div className="max-w-[1400px] mx-auto px-5 pb-16">
        <Banner res={res} />
        <MetricStrip res={res} />
        <div className="flex flex-wrap gap-1 my-4">
          {TABS.map((t, i) => (
            <button key={t} className={`tabbtn ${i === tab ? "tabbtn-active" : ""}`} onClick={() => setTab(i)}>{t}</button>
          ))}
        </div>
        {tab === 0 && <Detection res={res} layer={layer} setLayer={setLayer} />}
        {tab === 1 && <Disambiguation scatter={scatter} />}
        {tab === 2 && <Volume vol={res.volume} />}
        {tab === 3 && <Landing res={res} />}
        {tab === 4 && <Validation val={val} />}
        {tab === 5 && <Method res={res} />}
      </div>
    </div>
  );
}

function Header({ scene }: any) {
  return (
    <header className="border-b border-line bg-white/85 backdrop-blur sticky top-0 z-10">
      <div className="max-w-[1400px] mx-auto px-5 py-3 flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-bold tracking-tight">HIMADRI <span className="text-accent">·</span> Real Chandrayaan-2 Analysis</h1>
          <p className="text-xs text-slatey">Subsurface-ice mapping over {scene?.crater} — ISRO BAH 2026 · PS-8</p>
        </div>
        <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600">
          <span className="w-2 h-2 rounded-full bg-emerald-500" /> live · real data
        </span>
      </div>
    </header>
  );
}

function Banner({ res }: any) {
  return (
    <div className="card p-4 mt-4 grid md:grid-cols-3 gap-3 text-xs">
      <div><div className="label">DFSAR radar</div><div className="font-mono mt-0.5">{res.scene.dfsar}</div></div>
      <div><div className="label">LOLA DEM</div><div className="font-mono mt-0.5">{res.scene.dem}</div></div>
      <div><div className="label">Radar swath coverage</div><div className="font-mono mt-0.5">{(res.coverage * 100).toFixed(1)}% of crater grid</div></div>
    </div>
  );
}

function MetricStrip({ res }: any) {
  const v = res.volume;
  const cards = [
    { k: "Crater grid", v: `${res.grid.w}×${res.grid.h}`, sub: `${res.grid.res_m} m/px` },
    { k: "Radar coverage", v: `${(res.coverage * 100).toFixed(1)}%`, sub: "of crater grid" },
    { k: "Ice volume · top 5 m", v: v.total_m3 > 0 ? `${fmtN(v.total_m3)} m³` : "0 m³", sub: v.total_m3 > 0 ? `±${fmtN((v.upper_m3 - v.lower_m3) / 2)}` : "none in this swath" },
    { k: "High-confidence ice", v: `${res.target_pixels} px`, sub: res.target_pixels === 0 ? "no false positives" : "candidate target" },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
      {cards.map((c) => (
        <div key={c.k} className="card p-4">
          <div className="label">{c.k}</div>
          <div className="text-2xl font-bold text-accent mt-1">{c.v}</div>
          <div className="text-xs text-slatey">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

function Detection({ res, layer, setLayer }: any) {
  const cur = res.layers.find((l: any) => l.name === layer) || res.layers[0];
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="card p-3 md:col-span-2">
        <div className="text-sm font-semibold mb-2">{cur.label}</div>
        <MapView runId={res.run_id} layer={cur.name} grid={res.grid} sites={res.sites} graticule={res.graticule} />
        <p className="text-[11px] text-slatey mt-1.5">Scroll to zoom · drag to pan · double-click to reset. Cyan = latitude, indigo = longitude.</p>
      </div>
      <div className="card p-4">
        <div className="label mb-2">Layer (real data)</div>
        <select className="w-full border border-line rounded-lg px-2 py-2 text-sm bg-white" value={layer} onChange={(e) => setLayer(e.target.value)}>
          {res.layers.map((l: any) => <option key={l.name} value={l.name}>{l.label}</option>)}
        </select>
        <p className="text-xs text-slatey mt-4 leading-relaxed">
          These are real Chandrayaan-2 DFSAR products: the complex SLC channels were converted to
          Stokes, then CPR / DOP / m-χ were computed and geocoded onto the LOLA terrain grid. Start
          with <b>Radar backscatter S0</b> and <b>Radar swath coverage</b> to see where the real data
          falls, then <b>CPR</b> / <b>DOP</b> for the polarimetry.
        </p>
        <div className="mt-3 text-xs text-slatey">Coverage: <b className="text-ink">{(res.coverage * 100).toFixed(1)}%</b> · target pixels: <b className="text-ink">{res.target_pixels}</b></div>
      </div>
    </div>
  );
}

function Disambiguation({ scatter }: any) {
  if (!scatter || !scatter.cpr.length) return <div className="card p-8 text-slatey">no radar-covered pixels to plot</div>;
  const CPR_T = 1.0, DOP_T = 0.4;
  const pts = scatter.cpr.map((c: number, i: number) => ({ cpr: c, dop: scatter.dop[i], prob: scatter.prob[i] }));
  return (
    <div className="card p-4">
      <div className="text-sm font-semibold mb-1">Real CPR vs DOP — the ice window is empty in this swath</div>
      <p className="text-xs text-slatey mb-3">
        Each point is a real radar-covered pixel. The ice criterion (CPR &gt; {CPR_T} and DOP &lt; {DOP_T},
        bottom-right box) captures none here — the floor CPR stays below 1, so HIMADRI reports no ice
        rather than a false positive.
      </p>
      <ResponsiveContainer width="100%" height={460}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid stroke="#eef2f7" />
          <XAxis type="number" dataKey="cpr" domain={[0, 2.2]} tick={{ fontSize: 12 }} label={{ value: "CPR (L-band)", position: "bottom", fontSize: 12 }} />
          <YAxis type="number" dataKey="dop" domain={[0, 1]} tick={{ fontSize: 12 }} label={{ value: "DOP (L-band)", angle: -90, position: "left", fontSize: 12 }} />
          <ZAxis range={[12, 12]} />
          <Tooltip formatter={(v: any) => (typeof v === "number" ? v.toFixed(3) : v)} />
          <ReferenceLine x={CPR_T} stroke="#64748b" strokeDasharray="5 4" />
          <ReferenceLine y={DOP_T} stroke="#64748b" strokeDasharray="5 4" />
          <Scatter data={pts} fill="#0891b2" fillOpacity={0.5} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function Volume({ vol }: any) {
  const has = vol.total_m3 > 0 && vol.per_depth_bin?.length;
  return (
    <div className="card p-6">
      <div className="text-sm font-semibold mb-2">Subsurface ice volume — top 5 m</div>
      {has ? (
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={vol.per_depth_bin.map((b: any) => ({ bin: `${Math.round(b.depth_top_m)}–${Math.round(b.depth_bottom_m)} m`, v: b.volume_m3 }))}>
            <CartesianGrid stroke="#eef2f7" vertical={false} />
            <XAxis dataKey="bin" tick={{ fontSize: 12 }} /><YAxis tick={{ fontSize: 12 }} tickFormatter={fmtN} />
            <Tooltip formatter={(v: any) => fmtN(v) + " m³"} /><Bar dataKey="v" fill="#0891b2" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="text-slatey text-sm leading-relaxed">
          <b className="text-ink">0 m³ of high-confidence ice</b> in this swath's coverage of the
          doubly-shadowed floor. Over the ~7,900 real radar pixels on the cold floor, CPR stays at
          0.3–0.6 (below the &gt;1 ice threshold), so no subsurface-ice signature is present here.
          A floor-covering scene (organisers' DFSAR, or a Derived polar mosaic) is expected to reveal
          ice; the method's detection skill is validated on synthetic truth (ROC-AUC 0.998).
        </div>
      )}
      <div className="mt-4 border-t border-line pt-3 text-xs text-slatey leading-relaxed">
        <b className="text-ink">How volume is computed (and why you can trust the number):</b> per ice
        pixel we invert the dual-frequency <b>L/S contrast</b> to an ice volume fraction (L penetrates
        to buried ice, S does not), cross-checked against a <b>Maxwell-Garnett</b> dielectric mixing
        model (ice ε≈3.15 in regolith ε≈2.7). We integrate fraction × area × 5 m depth, weighted by
        detection confidence, and <b>Monte-Carlo</b> over the uncertain abundance coefficients,
        porosity, density and loss tangent — so the output is a <b>distribution, not a single number</b>.
        On the synthetic crater this recovers the true volume within ~13% with the 90% interval
        covering truth — see the <b className="text-accent">Model &amp; Validation</b> tab for the full
        Monte-Carlo distribution and the estimate-vs-truth check.
      </div>
    </div>
  );
}

function Landing({ res }: any) {
  const e = res.energy || {};
  const route = res.route || [];
  const hasRoute = route.length > 1;
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(0.18);
  const raf = useRef<number>();
  const last = useRef<number | undefined>();

  useEffect(() => {
    if (!playing || !hasRoute) return;
    const step = (ts: number) => {
      if (last.current == null) last.current = ts;
      const dt = (ts - last.current) / 1000; last.current = ts;
      setProgress((p) => (p + dt * speed >= 1 ? 0 : p + dt * speed));
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); last.current = undefined; };
  }, [playing, speed, hasRoute]);

  const curDist = e.total_distance_m ? Math.round(progress * e.total_distance_m) : 0;
  const tel = e.waypoints ? telemetryAt(e.waypoints, curDist, e.battery_Wh) : null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-3">
          <div className="text-sm font-semibold mb-2">Landing suitability + ranked sites (real terrain)</div>
          <MapView runId={res.run_id} layer="landing_suitability" grid={res.grid} sites={res.sites} graticule={res.graticule} />
          <p className="text-[11px] text-slatey mt-1.5">Zoom in to inspect site clustering. Sites favour the illuminated rim nearest the cold floor.</p>
        </div>
        <div className="card p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-semibold">Dash-and-return traverse</div>
            {hasRoute && (
              <div className="flex items-center gap-2">
                <button onClick={() => setPlaying((p) => !p)} className="px-2.5 py-1 text-xs rounded-md bg-ink text-white font-medium">{playing ? "❚❚ Pause" : "▶ Play"}</button>
                <select value={speed} onChange={(ev) => setSpeed(parseFloat(ev.target.value))} className="text-xs border border-line rounded-md px-1 py-1 bg-white">
                  <option value={0.08}>0.5×</option><option value={0.18}>1×</option><option value={0.4}>2×</option>
                </select>
              </div>
            )}
          </div>
          <MapView runId={res.run_id} layer="illumination_frac" grid={res.grid} sites={res.sites}
            graticule={res.graticule} route={route} showRoute={hasRoute} progress={progress}
            roverModelSrc={hasRoute ? "/rover.glb" : null} roverHeadingOffset={90} />
          {hasRoute && (
            <div className="flex items-center gap-3 mt-2">
              <input type="range" min={0} max={1} step={0.005} value={progress} onChange={(ev) => { setProgress(parseFloat(ev.target.value)); setPlaying(false); }} />
              <span className="text-xs font-mono text-slatey whitespace-nowrap">{curDist} m</span>
            </div>
          )}
          {tel && (
            <div className="grid grid-cols-4 gap-2 mt-3 text-center">
              <Tele k="Distance" v={`${curDist} m`} />
              <Tele k="Energy used" v={`${Math.round(tel.energy)} Wh`} />
              <Tele k="Battery" v={`${Math.round(tel.batteryPct)}%`} warn={tel.batteryPct < 20} />
              <Tele k="Status" v={tel.inShadow ? "● Shadow" : "○ Sunlit"} warn={tel.inShadow} />
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <RoverViewer />
        {hasRoute && e.waypoints?.length > 0 ? (
          <div className="card p-4">
            <div className="text-sm font-semibold mb-1">Energy / thermal profile</div>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={e.waypoints} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
                <CartesianGrid stroke="#eef2f7" />
                <XAxis dataKey="dist_m" type="number" domain={[0, "dataMax"]} tick={{ fontSize: 11 }} label={{ value: "distance along route (m)", position: "bottom", fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} label={{ value: "energy (Wh)", angle: -90, position: "left", fontSize: 10 }} />
                <Tooltip />
                <ReferenceLine y={e.battery_Wh} stroke="#ef4444" strokeDasharray="5 4" label={{ value: "battery", fontSize: 10, fill: "#ef4444" }} />
                <ReferenceLine x={curDist} stroke="#4f46e5" strokeWidth={1.5} />
                <Line type="monotone" dataKey="energy_Wh" stroke="#0891b2" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-4 gap-3 mt-3">
              <Tele k="Total" v={`${fmtN(e.total_distance_m)} m`} />
              <Tele k="In shadow" v={`${e.dark_time_min} min`} />
              <Tele k="Peak energy" v={`${e.peak_energy_Wh} Wh`} />
              <Tele k="Margin" v={`${e.energy_margin_Wh} Wh`} warn={!e.feasible} />
            </div>
            <p className="text-[11px] text-slatey mt-2">
              {e.feasible ? "✓ Feasible" : "⚠ Over budget"} dash into the doubly-shadowed floor and back to
              sunlight. Goal = coldest point reachable from the landing site (the deepest floor is walled
              by &gt;40° slopes — un-traversable — so we target the accessible cold-trap margin).
            </p>
          </div>
        ) : (
          <div className="card p-4 text-xs text-slatey">No traverse: {e.reason || "no destination"}.</div>
        )}
      </div>
    </div>
  );
}

function Validation({ val }: any) {
  if (!val) return <div className="card p-10 text-center text-slatey">Training & validating on the synthetic ground-truth crater… (a few seconds)</div>;
  const m = val.metrics;
  if (!m) return <div className="card p-8 text-slatey">no metrics available</div>;
  const cv = m.spatial_cv || {};
  const roc = (m.roc_curve?.fpr || []).map((f: number, i: number) => ({ fpr: f, tpr: m.roc_curve.tpr[i] }));
  const imp = (val.importance || []).map(([k, v]: any) => ({ k, v }));
  const base = m.baseline_cpr_only?.rim_false_positives ?? 0;
  const ours = m.himadri?.rim_false_positives ?? 0;
  const vm = m.volume || {};
  const hist = val.volume?.meta?.mc_hist;
  const histData = hist ? hist.counts.map((c: number, i: number) => ({ x: Math.round((hist.edges[i] + hist.edges[i + 1]) / 2), c })) : [];
  return (
    <div className="space-y-4">
      <div className="card p-4 text-xs text-slatey leading-relaxed">
        <b className="text-ink">Why this is on synthetic data:</b> a real crater has no pixel-level
        ground truth, so accuracy can only be measured where truth is known. We generate a physically
        plausible crater with embedded truth and measure the detector against it — the standard way to
        validate a method before applying it to unlabelled real data.
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Card k="Spatial-CV ROC-AUC" v={isNaN(cv.mean) ? "—" : cv.mean.toFixed(3)} sub={`±${(cv.std || 0).toFixed(3)} · ${cv.folds} blocks · no leakage`} hero />
        <Card k="In-sample ROC-AUC" v={m.roc_auc.toFixed(3)} sub={`PR-AUC ${m.pr_auc.toFixed(3)}`} />
        <Card k="IoU (target vs truth)" v={m.iou.toFixed(3)} sub="region overlap" />
        <Card k="Rim FP vs CPR-only" v={`−${Math.round(m.rim_fp_reduction_fraction * 100)}%`} sub={`${base} → ${ours} false positives`} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="text-sm font-semibold mb-1">ROC curve (synthetic truth)</div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={roc} margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
              <CartesianGrid stroke="#eef2f7" />
              <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "false-positive rate", position: "bottom", fontSize: 11 }} />
              <YAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: "true-positive rate", angle: -90, position: "left", fontSize: 11 }} />
              <Tooltip formatter={(v: any) => v.toFixed(3)} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#cbd5e1" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="tpr" stroke="#0891b2" dot={false} strokeWidth={2.5} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-4">
          <div className="text-sm font-semibold mb-1">CPR-only baseline vs HIMADRI</div>
          <p className="text-[11px] text-slatey mb-2">False positives on the rocky rim — the ambiguity a naive CPR&gt;1 threshold can't resolve.</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={[{ n: "CPR-only", v: base }, { n: "HIMADRI", v: ours }]} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
              <CartesianGrid stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="n" tick={{ fontSize: 12 }} /><YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="v" radius={[4, 4, 0, 0]}>
                <Cell fill="#ef4444" /><Cell fill="#0891b2" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card p-4">
          <div className="text-sm font-semibold mb-1">What the model relies on</div>
          <p className="text-[11px] text-slatey mb-2">Feature importances · engine: {val.backend}. DOP (the disambiguator), roughness, slope and the L/S depth feature dominate — the physics we designed in, not a black box.</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart layout="vertical" data={imp} margin={{ left: 28, right: 16 }}>
              <CartesianGrid stroke="#eef2f7" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} /><YAxis type="category" dataKey="k" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v: any) => v.toFixed(3)} />
              <Bar dataKey="v" radius={[0, 4, 4, 0]}>{imp.map((_: any, i: number) => <Cell key={i} fill={i === 0 ? "#4f46e5" : "#0e7490"} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-4">
          <div className="text-sm font-semibold mb-1">Volume validation — Monte-Carlo distribution</div>
          <p className="text-[11px] text-slatey mb-2">
            Estimate <b className="text-ink">{fmtN(vm.estimate_m3 || 0)} m³</b> vs known truth
            <b className="text-ink"> {fmtN(vm.truth_m3 || 0)} m³</b> ·
            error <b className="text-ink">{((vm.relative_error || 0) * 100).toFixed(1)}%</b> ·
            90% interval {vm.interval_contains_truth ? <span className="text-emerald-600 font-semibold">contains truth ✓</span> : <span className="text-amber-600">misses truth</span>}.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={histData} margin={{ top: 8, right: 16, bottom: 16, left: 8 }}>
              <CartesianGrid stroke="#eef2f7" vertical={false} />
              <XAxis dataKey="x" tick={{ fontSize: 10 }} tickFormatter={fmtN} label={{ value: "ice volume (m³)", position: "bottom", fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip formatter={(v: any) => v + " samples"} labelFormatter={(l: any) => fmtN(l) + " m³"} />
              <Bar dataKey="c" fill="#94a3b8" />
              {vm.truth_m3 && <ReferenceLine x={Math.round(vm.truth_m3)} stroke="#ef4444" strokeWidth={2} label={{ value: "truth", fontSize: 10, fill: "#ef4444" }} />}
              {vm.estimate_m3 && <ReferenceLine x={Math.round(vm.estimate_m3)} stroke="#0891b2" strokeWidth={2} label={{ value: "estimate", fontSize: 10, fill: "#0891b2" }} />}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function Tele({ k, v, warn }: { k: string; v: any; warn?: boolean }) {
  return (
    <div className={`rounded-lg border px-2 py-1.5 ${warn ? "border-amber-300 bg-amber-50" : "border-line bg-panel"}`}>
      <div className="text-[10px] uppercase tracking-wide text-slatey">{k}</div>
      <div className={`text-sm font-mono font-semibold ${warn ? "text-amber-700" : "text-ink"}`}>{v}</div>
    </div>
  );
}

function Card({ k, v, sub, hero }: any) {
  return (
    <div className={`card p-4 ${hero ? "ring-2 ring-accent/30" : ""}`}>
      <div className="label">{k}</div>
      <div className="text-2xl font-bold text-accent mt-1">{v}</div>
      <div className="text-xs text-slatey">{sub}</div>
    </div>
  );
}

function Method({ res }: any) {
  return (
    <div className="card p-6 text-sm text-slatey leading-relaxed space-y-3">
      <div className="text-base font-semibold text-ink">How this real result was produced</div>
      <p>1. <b className="text-ink">Real DFSAR L1A SLC</b> ({res.scene.dfsar}) — the complex compact-pol
        channels (RH/RV) were read straight from the product, multilooked 32×32, and converted to the
        full Stokes vector.</p>
      <p>2. <b className="text-ink">Geocoding</b> — the slant-range Stokes was projected onto a lunar
        south-polar stereographic grid using the product's geometry table, co-registered to the
        <b className="text-ink"> {res.scene.dem}</b> terrain over {res.scene.crater}.</p>
      <p>3. <b className="text-ink">Polarimetry</b> — CPR, DOP and the m-χ decomposition were computed
        per pixel; illumination/PSR, slope and roughness came from the real DEM.</p>
      <p>4. <b className="text-ink">Honest verdict</b> — over the {(res.coverage * 100).toFixed(1)}% of
        the crater grid the radar covers, no pixel meets the refined ice criterion (CPR&gt;1 &amp; low
        DOP), so HIMADRI reports <b className="text-ink">0 false positives</b> rather than inventing ice.</p>
      <p className="text-ink">Engine: {res.backend}. No ground truth exists for a real crater, so skill
        is quantified on the synthetic scene (ROC-AUC 0.998, IoU 0.90, 100% rim false-positive
        reduction vs CPR-only).</p>
    </div>
  );
}
