import "@google/model-viewer";
import { useEffect, useRef, useState } from "react";
import { layerUrl } from "../api";

interface Line { label: string; pts: number[][]; }
interface Lbl { text: string; row: number; col: number; kind: string; }
interface Props {
  runId: string;
  layer: string;
  grid: { h: number; w: number };
  route?: number[][];
  sites?: { row: number; col: number; score: number }[];
  showRoute?: boolean;
  progress?: number;
  roverModelSrc?: string | null;
  roverHeadingOffset?: number;
  graticule?: { lat: Line[]; lon: Line[]; labels: Lbl[] };
  labels?: Lbl[];
}

function roverAt(route: number[][], t: number) {
  if (!route || route.length < 2) return null;
  const seg: number[] = [0];
  let total = 0;
  for (let i = 1; i < route.length; i++) {
    total += Math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]);
    seg.push(total);
  }
  const target = Math.min(Math.max(t, 0), 1) * total;
  let i = 1;
  while (i < seg.length && seg[i] < target) i++;
  const i0 = Math.max(i - 1, 0);
  const f = (target - seg[i0]) / ((seg[i] - seg[i0]) || 1);
  return {
    r: route[i0][0] + (route[i][0] - route[i0][0]) * f,
    c: route[i0][1] + (route[i][1] - route[i0][1]) * f,
    heading: Math.atan2(route[i][0] - route[i0][0], route[i][1] - route[i0][1]) * 180 / Math.PI,
    idx: i0,
  };
}

export default function MapView(p: Props) {
  const { runId, layer, grid, route, sites, showRoute, progress, roverModelSrc,
          roverHeadingOffset = 0, graticule, labels } = p;
  const box = useRef<HTMLDivElement>(null);
  const [v, setV] = useState({ k: 1, x: 0, y: 0 });
  const drag = useRef<{ x: number; y: number } | null>(null);

  const clampK = (k: number) => Math.min(Math.max(k, 1), 16);

  // non-passive wheel zoom toward cursor
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      setV((s) => {
        const nk = clampK(s.k * (e.deltaY < 0 ? 1.15 : 1 / 1.15));
        const ratio = nk / s.k;
        return { k: nk, x: mx - (mx - s.x) * ratio, y: my - (my - s.y) * ratio };
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const zoomBtn = (f: number) => setV((s) => {
    const el = box.current!; const w = el.clientWidth / 2, h = el.clientHeight / 2;
    const nk = clampK(s.k * f); const ratio = nk / s.k;
    return { k: nk, x: w - (w - s.x) * ratio, y: h - (h - s.y) * ratio };
  });

  const rover = showRoute && progress != null && route ? roverAt(route, progress) : null;
  const gw = grid.w;
  const k = v.k;
  const lineW = (gw / 700) / k, routeW = (gw / 220) / k, siteR = (gw / 75) / k;
  const font = (gw / 55) / k, labFont = (gw / 42) / k;

  return (
    <div
      ref={box}
      className="relative w-full overflow-hidden rounded-lg select-none"
      style={{ aspectRatio: `${grid.w} / ${grid.h}`, cursor: drag.current ? "grabbing" : "grab", background: "#fff", border: "1px solid #e2e8f0" }}
      onMouseDown={(e) => { drag.current = { x: e.clientX - v.x, y: e.clientY - v.y }; }}
      onMouseMove={(e) => { if (e.buttons === 1 && drag.current) setV((s) => ({ ...s, x: e.clientX - drag.current!.x, y: e.clientY - drag.current!.y })); }}
      onMouseUp={() => (drag.current = null)}
      onMouseLeave={() => (drag.current = null)}
      onDoubleClick={() => setV({ k: 1, x: 0, y: 0 })}
    >
      <div className="absolute inset-0" style={{ transform: `translate(${v.x}px, ${v.y}px) scale(${v.k})`, transformOrigin: "0 0" }}>
        <img src={layerUrl(runId, layer)} alt={layer} className="absolute inset-0 w-full h-full" draggable={false} />
        <svg className="absolute inset-0 w-full h-full" viewBox={`0 0 ${grid.w} ${grid.h}`} preserveAspectRatio="none">
          {/* lat/lon graticule */}
          {graticule && [...graticule.lat.map((l) => ({ ...l, lat: true })), ...graticule.lon.map((l) => ({ ...l, lat: false }))].map((l, i) => (
            <g key={i}>
              <polyline points={l.pts.map(([r, c]) => `${c},${r}`).join(" ")} fill="none"
                stroke={l.lat ? "#0ea5e9" : "#6366f1"} strokeWidth={lineW}
                strokeDasharray={`${lineW * 5} ${lineW * 4}`} opacity={0.55} />
              <text x={l.pts[0][1]} y={l.pts[0][0]} fontSize={font} fill={l.lat ? "#0369a1" : "#4338ca"}
                style={{ paintOrder: "stroke", stroke: "#fff", strokeWidth: font * 0.18 }}>{l.label}</text>
            </g>
          ))}
          {showRoute && route && route.length > 1 && (
            <>
              <polyline points={route.map(([r, c]) => `${c},${r}`).join(" ")} fill="none" stroke="#c7d2fe" strokeWidth={routeW} strokeLinejoin="round" />
              {rover && <polyline points={route.slice(0, rover.idx + 1).concat([[rover.r, rover.c]]).map(([r, c]) => `${c},${r}`).join(" ")} fill="none" stroke="#4f46e5" strokeWidth={routeW * 1.4} strokeLinejoin="round" />}
              <circle cx={route[0][1]} cy={route[0][0]} r={siteR} fill="#0f172a" stroke="#fff" strokeWidth={lineW} />
            </>
          )}
          {sites && sites.map((s, i) => (
            <g key={`s${i}`}>
              <circle cx={s.col} cy={s.row} r={siteR} fill="#0e7490" stroke="#fff" strokeWidth={lineW} />
              <text x={s.col + siteR * 1.3} y={s.row} fontSize={font} fill="#0e7490" fontWeight={700}
                style={{ paintOrder: "stroke", stroke: "#fff", strokeWidth: font * 0.18 }}>{i + 1}</text>
            </g>
          ))}
          {/* area labels */}
          {(labels || graticule?.labels || []).map((l, i) => (
            <g key={`l${i}`}>
              <circle cx={l.col} cy={l.row} r={siteR * 0.8} fill={l.kind === "crater" ? "#0f172a" : "#7c3aed"} stroke="#fff" strokeWidth={lineW} />
              <text x={l.col + siteR} y={l.row - siteR} fontSize={labFont} fontWeight={700}
                fill={l.kind === "crater" ? "#0f172a" : "#6d28d9"}
                style={{ paintOrder: "stroke", stroke: "#fff", strokeWidth: labFont * 0.22 }}>{l.text}</text>
            </g>
          ))}
        </svg>
        {rover && roverModelSrc && (
          <div style={{ position: "absolute", left: `${(rover.c / grid.w) * 100}%`, top: `${(rover.r / grid.h) * 100}%`, width: "16%", transform: `translate(-50%, -50%) rotate(${rover.heading + roverHeadingOffset}deg)`, pointerEvents: "none" }}>
            {/* @ts-ignore */}
            <model-viewer src={roverModelSrc} loading="eager" camera-orbit="0deg 0deg auto" disable-zoom interaction-prompt="none" environment-image="neutral" shadow-intensity="0.3" exposure="1.2" style={{ width: "100%", aspectRatio: "1 / 1", backgroundColor: "transparent" }} />
          </div>
        )}
      </div>
      {/* zoom controls */}
      <div className="absolute top-2 right-2 flex flex-col gap-1">
        {[["+", () => zoomBtn(1.4)], ["−", () => zoomBtn(1 / 1.4)], ["⤢", () => setV({ k: 1, x: 0, y: 0 })]].map(([t, fn]: any, i) => (
          <button key={i} onClick={fn} className="w-7 h-7 rounded-md bg-white/90 border border-line shadow-card text-ink text-sm font-bold hover:bg-white">{t}</button>
        ))}
      </div>
      {v.k > 1.01 && <div className="absolute bottom-2 left-2 text-[10px] font-mono bg-white/85 px-1.5 py-0.5 rounded border border-line text-slatey">{v.k.toFixed(1)}× · drag to pan · dbl-click reset</div>}
    </div>
  );
}
