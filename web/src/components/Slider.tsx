interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  fmt?: (v: number) => string;
  hint?: string;
}

export default function Slider({ label, value, min, max, step, onChange, fmt, hint }: Props) {
  return (
    <div className="mb-3">
      <div className="flex items-baseline justify-between">
        <span className="label">{label}</span>
        <span className="text-sm font-mono text-ink">{fmt ? fmt(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      {hint && <p className="text-[11px] text-slatey mt-0.5 leading-snug">{hint}</p>}
    </div>
  );
}
