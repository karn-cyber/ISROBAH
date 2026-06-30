import "@google/model-viewer";

// model-viewer is a web component; declare it for TSX.
declare global {
  namespace JSX {
    interface IntrinsicElements {
      "model-viewer": any;
    }
  }
}

export default function RoverViewer() {
  return (
    <div className="card p-3">
      <div className="text-sm font-semibold mb-2">Mission asset · Perseverance-class rover</div>
      {/* @ts-ignore custom element */}
      <model-viewer
        src="/rover.glb"
        alt="rover"
        loading="eager"
        camera-controls
        auto-rotate
        auto-rotate-delay="0"
        rotation-per-second="20deg"
        shadow-intensity="1"
        exposure="1.1"
        camera-orbit="45deg 65deg 4m"
        style={{ width: "100%", height: "280px", background: "#f8fafc", borderRadius: "10px" }}
      />
      <p className="text-[11px] text-slatey mt-2">
        Drag to orbit · model drives the animated marker on the traverse map.
      </p>
    </div>
  );
}
