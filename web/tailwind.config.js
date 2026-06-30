/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0f172a",
        slatey: "#475569",
        line: "#e2e8f0",
        accent: "#0e7490",   // teal-700
        accent2: "#4f46e5",  // indigo-600
        ice: "#0891b2",
        paper: "#ffffff",
        panel: "#f8fafc",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.05)",
      },
    },
  },
  plugins: [],
};
