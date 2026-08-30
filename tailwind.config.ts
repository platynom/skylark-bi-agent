import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#122018",
        forest: "#173f2c",
        moss: "#7fa76b",
        mint: "#dff4dc",
        paper: "#f6f7f2",
        amber: "#e7a83e"
      },
      boxShadow: { panel: "0 18px 50px rgba(23, 63, 44, 0.10)" }
    }
  },
  plugins: []
};
export default config;
