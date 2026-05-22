import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../moshi/moshi/web_client",
    emptyOutDir: true,
  },
});
