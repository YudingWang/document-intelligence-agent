import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    proxy: {
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/docs": "http://localhost:8000",
      "/redoc": "http://localhost:8000",
      "/openapi.json": "http://localhost:8000",
    },
  },
});
