import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',           // Use relative paths so assets work inside chrome-extension://
  build: {
    outDir: '../extension/dashboard',   // Build directly into the extension folder
    emptyOutDir: true,
  },
})
