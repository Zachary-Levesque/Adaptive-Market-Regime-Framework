/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        regime: {
          bull: '#10b981', // Emerald
          compression: '#3b82f6', // Blue
          bear: '#f59e0b', // Amber
          crisis: '#ef4444', // Red
        }
      }
    },
  },
  plugins: [],
}
