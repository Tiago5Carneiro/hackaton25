/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'f1-red': 'var(--f1-red)',
        'f1-red-dark': 'var(--f1-red-dark)',
        'f1-black': 'var(--f1-black)',
        'f1-gray-dark': 'var(--f1-gray-dark)',
        'f1-gray': 'var(--f1-gray)',
        'f1-gray-light': 'var(--f1-gray-light)',
        'f1-gold': 'var(--f1-gold)',
        'f1-silver': 'var(--f1-silver)',
        'f1-bronze': 'var(--f1-bronze)',
      },
    },
  },
  plugins: [],
}