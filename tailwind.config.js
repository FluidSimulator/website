/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./app/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      fontFamily: {
        pixel: ['"Press Start 2P"', 'monospace'],
        game: ['"Fredoka"', 'sans-serif'],
        silk: ['"Silkscreen"', 'monospace'],
      },
      animation: {
        'float': 'float 3s ease-in-out infinite',
        'sparkle': 'sparkle 0.65s ease-out forwards',
        'poke-shake': 'pokeShake 0.5s ease-in-out',
        'fade-in': 'fadeIn 0.4s ease-out forwards',
        'slide-up': 'slideUp 0.5s ease-out forwards',
      },
      keyframes: {
        float: { '0%,100%': { transform: 'translateY(0)' }, '50%': { transform: 'translateY(-10px)' } },
        sparkle: { '0%': { opacity: 0, transform: 'scale(0) rotate(0)' }, '40%': { opacity: 1, transform: 'scale(1.3) rotate(140deg)' }, '100%': { opacity: 0, transform: 'scale(0) rotate(360deg)' } },
        pokeShake: { '0%,100%': { transform: 'translateX(0)' }, '20%': { transform: 'translateX(-3px) rotate(-3deg)' }, '40%': { transform: 'translateX(3px) rotate(3deg)' }, '60%': { transform: 'translateX(-2px) rotate(-1deg)' }, '80%': { transform: 'translateX(2px) rotate(1deg)' } },
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: { '0%': { opacity: 0, transform: 'translateY(20px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
      },
    },
  },
  plugins: [],
};
