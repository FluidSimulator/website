import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Pixel Art Adventure — Interactive Elemental Regions',
  description: 'Explore Wind, Fire, Water, and Paint regions with interactive fluid simulations powered by Taichi.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
