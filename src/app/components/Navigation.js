"use client";
import Link from 'next/link';
import { useState } from 'react';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  
  const routes = [
    { name: 'Home', path: '/' },
    { name: 'News', path: '/news' },
    { name: 'Schedule', path: '/schedule' },
    { name: 'Standings', path: '/standings' },
    { name: 'Drivers', path: '/drivers' },
    { name: 'Chat', path: '/chat' },
  ];

  return (
    <nav className="bg-red-600 text-white">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center">
            <Link href="/" className="text-xl font-bold">F1 Hub</Link>
          </div>
          
          <div className="hidden md:block">
            <div className="flex space-x-4">
              {routes.map((route) => (
                <Link
                  key={route.path}
                  href={route.path}
                  className={`px-3 py-2 rounded-md text-sm font-medium ${
                    pathname === route.path ? 'bg-red-800' : 'hover:bg-red-700'
                  }`}
                >
                  {route.name}
                </Link>
              ))}
            </div>
          </div>
          
          <div className="md:hidden">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md hover:bg-red-700"
            >
              <span className="sr-only">Open menu</span>
              {/* Hamburger icon */}
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {routes.map((route) => (
              <Link
                key={route.path}
                href={route.path}
                className={`block px-3 py-2 rounded-md text-base font-medium ${
                  pathname === route.path ? 'bg-red-800' : 'hover:bg-red-700'
                }`}
                onClick={() => setMobileMenuOpen(false)}
              >
                {route.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}