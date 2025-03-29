"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [pathname]);

  // Check if link is active
  const isActive = (path) => {
    if (path === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(path);
  };

  return (
    <header>
      
      <div className="header-container">
        <div className="racing-flag"></div>
        
        <div className="logo-container">
          <Link href="/" className="logo">
            <span className="logo-f1">F1</span> Hub
          </Link>
        </div>

        <nav>
          <ul className={`nav-list ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
            <li className="nav-item">
              <Link 
                href="/" 
                className={`nav-link ${isActive('/') ? 'active' : ''}`}
              >
                Home
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                href="/news" 
                className={`nav-link ${isActive('/news') ? 'active' : ''}`}
              >
                News
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                href="/schedule" 
                className={`nav-link ${isActive('/schedule') ? 'active' : ''}`}
              >
                Race Calendar
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                href="/standings" 
                className={`nav-link ${isActive('/standings') ? 'active' : ''}`}
              >
                Standings
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                href="/chat"
                className={`nav-link ${isActive('/chat') ? 'active' : ''}`}
              >
                Chat
              </Link>
            </li>
          </ul>
        </nav>

        <div className="nav-actions">
          
          <Link href="/login" className="nav-button">
            Sign In
          </Link>
          
          <button 
            className="mobile-menu-button" 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
          >
            {isMobileMenuOpen ? (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
            )}
          </button>
        </div>
      </div>
    </header>
  );
}