"use client";

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { motion } from 'framer-motion';

export default function Navigation() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const [dimensions, setDimensions] = useState({ width: 0, left: 0 });
  const navRef = useRef(null);
  const activeItemRef = useRef(null);

  // Close mobile menu on route change
  useEffect(() => {
    setIsMobileMenuOpen(false);
    
    // Update the underline position when path changes
    if (activeItemRef.current && navRef.current) {
      const navItem = activeItemRef.current;
      const rect = navItem.getBoundingClientRect();
      const navRect = navRef.current.getBoundingClientRect();
      
      setDimensions({
        width: rect.width,
        left: rect.left - navRect.left
      });
    }
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
        {/* Removing the racing flag element that's causing the issue */}
        {/* <div className="racing-flag"></div> */}
        
        <div className="logo-container">
          <Link href="/" className="logo">
            <motion.div 
              className="logo-icon"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Image 
                src="/favicon.ico" 
                alt="F1 Hub Logo" 
                width={24} 
                height={24}
                className="favicon-logo"
              />
            </motion.div>
            <motion.span
              initial={{ opacity: 0, x: -5 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              Hub
            </motion.span>
          </Link>
        </div>

        <nav ref={navRef}>
          <ul className={`nav-list ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
            {[
              { path: '/', label: 'Home' },
              { path: '/news', label: 'News' },
              { path: '/schedule', label: 'Race Calendar' },
              { path: '/standings', label: 'Standings' },
              { path: '/chat', label: 'Chat' }
            ].map(({ path, label }) => (
              <motion.li 
                key={path} 
                className="nav-item"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Link 
                  href={path} 
                  className={`nav-link ${isActive(path) ? 'active' : ''}`}
                  ref={isActive(path) ? activeItemRef : null}
                >
                  {label}
                  {isActive(path) && (
                    <motion.span 
                      className="active-indicator"
                      layoutId="activeIndicator"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </Link>
              </motion.li>
            ))}
          </ul>
        </nav>

        <div className="nav-actions">
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link href="/login" className="nav-button">
              Sign In
            </Link>
          </motion.div>
          
          <motion.button 
            className="mobile-menu-button" 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle menu"
            whileTap={{ scale: 0.9 }}
          >
            {isMobileMenuOpen ? (
              <motion.svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
                initial={{ rotate: 0 }}
                animate={{ rotate: 90 }}
                transition={{ duration: 0.2 }}
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </motion.svg>
            ) : (
              <motion.svg 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 24 24" 
                stroke="currentColor"
                initial={{ rotate: 90 }}
                animate={{ rotate: 0 }}
                transition={{ duration: 0.2 }}
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
              </motion.svg>
            )}
          </motion.button>
        </div>
      </div>
    </header>
  );
}