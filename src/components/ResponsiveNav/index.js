import React, { useState, useEffect } from 'react';
import clsx from 'clsx';

import styles from './styles.module.css';

// Responsive navigation component that adapts to different screen sizes
export default function ResponsiveNav({ children, className = '' }) {
  const [isMobile, setIsMobile] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  // Check screen size and update mobile state
  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth < 997); // Matches Docusaurus' mobile breakpoint
    };

    // Initial check
    checkScreenSize();

    // Add event listener
    window.addEventListener('resize', checkScreenSize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', checkScreenSize);
    };
  }, []);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const closeMenu = () => {
    setIsMenuOpen(false);
  };

  return (
    <nav
      className={clsx(
        styles.responsiveNav,
        className,
        isMobile && styles.responsiveNavMobile,
        isMenuOpen && styles.responsiveNavOpen
      )}
      aria-label="Main navigation"
    >
      {isMobile ? (
        // Mobile view
        <div className={styles.mobileWrapper}>
          <button
            className={clsx(styles.menuToggle, isMenuOpen && styles.menuToggleOpen)}
            onClick={toggleMenu}
            aria-expanded={isMenuOpen}
            aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          >
            <span className={styles.burger} />
          </button>

          {isMenuOpen && (
            <div className={styles.mobileMenu}>
              <div className={styles.mobileMenuContent}>
                {React.Children.map(children, (child) =>
                  React.cloneElement(child, { onClick: closeMenu, isMobile: true })
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        // Desktop view
        <div className={styles.desktopNav}>
          {children}
        </div>
      )}
    </nav>
  );
}