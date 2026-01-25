import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import { useThemeConfig } from '@docusaurus/theme-common';

import styles from './styles.module.css';

function MobileMenuNavLink({ to, href, label, onClick, className = '' }) {
  const toUrl = to;

  return (
    <Link
      className={`${styles.navLink} ${className}`}
      to={toUrl}
      {...(href && { href })}
      onClick={onClick}
    >
      {label}
    </Link>
  );
}

export default function MobileMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const themeConfig = useThemeConfig();
  const { navbar: { items = [] } } = themeConfig;

  // Close menu when route changes
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const closeMenu = () => {
    setIsOpen(false);
  };

  return (
    <div className={styles.mobileMenu}>
      <button
        className={`${styles.menuButton} ${isOpen ? styles.menuButtonOpen : ''}`}
        onClick={toggleMenu}
        aria-label={isOpen ? 'Close menu' : 'Open menu'}
        aria-expanded={isOpen}
      >
        <div className={styles.burgerLine}></div>
        <div className={styles.burgerLine}></div>
        <div className={styles.burgerLine}></div>
      </button>

      {isOpen && (
        <div className={styles.menuOverlay} onClick={closeMenu}>
          <div className={styles.menuContent} onClick={(e) => e.stopPropagation()}>
            <div className={styles.menuHeader}>
              <div className={styles.menuTitle}>Menu</div>
              <button
                className={styles.closeButton}
                onClick={closeMenu}
                aria-label="Close menu"
              >
                ✕
              </button>
            </div>

            <nav className={styles.nav}>
              <ul className={styles.navList}>
                {items.map((item, idx) => (
                  <li key={idx} className={styles.navItem}>
                    <MobileMenuNavLink
                      to={item.to}
                      href={item.href}
                      label={item.label}
                      onClick={closeMenu}
                      className={styles.navItemLink}
                    />
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </div>
      )}
    </div>
  );
}