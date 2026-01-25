import React, { useState, useEffect } from 'react';
import { useLocation } from '@docusaurus/router';

import styles from './styles.module.css';

// This is a simplified Table of Contents component
// In a real implementation, this would extract headings from the page content
export default function Toc() {
  const location = useLocation();
  const [activeId, setActiveId] = useState('');
  const [headings, setHeadings] = useState([]);

  // In a real implementation, we would extract headings from the document
  // For now, we'll use a mock structure that would typically come from page metadata
  useEffect(() => {
    // This is a mock implementation - in a real app, this data would come from the page
    const mockHeadings = [
      { id: 'introduction', text: 'Introduction', level: 2 },
      { id: 'setup', text: 'Setup', level: 2 },
      { id: 'configuration', text: 'Configuration', level: 2 },
      { id: 'advanced-features', text: 'Advanced Features', level: 2 },
      { id: 'troubleshooting', text: 'Troubleshooting', level: 2 },
      { id: 'faq', text: 'FAQ', level: 3 },
    ];

    setHeadings(mockHeadings);
  }, [location.pathname]);

  // Handle scroll events to highlight active heading
  useEffect(() => {
    const handleScroll = () => {
      // In a real implementation, this would check which heading is in view
      // For now, we'll just use a simple approach
      const scrollPosition = window.scrollY + 100; // offset to account for header

      // Find the heading that corresponds to the current scroll position
      const elements = headings.map(heading =>
        document.getElementById(heading.id)
      ).filter(Boolean);

      let activeHeading = null;
      for (const element of elements) {
        if (element && element.offsetTop <= scrollPosition) {
          activeHeading = element.id;
        } else {
          break;
        }
      }

      if (activeHeading) {
        setActiveId(activeHeading);
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Set initial active heading

    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, [headings]);

  const handleHeadingClick = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setActiveId(id);
    }
  };

  if (headings.length === 0) {
    return null;
  }

  return (
    <nav className={styles.toc} aria-label="Table of contents">
      <div className={styles.tocHeader}>
        <h3 className={styles.tocTitle}>On this page</h3>
      </div>
      <ul className={styles.tocList}>
        {headings.map((heading) => (
          <li
            key={heading.id}
            className={`${styles.tocItem} ${styles[`tocLevel${heading.level}`]}`}
          >
            <a
              href={`#${heading.id}`}
              onClick={(e) => {
                e.preventDefault();
                handleHeadingClick(heading.id);
              }}
              className={`${styles.tocLink} ${
                activeId === heading.id ? styles.tocLinkActive : ''
              }`}
            >
              {heading.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}