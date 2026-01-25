import React from 'react';
import { useLocation } from '@docusaurus/router';
import Link from '@docusaurus/Link';
import { useDoc } from '@docusaurus/plugin-content-docs/client';

import styles from './styles.module.css';

function useBreadcrumbItems() {
  const location = useLocation();
  const { pathname } = location;
  const doc = useDoc();

  // Create breadcrumb items based on the current path
  const pathSegments = pathname
    .replace(/\/$/, '') // Remove trailing slash
    .split('/')
    .filter(segment => segment !== ''); // Remove empty segments

  const breadcrumbItems = [];

  // Add home link
  breadcrumbItems.push({
    label: 'Home',
    to: '/',
    isLast: pathSegments.length === 0,
  });

  // Build path incrementally for each segment
  let currentPath = '';
  pathSegments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    const isLast = index === pathSegments.length - 1;

    // Try to get a nice label for the segment
    let label = segment
      .replace(/-/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize first letter of each word

    // If it's a document page, try to get the title from the doc data
    if (doc && isLast && doc.sidebar_label) {
      label = doc.sidebar_label;
    } else if (doc && isLast && doc.title) {
      label = doc.title;
    }

    breadcrumbItems.push({
      label,
      to: currentPath,
      isLast,
    });
  });

  return breadcrumbItems;
}

export default function Breadcrumb() {
  const breadcrumbItems = useBreadcrumbItems();

  if (breadcrumbItems.length <= 1) {
    // Don't show breadcrumbs if there's only one item (home)
    return null;
  }

  return (
    <nav className={styles.breadcrumb} aria-label="Breadcrumb">
      <ol className={styles.breadcrumbList}>
        {breadcrumbItems.map((item, index) => (
          <li key={index} className={styles.breadcrumbItem}>
            {item.isLast ? (
              <span className={styles.breadcrumbCurrent}>{item.label}</span>
            ) : (
              <>
                <Link
                  to={item.to}
                  className={styles.breadcrumbLink}
                  aria-label={`Go to ${item.label}`}
                >
                  {item.label}
                </Link>
                <span className={styles.breadcrumbSeparator} aria-hidden="true">
                  {' > '}
                </span>
              </>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}