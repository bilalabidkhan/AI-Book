import React, { useState, useRef, useEffect } from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

// This is a simplified search component that demonstrates UI enhancements
// In a real Docusaurus implementation, you would typically use the built-in search
// but this shows how to create an enhanced UI for search functionality
export default function Search() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const searchRef = useRef(null);

  // Simulate search functionality
  const handleSearch = (query) => {
    if (query.trim() === '') {
      setSearchResults([]);
      return;
    }

    // In a real implementation, this would be an API call to search
    // For now, we'll just simulate some results
    const mockResults = [
      { id: 1, title: `Search result for "${query}" - Item 1`, url: '#' },
      { id: 2, title: `Search result for "${query}" - Item 2`, url: '#' },
      { id: 3, title: `More results for "${query}" - Item 3`, url: '#' },
    ];

    setSearchResults(mockResults);
    setShowResults(true);
  };

  const handleInputChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    handleSearch(value);
  };

  const handleFocus = () => {
    setIsFocused(true);
    if (searchQuery) {
      setShowResults(true);
    }
  };

  const handleBlur = () => {
    // Delay hiding results to allow for clicking on them
    setTimeout(() => {
      setIsFocused(false);
      setShowResults(false);
    }, 200);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      setShowResults(false);
      e.target.blur();
    }
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowResults(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <div className={styles.searchContainer} ref={searchRef}>
      <div className={clsx(
        styles.searchWrapper,
        isFocused && styles.searchWrapperFocused
      )}>
        <div className={styles.searchIcon}>🔍</div>
        <input
          type="text"
          value={searchQuery}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          placeholder="Search documentation..."
          className={styles.searchInput}
          aria-label="Search"
        />
        {searchQuery && (
          <button
            onClick={() => {
              setSearchQuery('');
              setSearchResults([]);
            }}
            className={styles.clearButton}
            aria-label="Clear search"
          >
            ✕
          </button>
        )}
      </div>

      {showResults && searchResults.length > 0 && (
        <div className={styles.searchResults}>
          <div className={styles.resultsHeader}>
            <span className={styles.resultsCount}>
              {searchResults.length} results for "{searchQuery}"
            </span>
          </div>
          <ul className={styles.resultsList}>
            {searchResults.map((result) => (
              <li key={result.id} className={styles.resultItem}>
                <a href={result.url} className={styles.resultLink}>
                  <div className={styles.resultTitle}>{result.title}</div>
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}