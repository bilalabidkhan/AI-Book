import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { useColorMode } from '@docusaurus/theme-common';
import { useThemeConfig } from '@docusaurus/theme-common';
import useBaseUrl from '@docusaurus/useBaseUrl';

import styles from './styles.module.css';

function HeaderNavLink({ to, href, label, position = 'left' }) {
  const toUrl = useBaseUrl(to);

  return (
    <Link
      className={clsx(
        'navbar__item navbar__link',
        position === 'right' && 'navbar__item--right',
      )}
      to={toUrl}
      {...(href && { href })}
    >
      {label}
    </Link>
  );
}

function NavbarMobileSidebarToggle() {
  return (
    <div
      aria-label="Navigation bar toggle"
      className="navbar__toggle"
      role="button"
      tabIndex={0}
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="30"
        height="30"
        viewBox="0 0 30 30"
        role="img"
        focusable="false"
      >
        <path
          stroke="currentColor"
          strokeLinecap="round"
          strokeMiterlimit="10"
          strokeWidth="2"
          d="M4 7h22M4 15h22M4 23h22"
        />
      </svg>
    </div>
  );
}

function ThemeToggleButton() {
  const { colorMode, setColorMode } = useColorMode();
  return (
    <div
      aria-label="Color mode toggle"
      className="navbar__item"
      role="button"
      tabIndex={0}
      onClick={() => setColorMode(colorMode === 'dark' ? 'light' : 'dark')}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          setColorMode(colorMode === 'dark' ? 'light' : 'dark');
        }
      }}
    >
      {colorMode === 'dark' ? '☀️' : '🌙'}
    </div>
  );
}

export default function Header() {
  const themeConfig = useThemeConfig();
  const { navbar: { title, items = [], hideOnScroll = false } } = themeConfig;

  return (
    <header
      className={clsx(
        'navbar',
        'navbar--fixed-top',
        hideOnScroll && 'navbar--fixed-top--hide-on-scroll',
        styles.navbar
      )}
    >
      <div className="navbar__inner">
        <div className="navbar__items">
          <NavbarMobileSidebarToggle />
          <Link className="navbar__brand" to="/">
            <span className="navbar__title">{title}</span>
          </Link>
          {items.map((item, idx) => (
            <HeaderNavLink
              key={idx}
              to={item.to}
              href={item.href}
              label={item.label}
              position={item.position}
            />
          ))}
        </div>
        <div className="navbar__items navbar__items--right">
          <ThemeToggleButton />
        </div>
      </div>
      <div role="presentation" className="navbar-sidebar__seashell">
        <div className="navbar-sidebar__brand">
          <Link className="navbar__brand" to="/">
            <span className="navbar__title">{title}</span>
          </Link>
        </div>
        <div className="navbar-sidebar__items">
          <div className="navbar-sidebar__item menu">
            <ul className="menu__list">
              {items.map((item, idx) => (
                <li key={idx} className="menu__list-item">
                  <Link
                    className="menu__link"
                    to={item.to}
                    {...(item.href && { href: item.href })}
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </header>
  );
}