import styles from "./Header.module.css";

export default function Header({ healthy, stats }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.brand}>
          <div className={styles.mark}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="M10 2L18 18H2L10 2Z" stroke="currentColor" strokeWidth="1.5" fill="none" />
              <circle cx="10" cy="12" r="2" fill="currentColor" />
            </svg>
          </div>
          <span className={styles.wordmark}>
            AXIOM<span className={styles.rx}>Rx</span>
          </span>
        </div>

        <div className={styles.meta}>
          {stats && (
            <span className={styles.stat}>
              {stats.unique_drugs} drugs / {stats.total_interactions} interactions
            </span>
          )}
          <div className={styles.statusGroup}>
            <span
              className={`${styles.dot} ${
                healthy === true
                  ? styles.online
                  : healthy === false
                  ? styles.offline
                  : styles.pending
              }`}
            />
            <span className={styles.statusLabel}>
              {healthy === true
                ? "Online"
                : healthy === false
                ? "Offline"
                : "..."}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
