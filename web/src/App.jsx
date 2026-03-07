import { useState, useCallback } from "react";
import Header from "./components/Header";
import DrugInput from "./components/DrugInput";
import ResultCard from "./components/ResultCard";
import { useDrugChecker, useServerHealth } from "./hooks/useDrugChecker";
import styles from "./App.module.css";

function App() {
  const { healthy, stats } = useServerHealth();
  const {
    drugA,
    setDrugA,
    drugB,
    setDrugB,
    result,
    loading,
    error,
    history,
    check,
    clear,
  } = useDrugChecker();

  const [showHistory, setShowHistory] = useState(false);

  const handleCheck = useCallback(() => {
    if (!loading) check();
  }, [loading, check]);

  const isReady = drugA.trim().length > 0 && drugB.trim().length > 0;

  return (
    <>
      <Header healthy={healthy} stats={stats} />

      <main className={styles.main}>
        {/* Hero */}
        <section className={styles.hero}>
          <h2 className={styles.heroTitle}>Drug Interaction Check</h2>
          <p className={styles.heroSubtitle}>
            Enter two medications. The HDC engine runs entirely offline
            &mdash; no data leaves this device.
          </p>
        </section>

        {/* Input panel */}
        <section className={styles.panel}>
          <div className={styles.inputRow}>
            <DrugInput
              value={drugA}
              onChange={setDrugA}
              placeholder="e.g. Warfarin"
              label="Medication A"
              disabled={loading}
              onSubmit={handleCheck}
            />

            <div className={styles.divider}>
              <span className={styles.dividerLine} />
              <span className={styles.dividerPlus}>+</span>
              <span className={styles.dividerLine} />
            </div>

            <DrugInput
              value={drugB}
              onChange={setDrugB}
              placeholder="e.g. Aspirin"
              label="Medication B"
              disabled={loading}
              onSubmit={handleCheck}
            />
          </div>

          <div className={styles.actions}>
            <button
              className={`${styles.checkBtn} ${loading ? styles.loading : ""}`}
              onClick={handleCheck}
              disabled={!isReady || loading || !healthy}
              aria-label="Check drug interaction"
            >
              {loading ? (
                <>
                  <span className={styles.spinner} />
                  Checking
                </>
              ) : (
                "Check Interaction"
              )}
            </button>

            {(drugA || drugB || result) && (
              <button
                className={styles.resetBtn}
                onClick={clear}
                disabled={loading}
                aria-label="Clear all fields"
              >
                Clear
              </button>
            )}
          </div>

          {error && (
            <div className={`${styles.errorBar} animate-shake`}>{error}</div>
          )}

          {!healthy && healthy !== null && (
            <div className={styles.offlineBar}>
              Server offline. Run <code>python server.py</code> to start.
            </div>
          )}
        </section>

        {/* Result */}
        {result && (
          <section className={styles.resultSection}>
            <ResultCard result={result} />
          </section>
        )}

        {/* History */}
        {history.length > 0 && (
          <section className={styles.historySection}>
            <button
              className={styles.historyToggle}
              onClick={() => setShowHistory((prev) => !prev)}
            >
              <span className={styles.historyToggleIcon}>
                {showHistory ? "\u25BE" : "\u25B8"}
              </span>
              History
              <span className={styles.historyCount}>{history.length}</span>
            </button>

            {showHistory && (
              <div className={styles.historyList}>
                {history.map((h) => (
                  <button
                    key={`${h.drug_a}-${h.drug_b}-${h.timestamp}`}
                    className={styles.historyItem}
                    onClick={() => {
                      setDrugA(h.drug_a);
                      setDrugB(h.drug_b);
                    }}
                    aria-label={`Re-check ${h.drug_a} and ${h.drug_b}`}
                  >
                    <span className={styles.historyPair}>
                      {h.drug_a} + {h.drug_b}
                    </span>
                    <span
                      className={`${styles.historyBadge} ${
                        styles[`badge_${h.severity || "none"}`]
                      }`}
                    >
                      {h.severity?.toUpperCase() || "SAFE"}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>
        )}

        {/* Footer */}
        <footer className={styles.footer}>
          <p>
            AXIOMRx &middot;{" "}
            <a
              href="https://github.com/tflux2011/axiom"
              target="_blank"
              rel="noopener noreferrer"
            >
              AXIOM HDC Framework
            </a>
          </p>
          <p className={styles.disclaimer}>
            For informational purposes only. Always consult a licensed
            pharmacist or physician.
          </p>
        </footer>
      </main>
    </>
  );
}

export default App;
