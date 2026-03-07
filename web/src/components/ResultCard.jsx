import { useState } from "react";
import styles from "./ResultCard.module.css";

const SEVERITY_CONFIG = {
  major: {
    label: "MAJOR",
    className: "major",
    description: "Avoid combination — clinically significant risk",
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path d="M9 1.5L1.5 15h15L9 1.5z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
        <path d="M9 7v3M9 12.5v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  moderate: {
    label: "MODERATE",
    className: "moderate",
    description: "Use with caution — monitor closely",
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M9 6v4M9 12.5v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  minor: {
    label: "MINOR",
    className: "minor",
    description: "Generally safe — be aware",
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M6 9h6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    ),
  },
  none: {
    label: "NO INTERACTION",
    className: "safe",
    description: "No known interaction found",
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.5"/>
        <path d="M6 9l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
};

const PROCESS_STEPS = [
  { label: "Encoded", description: "Drugs mapped to hypervectors" },
  { label: "Compared", description: "Probe matched against Axiom Map" },
  { label: "Verified", description: "Result checked by Safety Governor" },
];

export default function ResultCard({ result }) {
  const [showHdcDetail, setShowHdcDetail] = useState(false);

  if (!result) return null;

  const config = SEVERITY_CONFIG[result.severity] || SEVERITY_CONFIG.none;

  return (
    <div
      className={`${styles.card} ${styles[config.className]} animate-fade-in-up`}
    >
      {/* Severity header */}
      <div className={styles.severityBar}>
        <div className={styles.severityInfo}>
          <span className={styles.severityIcon}>{config.icon}</span>
          <div className={styles.severityText}>
            <span className={styles.severityLabel}>{config.label}</span>
            <span className={styles.severityDesc}>{config.description}</span>
          </div>
        </div>
        {result.confidence_label && (
          <span className={`${styles.confidenceBadge} ${styles[`conf${result.confidence_label}`]}`}>
            {result.confidence_label} confidence
          </span>
        )}
      </div>

      {/* Drug Pair */}
      <div className={styles.drugPair}>
        <span className={styles.drugName}>{result.drug_a}</span>
        <span className={styles.separator}>
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            {result.found ? (
              <path d="M4 10h12M12 6l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M4 10h12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            )}
          </svg>
        </span>
        <span className={styles.drugName}>{result.drug_b}</span>
      </div>

      {/* Explanation — the main descriptive text */}
      {result.explanation && (
        <div className={styles.explanation}>
          <p className={styles.explanationText}>{result.explanation}</p>
        </div>
      )}

      {/* Process steps — shows what the engine did */}
      <div className={styles.processBar}>
        {PROCESS_STEPS.map((step, i) => (
          <div key={step.label} className={styles.processStep}>
            <div className={styles.stepDot}>
              <span className={styles.stepNumber}>{i + 1}</span>
            </div>
            <div className={styles.stepInfo}>
              <span className={styles.stepLabel}>{step.label}</span>
              <span className={styles.stepDesc}>{step.description}</span>
            </div>
            {i < PROCESS_STEPS.length - 1 && (
              <span className={styles.stepConnector} />
            )}
          </div>
        ))}
      </div>

      {/* Details (mechanism, clinical note) */}
      {result.found && (
        <div className={styles.details}>
          {result.relation !== "NONE" && (
            <div className={styles.row}>
              <span className={styles.rowLabel}>Relation</span>
              <span className={styles.rowValue}>
                {result.relation.replace(/_/g, " ")}
              </span>
            </div>
          )}

          {result.mechanism && (
            <div className={styles.row}>
              <span className={styles.rowLabel}>Mechanism</span>
              <span className={styles.rowValue}>{result.mechanism}</span>
            </div>
          )}

          {result.clinical_note && (
            <div className={`${styles.row} ${styles.noteRow}`}>
              <span className={styles.rowLabel}>Clinical Note</span>
              <span className={styles.rowValue}>
                {result.clinical_note}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Action items */}
      {result.action_items && result.action_items.length > 0 && (
        <div className={styles.actionSection}>
          <span className={styles.actionTitle}>What you should do</span>
          <ul className={styles.actionList}>
            {result.action_items.map((item, i) => (
              <li key={i} className={styles.actionItem}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      {/* HDC detail toggle */}
      <div className={styles.hdcToggle}>
        <button
          className={styles.hdcToggleBtn}
          onClick={() => setShowHdcDetail((prev) => !prev)}
          aria-expanded={showHdcDetail}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            className={`${styles.chevron} ${showHdcDetail ? styles.chevronOpen : ""}`}
            aria-hidden="true"
          >
            <path d="M4 5l3 3 3-3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          How did the engine reach this result?
        </button>
        {showHdcDetail && result.hdc_detail && (
          <p className={styles.hdcDetailText}>{result.hdc_detail}</p>
        )}
      </div>

      {/* Footer */}
      <div className={styles.footer}>
        <span className={styles.tag}>
          {result.interaction_type === "drug_condition"
            ? "Drug-Condition"
            : result.interaction_type === "drug_drug"
            ? "Drug-Drug"
            : "No Match"}
        </span>
        <span className={styles.confidence}>
          HDC {(result.confidence * 100).toFixed(1)}%
        </span>
        <span className={styles.engine}>AXIOM HDC</span>
      </div>
    </div>
  );
}
