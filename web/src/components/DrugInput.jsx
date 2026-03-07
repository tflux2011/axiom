import { useState, useRef, useEffect } from "react";
import styles from "./DrugInput.module.css";
import { useDrugList } from "../hooks/useDrugChecker";

export default function DrugInput({
  value,
  onChange,
  placeholder,
  label,
  disabled,
  onSubmit,
}) {
  const { filter } = useDrugList();
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(-1);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  useEffect(() => {
    if (value.length >= 2) {
      const matches = filter(value);
      setSuggestions(matches);
      setShowSuggestions(matches.length > 0);
      setSelectedIdx(-1);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [value, filter]);

  const handleSelect = (drug) => {
    onChange(drug);
    setShowSuggestions(false);
    setSuggestions([]);
  };

  const handleKeyDown = (e) => {
    if (!showSuggestions) {
      if (e.key === "Enter" && onSubmit) {
        e.preventDefault();
        onSubmit();
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIdx((prev) =>
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIdx((prev) =>
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
      case "Enter":
        e.preventDefault();
        if (selectedIdx >= 0 && selectedIdx < suggestions.length) {
          handleSelect(suggestions[selectedIdx]);
        } else if (onSubmit) {
          setShowSuggestions(false);
          onSubmit();
        }
        break;
      case "Escape":
        setShowSuggestions(false);
        break;
    }
  };

  const handleBlur = () => {
    setTimeout(() => setShowSuggestions(false), 200);
  };

  return (
    <div className={styles.wrapper}>
      <label className={styles.label}>{label}</label>
      <div className={styles.inputContainer}>
        <input
          ref={inputRef}
          type="text"
          className={styles.input}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
          onBlur={handleBlur}
          placeholder={placeholder}
          disabled={disabled}
          autoComplete="off"
          spellCheck="false"
          aria-label={label}
          aria-autocomplete="list"
          aria-controls="drug-suggestions"
          aria-expanded={showSuggestions}
        />
        {value && (
          <button
            className={styles.clearBtn}
            onClick={() => {
              onChange("");
              inputRef.current?.focus();
            }}
            aria-label="Clear input"
            type="button"
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M9 3L3 9M3 3l6 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        )}
      </div>

      {showSuggestions && (
        <ul
          ref={listRef}
          className={styles.suggestions}
          id="drug-suggestions"
          role="listbox"
        >
          {suggestions.map((drug, idx) => (
            <li
              key={drug}
              className={`${styles.suggestion} ${
                idx === selectedIdx ? styles.suggestionActive : ""
              }`}
              onClick={() => handleSelect(drug)}
              role="option"
              aria-selected={idx === selectedIdx}
            >
              {drug}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
