import { useState, useEffect, useRef, useCallback } from "react";
import { checkInteraction, listDrugs, getStats, healthCheck } from "../api";

/**
 * Hook for drug interaction checking with debounced autocomplete.
 */
export function useDrugChecker() {
  const [drugA, setDrugA] = useState("");
  const [drugB, setDrugB] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const check = useCallback(async () => {
    if (!drugA.trim() || !drugB.trim()) {
      setError("Please enter both medications");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await checkInteraction(drugA, drugB);
      setResult(res);
      setHistory((prev) => {
        const entry = { ...res, timestamp: Date.now() };
        const updated = [entry, ...prev.filter(
          (h) => !(h.drug_a === res.drug_a && h.drug_b === res.drug_b)
        )];
        return updated.slice(0, 20); // keep last 20
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [drugA, drugB]);

  const clear = useCallback(() => {
    setDrugA("");
    setDrugB("");
    setResult(null);
    setError(null);
  }, []);

  return {
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
  };
}

/**
 * Hook for loading and filtering the drug list (autocomplete).
 */
export function useDrugList() {
  const [drugs, setDrugs] = useState([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listDrugs()
      .then((list) => {
        if (!cancelled) {
          setDrugs(list);
          setLoaded(true);
        }
      })
      .catch(() => {
        // silently fail — autocomplete is optional
      });
    return () => { cancelled = true; };
  }, []);

  const filter = useCallback(
    (query) => {
      if (!query || query.length < 2) return [];
      const q = query.toUpperCase();
      return drugs
        .filter((d) => d.includes(q))
        .slice(0, 8);
    },
    [drugs]
  );

  return { drugs, loaded, filter };
}

/**
 * Hook for server health status.
 */
export function useServerHealth() {
  const [healthy, setHealthy] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const h = await healthCheck();
        setHealthy(h.ready);

        const s = await getStats();
        setStats(s);
      } catch {
        setHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return { healthy, stats };
}
