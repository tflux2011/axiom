/**
 * AXIOM Drug Interaction Checker — API Client
 *
 * Handles all communication with the FastAPI backend.
 * Falls back gracefully when the server is unreachable.
 *
 * Security: All inputs are validated before sending.
 * No credentials are stored or transmitted.
 */

const API_BASE = "http://127.0.0.1:8000/api";

/**
 * Sanitise a drug name before sending to the API.
 * Strips dangerous characters client-side as a defence-in-depth measure.
 */
function sanitiseDrugName(name) {
  if (typeof name !== "string") return "";
  return name.replace(/[^\w\s\-/(),.']/g, "").trim();
}

/**
 * Generic fetch wrapper with error handling and timeout.
 */
async function apiFetch(endpoint, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out. Is the AXIOM server running?");
    }
    if (err.message.includes("fetch")) {
      throw new Error(
        "Cannot connect to AXIOM server. Start it with: python server.py"
      );
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Check interaction between two drugs.
 */
export async function checkInteraction(drugA, drugB) {
  const cleanA = sanitiseDrugName(drugA);
  const cleanB = sanitiseDrugName(drugB);

  if (!cleanA || !cleanB) {
    throw new Error("Please enter valid drug names");
  }

  return apiFetch("/check", {
    method: "POST",
    body: JSON.stringify({ drug_a: cleanA, drug_b: cleanB }),
  });
}

/**
 * Check interactions among multiple drugs (polypharmacy).
 */
export async function checkMultipleInteractions(drugs) {
  const cleanDrugs = drugs
    .map(sanitiseDrugName)
    .filter((d) => d.length > 0);

  if (cleanDrugs.length < 2) {
    throw new Error("Please enter at least two drug names");
  }

  return apiFetch("/check-multiple", {
    method: "POST",
    body: JSON.stringify({ drugs: cleanDrugs }),
  });
}

/**
 * Get list of all known drugs.
 */
export async function listDrugs() {
  return apiFetch("/drugs");
}

/**
 * Get knowledge base statistics.
 */
export async function getStats() {
  return apiFetch("/stats");
}

/**
 * Health check.
 */
export async function healthCheck() {
  return apiFetch("/health");
}
