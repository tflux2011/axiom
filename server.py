"""
AXIOM Drug Interaction Checker — FastAPI Backend

Provides a REST API for the React frontend to query drug interactions
via the AXIOM HDC engine. Fully offline after initial setup.

Endpoints:
    POST /api/check          — Check two drugs for interactions
    POST /api/check-multiple — Polypharmacy screening (multiple drugs)
    GET  /api/drugs          — List all known drugs
    GET  /api/stats          — Knowledge base statistics
    GET  /api/health         — Health check

Security:
    - CORS restricted to localhost in development
    - All inputs sanitised before processing
    - No external API calls — fully offline
    - Rate limiting via simple in-memory tracking
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from axiom_hdc.drug_checker import DrugInteractionChecker, _sanitise_drug_name
from axiom_hdc.utils import setup_logging

logger = logging.getLogger("axiom.server")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DISTILLED_DIR = DATA_DIR / "distilled"
DATASET_PATH = DATA_DIR / "drug_interactions.json"

# Rate limiting: max requests per IP per minute
RATE_LIMIT_MAX = 120
RATE_LIMIT_WINDOW = 60  # seconds

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

checker: DrugInteractionChecker | None = None
_rate_limits: dict[str, list[float]] = defaultdict(list)


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the drug interaction checker on startup."""
    global checker
    setup_logging()

    logger.info("Starting AXIOM Drug Interaction Server...")

    checker = DrugInteractionChecker()

    if DISTILLED_DIR.exists() and (DISTILLED_DIR / "axiom_map.pt").exists():
        # Load pre-distilled state (fast startup)
        logger.info("Loading pre-distilled Axiom Map from %s", DISTILLED_DIR)
        checker.load_dataset(DATASET_PATH)
        checker.load(DISTILLED_DIR)
        logger.info("Axiom Map loaded — ready for queries")
    elif DATASET_PATH.exists():
        # First run: distil from raw data
        logger.info("No pre-distilled map found. Running distillation...")
        checker.load_dataset(DATASET_PATH)
        checker.distill()
        checker.save(DISTILLED_DIR)
        logger.info("Distillation complete — saved for future restarts")
    else:
        logger.error("No dataset found at %s", DATASET_PATH)
        raise RuntimeError(f"Dataset not found: {DATASET_PATH}")

    logger.info("Server ready! %s", checker)

    yield

    logger.info("AXIOM server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AXIOM Drug Interaction Checker",
    description=(
        "Offline drug interaction checking powered by "
        "Hyperdimensional Computing. No internet required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Rate limiting middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries
    _rate_limits[client_ip] = [
        t for t in _rate_limits[client_ip]
        if now - t < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    _rate_limits[client_ip].append(now)
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CheckRequest(BaseModel):
    """Request to check interaction between two drugs."""

    drug_a: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="First drug name (generic or brand)",
        examples=["warfarin"],
    )
    drug_b: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Second drug name (generic or brand)",
        examples=["aspirin"],
    )

    @field_validator("drug_a", "drug_b")
    @classmethod
    def validate_drug_name(cls, v: str) -> str:
        """Sanitise drug names."""
        cleaned = _sanitise_drug_name(v)
        if not cleaned:
            raise ValueError("Invalid drug name")
        return v.strip()


class CheckMultipleRequest(BaseModel):
    """Request to check interactions among multiple drugs."""

    drugs: list[str] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="List of drug names to cross-check",
        examples=[["warfarin", "aspirin", "ibuprofen"]],
    )

    @field_validator("drugs")
    @classmethod
    def validate_drug_list(cls, v: list[str]) -> list[str]:
        """Sanitise all drug names in the list."""
        for drug in v:
            cleaned = _sanitise_drug_name(drug)
            if not cleaned:
                raise ValueError(f"Invalid drug name: {drug}")
        return [d.strip() for d in v]


class InteractionResponse(BaseModel):
    """Response for a single interaction check."""

    drug_a: str
    drug_b: str
    found: bool
    severity: str
    mechanism: str
    clinical_note: str
    confidence: float
    interaction_type: str
    relation: str
    is_unsafe: bool
    is_contraindicated: bool
    explanation: str  # Plain-English summary of the finding
    hdc_detail: str   # What the HDC engine did, step by step
    action_items: list[str]  # Concrete next steps for the user
    confidence_label: str  # Human-readable confidence descriptor


class StatsResponse(BaseModel):
    """Knowledge base statistics."""

    total_interactions: int
    unique_drugs: int
    aliases_count: int
    axiom_map_dimensions: int
    facts_distilled: int
    entities_in_memory: int
    map_size_bytes: int
    is_distilled: bool


# ---------------------------------------------------------------------------
# Descriptive response builder
# ---------------------------------------------------------------------------

def _build_descriptors(
    result,
) -> tuple[str, str, list[str], str]:
    """Build human-readable explanation, HDC detail, action items, and
    confidence label from an InteractionResult.

    Returns:
        (explanation, hdc_detail, action_items, confidence_label)
    """
    drug_a = result.drug_a
    drug_b = result.drug_b
    conf = result.confidence
    conf_pct = abs(conf * 100)

    # --- Confidence label ---
    if conf >= 0.10:
        confidence_label = "High"
    elif conf >= 0.04:
        confidence_label = "Moderate"
    elif conf >= 0.01:
        confidence_label = "Low"
    else:
        confidence_label = "Minimal"

    # --- HDC detail (what the engine did) ---
    hdc_detail = (
        f"The AXIOM engine encoded {drug_a} and {drug_b} as 10,000-dimensional "
        f"hypervectors, bound them with the INTERACTS_WITH relation using cyclic "
        f"permutation, and measured the cosine similarity of the resulting probe "
        f"against the Axiom Map. The similarity score was {conf_pct:.1f}%, "
        f"classified as {confidence_label.lower()} confidence. "
        f"All processing ran offline on this device — no data was transmitted."
    )

    if result.found:
        severity = result.severity
        # --- Explanation ---
        severity_desc = {
            "major": "a major, clinically significant",
            "moderate": "a moderate",
            "minor": "a minor, low-risk",
        }.get(severity, "a")

        if result.relation == "POTENTIAL_INTERACTION":
            explanation = (
                f"The HDC engine detected a pattern similarity between {drug_a} "
                f"and {drug_b} that suggests a potential interaction, even though "
                f"no exact match exists in the curated database. This warrants "
                f"further investigation with a pharmacist."
            )
        else:
            explanation = (
                f"A known interaction was found between {drug_a} and {drug_b}. "
                f"This is classified as {severity_desc} interaction. "
                f"{result.mechanism}."
            )

        # --- Action items ---
        action_items = []
        if severity == "major":
            action_items = [
                f"Avoid combining {drug_a} and {drug_b} unless directed by a physician",
                "Contact your prescribing doctor or pharmacist immediately",
                "Do not stop either medication abruptly without medical advice",
                "If already taking both, seek urgent clinical review",
            ]
        elif severity == "moderate":
            action_items = [
                "Inform your doctor or pharmacist about both medications",
                f"Watch for increased side effects from either {drug_a} or {drug_b}",
                "Attend any recommended monitoring (e.g. blood tests, INR checks)",
                "Report unusual symptoms promptly",
            ]
        elif severity == "minor":
            action_items = [
                "Generally safe to use together",
                "Be aware of possible mild side effects",
                "Mention both medications at your next appointment",
            ]
        else:
            action_items = [
                "Consult a pharmacist for personalised advice",
            ]
    else:
        explanation = (
            f"No known interaction was found between {drug_a} and {drug_b} in the "
            f"AXIOM knowledge base ({checker.get_stats()['total_interactions']} "
            f"curated interactions, {checker.get_stats()['unique_drugs']} drugs). "
            f"The HDC similarity score of {conf_pct:.1f}% is below the safety "
            f"threshold, indicating these two medications do not share a recognised "
            f"interaction pattern. This does not guarantee absolute safety."
        )
        action_items = [
            "No action needed based on current data",
            "Always disclose all medications to your healthcare provider",
            "Check again if your medication regimen changes",
            "This result covers drug-drug interactions only — other factors may apply",
        ]

    return explanation, hdc_detail, action_items, confidence_label


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine": "AXIOM HDC",
        "offline": True,
        "ready": checker is not None and checker._is_distilled,
    }


@app.post("/api/check", response_model=InteractionResponse)
async def check_interaction(req: CheckRequest):
    """
    Check for interactions between two drugs.

    Accepts both generic names (warfarin) and brand names (Coumadin).
    Returns severity, mechanism, clinical notes, and HDC confidence.
    """
    if checker is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    result = checker.check(req.drug_a, req.drug_b)

    explanation, hdc_detail, action_items, confidence_label = _build_descriptors(
        result
    )

    return InteractionResponse(
        drug_a=result.drug_a,
        drug_b=result.drug_b,
        found=result.found,
        severity=result.severity,
        mechanism=result.mechanism,
        clinical_note=result.clinical_note,
        confidence=round(result.confidence, 4),
        interaction_type=result.interaction_type,
        relation=result.relation,
        is_unsafe=result.is_unsafe,
        is_contraindicated=result.is_contraindicated,
        explanation=explanation,
        hdc_detail=hdc_detail,
        action_items=action_items,
        confidence_label=confidence_label,
    )


@app.post("/api/check-multiple", response_model=list[InteractionResponse])
async def check_multiple_interactions(req: CheckMultipleRequest):
    """
    Polypharmacy screening: check all pairwise interactions in a drug list.

    Returns interactions sorted by severity (major first).
    """
    if checker is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    results = checker.check_multiple(req.drugs)

    return [
        InteractionResponse(
            drug_a=r.drug_a,
            drug_b=r.drug_b,
            found=r.found,
            severity=r.severity,
            mechanism=r.mechanism,
            clinical_note=r.clinical_note,
            confidence=round(r.confidence, 4),
            interaction_type=r.interaction_type,
            relation=r.relation,
            is_unsafe=r.is_unsafe,
            is_contraindicated=r.is_contraindicated,
            **dict(zip(
                ["explanation", "hdc_detail", "action_items", "confidence_label"],
                _build_descriptors(r),
            )),
        )
        for r in results
    ]


@app.get("/api/drugs", response_model=list[str])
async def list_drugs():
    """List all known drug names (generic + brand names)."""
    if checker is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    return checker.list_known_drugs()


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Return knowledge base statistics."""
    if checker is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    stats = checker.get_stats()
    return StatsResponse(**stats)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
