"""
AXIOM Encoder — Biomedical NER Pipeline

Extracts medical entities and infers relations from raw text,
producing MedicalFact triples ready for the Distiller.

Pipeline:
    Raw text  →  scispaCy NER  →  Entity pairs  →  Relation inference  →  MedicalFact[]

Uses scispaCy for biomedical Named Entity Recognition.
Relation extraction employs a lightweight co-occurrence + pattern-matching
heuristic (no external API calls — fully offline).

Security note:
    All text is sanitised before processing. No user input is ever
    passed to eval() or shell commands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Generator

from src.config import NERConfig, ner as default_ner_cfg
from src.distiller import MedicalFact

logger = logging.getLogger("axiom.encoder")

# ---------------------------------------------------------------------------
# Relation pattern matchers (offline, rule-based)
# ---------------------------------------------------------------------------

# Compiled once at module level for performance
_RELATION_PATTERNS: dict[str, re.Pattern] = {
    "TREATS": re.compile(
        r"\b(treat(?:s|ed|ment)?|therap(?:y|ies|eutic)|prescri(?:be|bed))\b",
        re.IGNORECASE,
    ),
    "CAUSES": re.compile(
        r"\b(caus(?:e[sd]?|ing)|induc(?:e[sd]?|ing)|lead(?:s)?\s+to)\b",
        re.IGNORECASE,
    ),
    "PREVENTS": re.compile(
        r"\b(prevent(?:s|ed|ion)?|prophyla(?:xis|ctic)|protect(?:s|ed)?)\b",
        re.IGNORECASE,
    ),
    "INDICATES": re.compile(
        r"\b(indicat(?:e[sd]?|ion|ing)|suggest(?:s|ed)?|marker)\b",
        re.IGNORECASE,
    ),
    "CONTRAINDICATES": re.compile(
        r"\b(contraindic(?:at(?:e[sd]?|ion|ing))?|incompatib(?:le|ility))\b",
        re.IGNORECASE,
    ),
    "INTERACTS_WITH": re.compile(
        r"\b(interact(?:s|ion|ing)?|synerg(?:y|istic)|antagoni(?:s(?:m|tic)|ze))\b",
        re.IGNORECASE,
    ),
    "REGULATES": re.compile(
        r"\b(regulat(?:e[sd]?|ion|ing)|modulat(?:e[sd]?|ion|ing))\b",
        re.IGNORECASE,
    ),
    "INHIBITS": re.compile(
        r"\b(inhibit(?:s|ed|ion|ing|or)?|block(?:s|ed|ing|er)?|suppress(?:es|ed)?)\b",
        re.IGNORECASE,
    ),
    "ACTIVATES": re.compile(
        r"\b(activat(?:e[sd]?|ion|ing|or)|stimulat(?:e[sd]?|ion|ing))\b",
        re.IGNORECASE,
    ),
    "METABOLISES": re.compile(
        r"\b(metaboli[sz](?:e[sd]?|ing|m)|catabolism|biotransform)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Text sanitiser
# ---------------------------------------------------------------------------

def _sanitise(text: str) -> str:
    """
    Strip potentially dangerous or malformed content from raw input.

    - Removes control characters except newlines/tabs
    - Collapses excessive whitespace
    - No truncation (the NER model handles length)
    """
    # Remove control chars (keep \n, \t)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class AxiomEncoder:
    """
    Extracts MedicalFact triples from raw biomedical text.

    Usage:
        encoder = AxiomEncoder()
        for fact in encoder.extract("Aspirin treats headaches and thins blood."):
            print(fact)
        # => MedicalFact(subject='Aspirin', relation='TREATS', obj='headaches')
        # => MedicalFact(subject='Aspirin', relation='TREATS', obj='blood')
    """

    cfg: NERConfig = field(default_factory=lambda: default_ner_cfg)
    _nlp: object = field(default=None, repr=False)

    def _load_model(self) -> None:
        """Lazy-load spaCy model on first use."""
        if self._nlp is not None:
            return
        try:
            import spacy
            self._nlp = spacy.load(self.cfg.spacy_model)
            logger.info("Loaded spaCy model: %s", self.cfg.spacy_model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. "
                "Install with: python -m spacy download %s  or  "
                "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                "releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz",
                self.cfg.spacy_model,
                self.cfg.spacy_model,
            )
            raise

    def extract(self, text: str) -> Generator[MedicalFact, None, None]:
        """
        Extract medical fact triples from a block of text.

        Strategy:
            1. Run scispaCy NER to find biomedical entities.
            2. For every pair of entities in the same sentence,
               scan the intervening text for relation patterns.
            3. Yield a MedicalFact for each (entity, relation, entity) match.
        """
        self._load_model()
        text = _sanitise(text)

        doc = self._nlp(text)

        for sent in doc.sents:
            # Collect entities in this sentence
            entities = [ent for ent in sent.ents if ent.text.strip()]

            if len(entities) < 2:
                continue

            sent_text = sent.text

            # Check every ordered pair
            for ent_a, ent_b in combinations(entities, 2):
                # Get text between entities for relation detection
                start = min(ent_a.end_char, ent_b.end_char) - sent.start_char
                end = max(ent_a.start_char, ent_b.start_char) - sent.start_char
                between = sent_text[start:end] if end > start else sent_text

                for rel_label, pattern in _RELATION_PATTERNS.items():
                    if pattern.search(between) or pattern.search(sent_text):
                        yield MedicalFact(
                            subject=ent_a.text.strip(),
                            relation=rel_label,
                            obj=ent_b.text.strip(),
                        )

    def extract_batch(
        self, texts: list[str], batch_size: int = 256
    ) -> Generator[MedicalFact, None, None]:
        """
        Batch-extract facts from multiple texts using spaCy's pipe() for speed.
        """
        self._load_model()
        clean_texts = [_sanitise(t) for t in texts]

        for doc in self._nlp.pipe(clean_texts, batch_size=batch_size):
            for sent in doc.sents:
                entities = [ent for ent in sent.ents if ent.text.strip()]
                if len(entities) < 2:
                    continue

                sent_text = sent.text

                for ent_a, ent_b in combinations(entities, 2):
                    start = min(ent_a.end_char, ent_b.end_char) - \
                        sent.start_char
                    end = max(ent_a.start_char, ent_b.start_char) - \
                        sent.start_char
                    between = sent_text[start:end] if end > start else sent_text

                    for rel_label, pattern in _RELATION_PATTERNS.items():
                        if pattern.search(between) or pattern.search(sent_text):
                            yield MedicalFact(
                                subject=ent_a.text.strip(),
                                relation=rel_label,
                                obj=ent_b.text.strip(),
                            )

    def extract_from_structured(
        self,
        triples: list[tuple[str, str, str]],
    ) -> Generator[MedicalFact, None, None]:
        """
        Convert pre-extracted (subject, relation, object) tuples into
        MedicalFact instances.  Useful for datasets that already provide
        structured annotations (e.g., BioASQ, PubMed Knowledge Graph).
        """
        for subj, rel, obj in triples:
            subj = _sanitise(subj)
            rel = _sanitise(rel).upper().replace(" ", "_")
            obj = _sanitise(obj)
            if subj and rel and obj:
                yield MedicalFact(subject=subj, relation=rel, obj=obj)
