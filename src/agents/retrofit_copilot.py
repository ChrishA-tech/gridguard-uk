"""Guardrailed LangGraph Retrofit Advisory Copilot with automated statutory
document ingestion (.pdf/.md/.txt), hybrid BM25 + TF-IDF retrieval over Ofgem
ECO4 / GBIS / PAS 2035 rules, and a citation-verifier judge node.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[([A-Z0-9-]+)\]")
HEADING_RULE_PATTERN = re.compile(r"^#{1,4}\s*(?:\[([A-Z0-9-]+)\]\s*)?(.*)$")

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(your\s+)?system\s+prompt",
    r"bypass\s+(compliance|ofgem|pas\s*2035|guardrail)",
    r"forge\s+(an?\s+)?epc",
    r"fake\s+(epc|certificate|lodgement)",
]


@dataclass(frozen=True)
class PolicyRule:
    """Single regulatory clause from Ofgem ECO4, GBIS, or BSI PAS 2035."""

    rule_id: str
    framework: Literal["ECO4", "GBIS", "PAS2035"]
    title: str
    text: str
    keywords: tuple[str, ...]
    eligible_epc_bands: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
    source_path: str = "builtin_statutory_corpus"


POLICY_CORPUS: tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="ECO4-ELIG-01",
        framework="ECO4",
        title="ECO4 Owner-Occupied & Private Rented EPC Band Thresholds",
        text=(
            "Under Ofgem ECO4 (2022-2026), owner-occupied homes must hold EPC band D, E, F, or G "
            "and meet low-income or vulnerable-household criteria (or LA Flex). Private rented "
            "sector homes are eligible in bands E, F, or G (or D for specified innovation/flex routes)."
        ),
        keywords=("eco4", "owner-occupied", "rented", "private", "d", "e", "f", "g", "income"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="ECO4-ELIG-02",
        framework="ECO4",
        title="ECO4 Social Housing & Minimum Band Improvement",
        text=(
            "Social housing properties in EPC bands E, F, or G qualify under ECO4 for fabric-first "
            "insulation and heating upgrades, subject to minimum SAP improvement targets (bands F/G "
            "to at least D; bands D/E to at least C) and TrustMark lodgement."
        ),
        keywords=("eco4", "social", "sap", "minimum", "improvement", "trustmark", "e", "f", "g"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="ECO4-FLEX-03",
        framework="ECO4",
        title="ECO4 Local Authority Flexible Eligibility (LA Flex)",
        text=(
            "Local Authorities may refer households under ECO4 Flex Route 1 (gross annual income "
            "under £31,000), Route 2 (IMD decile 1-3 proxy indicators plus health/vulnerability), "
            "or Route 3 (NHS health referral) for EPC D-G dwellings."
        ),
        keywords=("eco4", "flex", "imd", "31000", "income", "local authority", "deprivation"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="ECO4-TECH-04",
        framework="ECO4",
        title="ECO4 Fabric-First Solid Wall & Cavity Wall Insulation",
        text=(
            "Solid brick or uninsulated cavity wall dwellings must prioritise fabric insulation "
            "(external/internal solid wall insulation or cavity fill) before or alongside clean "
            "heating (air source heat pump) upgrades under ECO4 multi-measure packages."
        ),
        keywords=("solid brick", "cavity", "wall", "insulation", "heat pump", "fabric"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="GBIS-ELIG-01",
        framework="GBIS",
        title="Great British Insulation Scheme (GBIS) General & Low-Income Routes",
        text=(
            "Ofgem GBIS funds single insulation measures for EPC D-G homes in Council Tax bands A-D "
            "(England) or A-E (Scotland and Wales) under the general group, and low-income/means-tested "
            "households in EPC D-G regardless of Council Tax band."
        ),
        keywords=("gbis", "insulation", "council tax", "general", "low-income", "d", "e", "f", "g"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="GBIS-TECH-02",
        framework="GBIS",
        title="GBIS Loft Top-Up and Cavity Wall Single Measures",
        text=(
            "Under GBIS, properties with loft insulation below 100mm qualify for loft top-up to "
            "270-300mm, and properties with unfilled cavity walls qualify for cavity wall insulation "
            "with low-cost customer contribution."
        ),
        keywords=("gbis", "loft", "roof", "100mm", "270mm", "cavity wall", "unfilled"),
        eligible_epc_bands=("D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="PAS2035-VENT-01",
        framework="PAS2035",
        title="PAS 2035 Ventilation & Moisture Risk Assessment",
        text=(
            "BSI PAS 2035:2023 mandates checking existing background, purge, and extract ventilation "
            "before installing insulation measures so airtightness gains do not trigger condensation, "
            "damp, or mould; compliant trickle vents or MEV/dMEV must be specified where inadequate."
        ),
        keywords=("pas 2035", "ventilation", "moisture", "condensation", "damp", "insulation", "wall", "loft"),
        eligible_epc_bands=("A", "B", "C", "D", "E", "F", "G"),
    ),
    PolicyRule(
        rule_id="PAS2035-RISK-02",
        framework="PAS2035",
        title="PAS 2035 Retrofit Coordinator & Risk Path Classification",
        text=(
            "Every ECO4 and GBIS installation requires a TrustMark-registered Retrofit Coordinator "
            "to assign Risk Path A, B, or C based on building age (e.g., pre-1919 solid wall), "
            "exposure, and measure interactions, plus post-installation lodgement in TrustMark Data Warehouse."
        ),
        keywords=("pas 2035", "coordinator", "risk path", "pre-1919", "solid brick", "trustmark"),
        eligible_epc_bands=("A", "B", "C", "D", "E", "F", "G"),
    ),
)

DOMAIN_KEYWORDS = (
    "eco4",
    "gbis",
    "pas 2035",
    "ventilation",
    "moisture",
    "condensation",
    "damp",
    "solid",
    "cavity",
    "loft",
    "roof",
    "unfilled",
    "flex",
    "imd",
    "income",
    "social",
    "owner-occupied",
    "private",
    "trustmark",
    "heat pump",
    "insulation",
    "coordinator",
    "risk path",
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _infer_framework(text: str, filename: str) -> Literal["ECO4", "GBIS", "PAS2035"]:
    combined = f"{filename} {text}".lower()
    if "pas2035" in combined or "pas 2035" in combined or "ventilation" in combined or "risk path" in combined:
        return "PAS2035"
    if "gbis" in combined or "great british insulation" in combined:
        return "GBIS"
    return "ECO4"


def _extract_keywords(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    matched = [kw for kw in DOMAIN_KEYWORDS if kw in lowered]
    if not matched:
        tokens = _tokenize(text)
        matched = list(dict.fromkeys(tokens[:8]))
    return tuple(dict.fromkeys(matched))


def _infer_epc_bands(text: str) -> tuple[str, ...]:
    if re.search(r"\bepc\s*(bands?|d-g)\b", text.lower()) or "d, e, f, or g" in text.lower():
        return ("D", "E", "F", "G")
    return ("A", "B", "C", "D", "E", "F", "G")


def _extract_file_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    if suffix in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8")
    return ""


def _chunk_statutory_text(raw_text: str, file_path: Path) -> list[PolicyRule]:
    """Split raw statutory guidance text into semantic PolicyRule clause chunks."""
    chunks: list[tuple[str | None, str, str]] = []
    current_id: str | None = None
    current_title = file_path.stem.replace("_", " ").title()
    buffer: list[str] = []

    def _flush() -> None:
        body = "\n".join(buffer).strip()
        if len(body) >= 35:
            chunks.append((current_id, current_title, body))
        buffer.clear()

    for line in raw_text.splitlines():
        stripped = line.strip()
        heading_match = HEADING_RULE_PATTERN.match(stripped)
        if heading_match:
            # Skip top-level document H1 banner when it only introduces the file
            if stripped.startswith("# ") and not heading_match.group(1) and not buffer:
                continue
            _flush()
            current_id = heading_match.group(1)
            current_title = heading_match.group(2).strip() or current_title
        elif not stripped and sum(len(x) for x in buffer) >= 450:
            _flush()
        elif stripped:
            buffer.append(stripped)

    _flush()

    # Fallback paragraph chunking if document had no Markdown headings
    if not chunks and raw_text.strip():
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if len(p.strip()) >= 35]
        for idx, para in enumerate(paragraphs, start=1):
            inline_id_match = CITATION_PATTERN.search(para)
            rule_id = inline_id_match.group(1) if inline_id_match else None
            first_line = para.splitlines()[0][:80]
            chunks.append((rule_id, first_line or f"Clause {idx}", para))

    rules: list[PolicyRule] = []
    for idx, (explicit_id, title, body) in enumerate(chunks, start=1):
        framework = _infer_framework(f"{title} {body}", file_path.name)
        if explicit_id:
            rule_id = explicit_id
        else:
            stem_tag = re.sub(r"[^A-Z0-9]+", "", file_path.stem.upper())[:6] or "DOC"
            rule_id = f"{framework}-{stem_tag}-{idx:02d}"
        rules.append(
            PolicyRule(
                rule_id=rule_id,
                framework=framework,
                title=title,
                text=body,
                keywords=_extract_keywords(f"{title} {body}"),
                eligible_epc_bands=_infer_epc_bands(body),
                source_path=str(file_path),
            )
        )
    return rules


def load_regulatory_corpus(docs_dir: Path | str = Path("data/regulatory_docs")) -> tuple[PolicyRule, ...]:
    """Load and chunk real .pdf / .md / .txt statutory guidance documents from ``docs_dir``,
    falling back to built-in ``POLICY_CORPUS`` if no files are present or parseable.
    """
    target_dir = Path(docs_dir)
    if not target_dir.exists() or not target_dir.is_dir():
        return POLICY_CORPUS

    doc_files = sorted(
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".md", ".txt"}
    )
    if not doc_files:
        return POLICY_CORPUS

    parsed_rules: list[PolicyRule] = []
    seen_ids: set[str] = set()
    for doc_file in doc_files:
        try:
            text = _extract_file_text(doc_file)
            for rule in _chunk_statutory_text(text, doc_file):
                unique_id = rule.rule_id
                if unique_id in seen_ids:
                    unique_id = f"{rule.rule_id}-{len(seen_ids) + 1}"
                    rule = PolicyRule(
                        rule_id=unique_id,
                        framework=rule.framework,
                        title=rule.title,
                        text=rule.text,
                        keywords=rule.keywords,
                        eligible_epc_bands=rule.eligible_epc_bands,
                        source_path=rule.source_path,
                    )
                seen_ids.add(unique_id)
                parsed_rules.append(rule)
        except Exception as exc:
            logger.warning("Failed to parse statutory guidance '%s': %s", doc_file, exc)

    if not parsed_rules:
        return POLICY_CORPUS
    return tuple(parsed_rules)


class HybridPolicyRetriever:
    """Hybrid BM25 + TF-IDF cosine + regulatory metadata retriever over Ofgem ECO4,
    GBIS, and PAS 2035 statutory guidance chunks.
    """

    def __init__(
        self,
        corpus: tuple[PolicyRule, ...] | None = None,
        docs_dir: Path | str = Path("data/regulatory_docs"),
    ) -> None:
        self.corpus = corpus if corpus is not None else load_regulatory_corpus(docs_dir=docs_dir)
        self._rule_map: dict[str, PolicyRule] = {r.rule_id: r for r in self.corpus}

        raw_texts = [
            f"{r.rule_id} {r.framework} {r.title} {r.text} {' '.join(r.keywords)}"
            for r in self.corpus
        ]
        docs_tokens = [_tokenize(txt) for txt in raw_texts]
        self._bm25 = BM25Okapi(docs_tokens)
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(raw_texts)

    def get_rule(self, rule_id: str) -> PolicyRule | None:
        return self._rule_map.get(rule_id)

    def retrieve(
        self,
        query: str,
        property_profile: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top-k policy rules combining BM25, TF-IDF cosine, and metadata boosts."""
        profile = property_profile or {}
        epc_band = str(profile.get("current_energy_rating", "")).upper().strip()
        wall_type = str(profile.get("wall_type", "")).lower()
        tenure = str(profile.get("tenure", "")).lower()
        roof_mm = float(profile.get("roof_insulation_mm", 150.0) or 150.0)
        imd_decile = int(profile.get("imd_decile", 5) or 5)
        income = float(profile.get("household_income_gbp", 35000.0) or 35000.0)

        enriched_query = " ".join(
            [
                query,
                epc_band,
                wall_type,
                tenure,
                "loft roof" if roof_mm < 150 else "",
                "imd flex low income" if (imd_decile <= 3 or income < 31000) else "",
                "pas 2035 ventilation risk path",
            ]
        )
        tokens = _tokenize(enriched_query)
        bm25_scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        max_bm25 = max(float(np.max(bm25_scores)), 1e-6)

        q_vec = self._tfidf.transform([enriched_query])
        tfidf_scores = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        max_tfidf = max(float(np.max(tfidf_scores)), 1e-6)

        scored: list[tuple[float, PolicyRule]] = []
        for idx, rule in enumerate(self.corpus):
            bm25_norm = float(bm25_scores[idx]) / max_bm25
            tfidf_norm = float(tfidf_scores[idx]) / max_tfidf

            meta_boost = 0.0
            if epc_band and epc_band in rule.eligible_epc_bands:
                meta_boost += 0.20
            if rule.framework == "PAS2035":
                meta_boost += 0.25
            if "solid" in wall_type and "solid" in " ".join(rule.keywords):
                meta_boost += 0.25
            if "unfilled" in wall_type and "cavity" in " ".join(rule.keywords):
                meta_boost += 0.25
            if roof_mm < 150 and "loft" in " ".join(rule.keywords):
                meta_boost += 0.25
            if (imd_decile <= 3 or income < 31000) and "flex" in " ".join(rule.keywords):
                meta_boost += 0.25
            if "social" in tenure and "social" in " ".join(rule.keywords):
                meta_boost += 0.25

            total_score = round(
                0.45 * bm25_norm + 0.30 * tfidf_norm + 0.25 * min(meta_boost, 1.0),
                4,
            )
            scored.append((total_score, rule))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "rule_id": rule.rule_id,
                "framework": rule.framework,
                "title": rule.title,
                "text": rule.text,
                "source_path": rule.source_path,
                "score": score,
            }
            for score, rule in scored[:top_k]
        ]


def verify_citations_against_rules(
    draft_text: str,
    retrieved_rules: list[dict[str, Any]],
) -> tuple[bool, list[str], str]:
    """Judge logic verifying that every inline ``[RULE-ID]`` in ``draft_text``
    is grounded in ``retrieved_rules`` and at least one valid citation is used.
    """
    valid_ids = {r["rule_id"] for r in retrieved_rules}
    found_ids = list(dict.fromkeys(CITATION_PATTERN.findall(draft_text)))

    if not found_ids:
        return False, [], "No inline regulatory citations found in draft advice."

    unknown_ids = [cid for cid in found_ids if cid not in valid_ids]
    if unknown_ids:
        return (
            False,
            found_ids,
            f"Unverified or hallucinated citations detected: {unknown_ids}",
        )

    mentions_insulation = any(
        term in draft_text.lower() for term in ("wall insulation", "loft", "cavity", "solid wall")
    )
    has_pas = any(cid.startswith("PAS2035") for cid in found_ids)
    if mentions_insulation and not has_pas and any(rid.startswith("PAS2035") for rid in valid_ids):
        return (
            False,
            found_ids,
            "Insulation recommended without citing mandatory PAS 2035 ventilation/risk rule.",
        )

    return True, found_ids, "All inline citations verified against retrieved Ofgem/PAS 2035 rules."


class CopilotState(TypedDict, total=False):
    """State passed across LangGraph nodes."""

    query: str
    property_profile: dict[str, Any]
    guardrail_passed: bool
    guardrail_reason: str
    retrieved_rules: list[dict[str, Any]]
    draft_advice: str
    draft_citations: list[str]
    recommended_measures: list[str]
    eligible_schemes: list[str]
    citation_verified: bool
    verification_notes: str
    retry_count: int
    final_response: dict[str, Any]


def build_retrofit_copilot_graph(
    retriever: HybridPolicyRetriever | None = None,
    force_unverified_first_draft: bool = False,
):
    """Compile and return the guardrailed LangGraph retrofit advisory workflow."""
    policy_retriever = retriever or HybridPolicyRetriever()

    def input_guardrail(state: CopilotState) -> CopilotState:
        query = state.get("query", "")
        profile = state.get("property_profile", {}) or {}
        combined = f"{query} {profile}".lower()

        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, combined):
                return {
                    "guardrail_passed": False,
                    "guardrail_reason": (
                        "Request blocked by regulatory compliance guardrail: cannot bypass Ofgem "
                        "eligibility checks, forge EPC certificates, or override system rules."
                    ),
                }

        epc = str(profile.get("current_energy_rating", "")).upper().strip()
        if epc and epc not in {"A", "B", "C", "D", "E", "F", "G"}:
            return {
                "guardrail_passed": False,
                "guardrail_reason": f"Invalid EPC energy rating '{epc}'. Expected band A through G.",
            }

        return {"guardrail_passed": True, "guardrail_reason": "Passed input compliance checks."}

    def hybrid_retrieve(state: CopilotState) -> CopilotState:
        rules = policy_retriever.retrieve(
            query=state.get("query", "retrofit eligibility and PAS 2035 measures"),
            property_profile=state.get("property_profile", {}),
            top_k=5,
        )
        return {"retrieved_rules": rules, "retry_count": state.get("retry_count", 0)}

    def synthesize_advice(state: CopilotState) -> CopilotState:
        profile = state.get("property_profile", {}) or {}
        rules = state.get("retrieved_rules", [])
        rule_ids = {r["rule_id"] for r in rules}
        retry_count = int(state.get("retry_count", 0))

        if force_unverified_first_draft and retry_count == 0:
            return {
                "draft_advice": "Install external wall insulation under [ECO4-FAKE-999].",
                "recommended_measures": ["External solid wall insulation"],
                "eligible_schemes": ["ECO4"],
                "retry_count": retry_count + 1,
            }

        epc = str(profile.get("current_energy_rating", "D")).upper()
        tenure = str(profile.get("tenure", "Owner-occupied"))
        wall_type = str(profile.get("wall_type", "Cavity wall (unfilled)"))
        roof_mm = float(profile.get("roof_insulation_mm", 100.0) or 100.0)
        heat_fuel = str(profile.get("main_heat_fuel", "Mains gas"))
        imd_decile = int(profile.get("imd_decile", 5) or 5)
        income = float(profile.get("household_income_gbp", 28000.0) or 28000.0)

        eligible_schemes: list[str] = []
        measures: list[str] = []
        bullets: list[str] = []

        if epc in {"D", "E", "F", "G"}:
            if "social" in tenure.lower() and epc in {"E", "F", "G"}:
                eligible_schemes.append("ECO4 Social Housing")
                if "ECO4-ELIG-02" in rule_ids:
                    bullets.append(
                        f"- **Scheme Eligibility (ECO4 Social Housing):** Band {epc} social dwelling qualifies "
                        "for multi-measure upgrades targeting minimum SAP band C/D [ECO4-ELIG-02]."
                    )
            else:
                if "ECO4-ELIG-01" in rule_ids and (income < 31000 or imd_decile <= 3 or epc in {"E", "F", "G"}):
                    eligible_schemes.append("ECO4")
                    bullets.append(
                        f"- **Scheme Eligibility (ECO4):** Band {epc} ({tenure}) meets Ofgem ECO4 EPC "
                        "band criteria [ECO4-ELIG-01]."
                    )
                if "ECO4-FLEX-03" in rule_ids and (income < 31000 or imd_decile <= 3):
                    if "ECO4 Flex (LA Flex)" not in eligible_schemes:
                        eligible_schemes.append("ECO4 Flex (LA Flex)")
                    bullets.append(
                        f"- **LA Flex Route:** Household income (£{income:,.0f}) / IMD decile ({imd_decile}) "
                        "supports Local Authority Flexible Eligibility referral [ECO4-FLEX-03]."
                    )
            if "GBIS-ELIG-01" in rule_ids:
                eligible_schemes.append("GBIS")
                bullets.append(
                    f"- **Great British Insulation Scheme (GBIS):** Band {epc} dwelling is eligible for a "
                    "funded single insulation measure under GBIS general or low-income routes [GBIS-ELIG-01]."
                )
        else:
            bullets.append(
                f"- **EPC Band {epc} Status:** Property already meets Band A-C efficiency thresholds; "
                "prioritise targeted maintenance and ventilation compliance rather than grant-funded deep fabric retrofit."
            )

        if "unfilled" in wall_type.lower() or "solid" in wall_type.lower():
            measure = (
                "External/Internal Solid Wall Insulation"
                if "solid" in wall_type.lower()
                else "Cavity Wall Insulation"
            )
            measures.append(measure)
            wall_cite = "ECO4-TECH-04" if "ECO4-TECH-04" in rule_ids else (
                "GBIS-TECH-02" if "GBIS-TECH-02" in rule_ids else next(iter(rule_ids))
            )
            bullets.append(
                f"- **Primary Wall Measure:** `{wall_type}` requires **{measure}** as a fabric-first "
                f"priority [{wall_cite}]."
            )

        if roof_mm < 150.0:
            measures.append("Loft Insulation Top-Up to 270-300mm")
            loft_cite = "GBIS-TECH-02" if "GBIS-TECH-02" in rule_ids else next(iter(rule_ids))
            bullets.append(
                f"- **Roof / Loft Top-Up:** Current loft insulation ({roof_mm:.0f}mm) is below 150mm; "
                f"top up to 270-300mm [{loft_cite}]."
            )

        if heat_fuel in {"Oil", "LPG", "Solid fuel", "Electricity"} and epc in {"D", "E", "F", "G"}:
            measures.append("Air Source Heat Pump (post-fabric upgrade)")
            if "ECO4-TECH-04" in rule_ids:
                bullets.append(
                    f"- **Clean Heating Transition:** Replace `{heat_fuel}` heating with an Air Source Heat Pump "
                    "once fabric insulation demand is reduced [ECO4-TECH-04]."
                )

        if "PAS2035-VENT-01" in rule_ids:
            bullets.append(
                "- **Mandatory PAS 2035 Ventilation Check:** Assess background/extract ventilation and install "
                "trickle vents or MEV/dMEV alongside insulation to prevent damp/mould [PAS2035-VENT-01]."
            )
        if "PAS2035-RISK-02" in rule_ids:
            bullets.append(
                "- **PAS 2035 Governance:** Appoint a TrustMark Retrofit Coordinator to assign Risk Path "
                "and complete post-install TrustMark lodgement [PAS2035-RISK-02]."
            )

        if not measures:
            measures.append("Smart Heating Controls & Draught-Proofing")

        if not CITATION_PATTERN.findall("\n".join(bullets)) and rules:
            fallback_id = rules[0]["rule_id"]
            bullets.append(f"- **Regulatory Reference:** Assessed under [{fallback_id}].")

        draft_advice = (
            f"### GridGuard UK Retrofit Assessment (EPC Band {epc}, {tenure})\n\n"
            + "\n".join(bullets)
        )
        return {
            "draft_advice": draft_advice,
            "recommended_measures": measures,
            "eligible_schemes": eligible_schemes,
            "retry_count": retry_count + 1,
        }

    def citation_verifier_judge(state: CopilotState) -> CopilotState:
        verified, found_ids, notes = verify_citations_against_rules(
            state.get("draft_advice", ""),
            state.get("retrieved_rules", []),
        )
        return {
            "citation_verified": verified,
            "draft_citations": found_ids,
            "verification_notes": notes,
        }

    def finalize_response(state: CopilotState) -> CopilotState:
        verified = bool(state.get("citation_verified", False))
        rules_by_id = {r["rule_id"]: r for r in state.get("retrieved_rules", [])}
        citations = [
            {
                "rule_id": cid,
                "framework": rules_by_id[cid]["framework"],
                "title": rules_by_id[cid]["title"],
            }
            for cid in state.get("draft_citations", [])
            if cid in rules_by_id
        ]

        advisory = state.get("draft_advice", "")
        if not verified:
            advisory = (
                "Unable to verify regulatory citations for the generated retrofit plan. "
                "Please consult a TrustMark Retrofit Coordinator directly."
            )

        return {
            "final_response": {
                "guardrail_passed": True,
                "citation_verified": verified,
                "verification_notes": state.get("verification_notes", ""),
                "eligible_schemes": state.get("eligible_schemes", []),
                "recommended_measures": state.get("recommended_measures", []),
                "citations": citations,
                "advisory_markdown": advisory,
            }
        }

    def refusal_node(state: CopilotState) -> CopilotState:
        reason = state.get("guardrail_reason", "Blocked by compliance guardrail.")
        return {
            "citation_verified": False,
            "final_response": {
                "guardrail_passed": False,
                "citation_verified": False,
                "verification_notes": reason,
                "eligible_schemes": [],
                "recommended_measures": [],
                "citations": [],
                "advisory_markdown": f"**Request Declined:** {reason}",
            },
        }

    def route_after_guardrail(state: CopilotState) -> str:
        return "hybrid_retrieve" if state.get("guardrail_passed", False) else "refusal_node"

    def route_after_judge(state: CopilotState) -> str:
        if state.get("citation_verified", False):
            return "finalize_response"
        if int(state.get("retry_count", 0)) <= 1:
            return "synthesize_advice"
        return "finalize_response"

    graph = StateGraph(CopilotState)
    graph.add_node("input_guardrail", input_guardrail)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("synthesize_advice", synthesize_advice)
    graph.add_node("citation_verifier_judge", citation_verifier_judge)
    graph.add_node("finalize_response", finalize_response)
    graph.add_node("refusal_node", refusal_node)

    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail",
        route_after_guardrail,
        {"hybrid_retrieve": "hybrid_retrieve", "refusal_node": "refusal_node"},
    )
    graph.add_edge("hybrid_retrieve", "synthesize_advice")
    graph.add_edge("synthesize_advice", "citation_verifier_judge")
    graph.add_conditional_edges(
        "citation_verifier_judge",
        route_after_judge,
        {"synthesize_advice": "synthesize_advice", "finalize_response": "finalize_response"},
    )
    graph.add_edge("finalize_response", END)
    graph.add_edge("refusal_node", END)

    return graph.compile()


def run_retrofit_copilot(
    query: str,
    property_profile: dict[str, Any],
    force_unverified_first_draft: bool = False,
    docs_dir: Path | str = Path("data/regulatory_docs"),
) -> dict[str, Any]:
    """Execute the guardrailed LangGraph retrofit copilot and return its structured response."""
    retriever = HybridPolicyRetriever(docs_dir=docs_dir)
    app = build_retrofit_copilot_graph(
        retriever=retriever,
        force_unverified_first_draft=force_unverified_first_draft,
    )
    result = app.invoke(
        {
            "query": query,
            "property_profile": property_profile,
            "retry_count": 0,
        }
    )
    return result["final_response"]
