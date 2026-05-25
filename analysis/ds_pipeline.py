"""
Data Science Pipeline for Code Similarity and Plagiarism Risk

This script mirrors the core DS logic used in the backend service, packaged as a
standalone, reproducible pipeline you can include in your report or run locally.

Key steps:
- Preprocess code (normalize whitespace; optional comment removal)
- Compute multi-view similarity metrics between pairs of files:
  * Levenshtein-based lexical similarity
  * SequenceMatcher (structural similarity)
  * TF-IDF + cosine (semantic similarity)
  * Jaccard (token overlap)
- Aggregate into a weighted similarity score and classify risk
- Produce a JSON report summarizing comparisons

Dependencies (already in backend/requirements.txt):
  python-Levenshtein, scikit-learn, numpy

Author: Your Name
Date: 2025-11-04
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json
import difflib
import Levenshtein  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------- Preprocessing ----------------------------- #

def normalize_code(text: str, remove_comments: bool = False) -> str:
    """Normalize code to stabilize string-based metrics.

    - Collapse repeated whitespace (spaces/newlines)
    - Optionally remove Python line comments beginning with '#'
    """
    if not isinstance(text, str):
        return ""

    if remove_comments:
        # Simple line comment removal; keeps docstrings and code intact
        lines = []
        for line in text.splitlines():
            # Keep '#' inside strings by splitting only when not in quotes is complex.
            # For robustness and speed, apply a simple heuristic: strip trailing comments.
            # You may replace with tokenize-based removal for strict accuracy.
            if '#' in line:
                hash_idx = line.find('#')
                # retain content before '#'
                line = line[:hash_idx]
            lines.append(line)
        text = "\n".join(lines)

    # Normalize whitespace (collapse runs)
    collapsed = " ".join(text.split())
    return collapsed.strip()


# --------------------------- Similarity Metrics -------------------------- #

def lexical_similarity(code1: str, code2: str) -> float:
    """Levenshtein-based similarity in [0,1]."""
    if not code1 and not code2:
        return 1.0
    if not code1 or not code2:
        return 0.0
    distance = Levenshtein.distance(code1, code2)
    denom = max(len(code1), len(code2))
    return max(0.0, 1.0 - distance / denom) if denom else 0.0


def sequence_similarity(code1: str, code2: str) -> float:
    """Structural similarity via difflib.SequenceMatcher ratio in [0,1]."""
    if not code1 and not code2:
        return 1.0
    return difflib.SequenceMatcher(None, code1, code2).ratio()


def tfidf_cosine_similarity(code1: str, code2: str) -> float:
    """TF-IDF + cosine similarity in [0,1]. Robust to token frequency differences."""
    try:
        vec = TfidfVectorizer(token_pattern=r"\b\w+\b", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform([code1, code2])
        return float(cosine_similarity(X[0:1], X[1:2])[0, 0])
    except Exception:
        # Extremely short or degenerate inputs may fail vectorization
        return 0.0


def jaccard_similarity(code1: str, code2: str) -> float:
    """Token set overlap in [0,1]. Order-insensitive."""
    s1 = set(code1.split())
    s2 = set(code2.split())
    union = s1 | s2
    if not union:
        return 0.0
    return len(s1 & s2) / len(union)


@dataclass
class SimilarityMetrics:
    similarity: float
    sim_lex: float
    seqmatch: float
    cosine: float
    jaccard: float
    sim_ast: float  # alias of seqmatch for compatibility with backend/UI


def aggregate_similarity(code1: str, code2: str) -> SimilarityMetrics:
    """Compute all metrics and aggregate with backend-aligned weights.

    similarity = 0.2*lex + 0.3*seq + 0.3*cos + 0.2*jac
    """
    c1 = normalize_code(code1)
    c2 = normalize_code(code2)

    lex = lexical_similarity(c1, c2)
    seq = sequence_similarity(c1, c2)
    cos = tfidf_cosine_similarity(c1, c2)
    jac = jaccard_similarity(c1, c2)

    similarity = 0.2 * lex + 0.3 * seq + 0.3 * cos + 0.2 * jac
    return SimilarityMetrics(
        similarity=similarity,
        sim_lex=lex,
        seqmatch=seq,
        cosine=cos,
        jaccard=jac,
        sim_ast=seq,  # UI expects sim_ast
    )


def risk_level(similarity: float) -> str:
    if similarity >= 0.80:
        return "high"
    if similarity >= 0.50:
        return "medium"
    return "low"


# --------------------------- Report Generation -------------------------- #

@dataclass
class FileRecord:
    student_id: str
    filename: str
    content: str


def load_py_files(folder: Path) -> List[FileRecord]:
    """Load all .py files in folder as FileRecord objects.
    Uses the stem (name without extension) as the student_id fallback.
    """
    records: List[FileRecord] = []
    for p in sorted(folder.glob("*.py")):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        sid = p.stem  # customize if you have a specific naming convention
        records.append(FileRecord(student_id=sid, filename=p.name, content=content))
    return records


def compare_pairs(files: List[FileRecord]) -> Dict:
    """Compare every pair and produce a report dict compatible with backend shape."""
    comparisons: List[Dict] = []
    high = medium = low = 0

    for a, b in combinations(files, 2):
        metrics = aggregate_similarity(a.content, b.content)
        rl = risk_level(metrics.similarity)
        if rl == "high":
            high += 1
        elif rl == "medium":
            medium += 1
        else:
            low += 1
        comparisons.append({
            "student1_id": a.student_id,
            "student1_file": a.filename,
            "student2_id": b.student_id,
            "student2_file": b.filename,
            "similarity_score": round(metrics.similarity, 4),
            "risk_level": rl,
            "sim_lex": round(metrics.sim_lex, 4),
            "sim_ast": round(metrics.sim_ast, 4),
            "jaccard": round(metrics.jaccard, 4),
            "seqmatch": round(metrics.seqmatch, 4),
        })

    report = {
        "assignment_name": "(local_folder_run)",
        "total_students": len(files),
        "total_comparisons": len(comparisons),
        "comparisons": comparisons,
        "plagiarism_detected": high > 0,
        "high_risk_count": high,
        "medium_risk_count": medium,
        "low_risk_count": low,
    }
    return report


def save_report(report: Dict, path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ------------------------------- CLI entry ------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code similarity DS pipeline")
    parser.add_argument(
        "--folder",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "submissions"),
        help="Folder containing .py submissions (default: repo /submissions)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="report.local.json",
        help="Output JSON report filename",
    )
    parser.add_argument(
        "--remove-comments",
        action="store_true",
        help="Optionally strip line comments before computing similarities",
    )

    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    # Load files
    files = load_py_files(folder)
    if not files or len(files) < 2:
        raise SystemExit("Need at least two .py files to compare.")

    if args.remove_comments:
        # re-normalize content with comment removal
        for fr in files:
            fr.content = normalize_code(fr.content, remove_comments=True)

    # Compare and save
    report = compare_pairs(files)
    out_path = Path(args.out)
    save_report(report, out_path)
    print(f"Saved DS report to: {out_path.resolve()}")
