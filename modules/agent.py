"""Automated validation agent: quality rules + CNN agreement + duplicate checks."""
from __future__ import annotations
import hashlib
import numpy as np

def sample_fingerprint(image: np.ndarray) -> str:
    # Stable exact fingerprint of the refined 28x28 sample.
    return hashlib.sha256(np.asarray(image, dtype=np.uint8).tobytes()).hexdigest()

def quality_findings(image: np.ndarray, diagnostics: dict) -> list[str]:
    findings = []
    if image is None:
        return ["Blank canvas: no handwriting was submitted."]
    ink = np.asarray(image) > 18
    count = int(ink.sum())
    if count < 12:
        findings.append("Too little visible ink: the digit may be incomplete.")
    if count > 600:
        findings.append("Very dense drawing: possible overwriting or multiple strokes.")
    if diagnostics.get("touches_edge"):
        findings.append("Ink touches the canvas boundary: the drawing may be clipped.")
    # Components are advisory only: disconnected strokes can be valid handwriting.
    try:
        import cv2
        n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype("uint8"), 8)
        meaningful = sum(1 for s in stats[1:] if s[cv2.CC_STAT_AREA] >= 5)
        if meaningful >= 4:
            findings.append("Several separated ink regions detected; check for accidental extra marks.")
    except Exception:
        pass
    return findings

def verify_sample(image, diagnostics, intended, predicted, confidence, prior_fingerprints):
    findings = quality_findings(image, diagnostics)
    fingerprint = sample_fingerprint(image)
    if fingerprint in prior_fingerprints:
        findings.append("Duplicate drawing: this same refined sample was already submitted in this session.")
    if predicted != intended:
        findings.append(f"Label mismatch: you selected {intended}, but the model predicted {predicted}.")
    if confidence < 0.70:
        findings.append(f"Low model confidence ({confidence:.1%}); the agent cannot verify this sample reliably.")
    # Conservative automated policy: accept only if clean, unique, confident and label agrees.
    accepted = len(findings) == 0 and predicted == intended and confidence >= 0.70
    return {
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "findings": findings,
        "fingerprint": fingerprint,
        "reason": "All automated checks passed." if accepted else "One or more automated checks did not pass."
    }
