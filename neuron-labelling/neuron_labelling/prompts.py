"""Prompt text for the investigating and verifying agents."""

from __future__ import annotations

import json
from typing import Any

from neuron_labelling.examples import PoleEvidence

INVESTIGATOR_SYSTEM = """\
You are an interpretability scientist studying one MLP neuron of a language model \
(Qwen3, SwiGLU MLP). Your job is to find out what the neuron does and write an \
explanation specific enough that it could be proven wrong.

How to read the evidence:
- The neuron has two poles. The POSITIVE pole is where it fires strongly positive, \
the NEGATIVE pole where it fires strongly negative. Explain each pole separately; \
the negative pole is often a related variant, a contrast, or the moment a context ends.
- Activations are only counted when the neuron's gate is open. Gate-closed "leak" \
firings are a small sign-flipped echo of the other pole; they are listed separately \
and are NOT evidence of the pole's job.
- In examples, the firing token is marked [[like this]]⟨n⟩ and other tokens in the \
window that fire in the same direction are marked token⟨n⟩, with n from 0 to 10 \
relative to the pole's strongest firing in the corpus.
- Neurons often do not fire on the word that carries the meaning. Common patterns: \
firing on punctuation (commas, full stops) where the model summarises the context so \
far; firing just before a word the model is about to predict; firing a few tokens \
after the topic. Read the whole window, including the text after the marked token.
- Strong firings matter most. Weaker firings usually broaden into related territory \
("offshoots"); note them but do not let them dilute the main explanation.
- Logit effects show which next tokens the pole pushes up or down directly. Steering \
shows what the model writes when the pole is forced on.

Categories for each pole:
- "one_job": one main function, possibly with small related offshoots.
- "two_related": two related functions.
- "several": three or more functions.
- "unclear": you cannot yet find a function that explains the strong firings.

Always reply with a single JSON object and nothing else."""

TEST_REQUEST = """\
Form a hypothesis for each pole and design tests that could falsify it. Write \
short, self-contained texts (one to three sentences). "should_fire" texts should \
make the pole fire strongly if the hypothesis is right; "near_misses" should be as \
similar as possible but should NOT fire. Reply as:
{"positive": {"hypothesis": "...", "should_fire": ["...", "...", "...", "..."], \
"near_misses": ["...", "...", "...", "..."]},
 "negative": {"hypothesis": "...", "should_fire": [...], "near_misses": [...]}}"""

RETEST_REQUEST = """\
Revise your hypotheses in light of these results and design new tests that probe \
whatever is still uncertain. Reply in the same JSON format."""

FINAL_REQUEST = """\
Write your final explanation of each pole. Lead with the main job in one sentence; \
mention offshoots separately. Reply as:
{"positive": {"explanation": "...", "category": "one_job|two_related|several|unclear", \
"offshoots": ["..."], "confidence": "low|medium|high"},
 "negative": {"explanation": "...", "category": "...", "offshoots": [...], "confidence": "..."},
 "summary": "one sentence describing the neuron as a whole"}"""

VERIFIER_SYSTEM = """\
You test explanations of language-model neurons. You will be given only an \
explanation of what makes a neuron fire. You never see the neuron itself.
Always reply with a single JSON object and nothing else."""

VERIFIER_REQUEST = """\
Explanation of what makes this neuron fire:
{explanation}

Write:
- "examples": 5 new, varied texts (one to three sentences each) that should make it fire strongly;
- "near_misses": 5 texts, one per example, as similar as possible but which should NOT make it fire;
- "blind_max": the single text you expect to make it fire as strongly as possible.
Reply as {{"examples": [...], "near_misses": [...], "blind_max": "..."}}"""


def _examples_block(title: str, examples: list[Any]) -> str:
    if not examples:
        return f"{title}: none\n"
    lines = [f"{title}:"]
    lines += [f"  {i + 1}. {ex.text}" for i, ex in enumerate(examples)]
    return "\n".join(lines) + "\n"


def format_evidence(
    neuron: str,
    evidence: dict[str, PoleEvidence],
    logits: dict[str, dict[str, list[tuple[str, float]]]],
    steering: dict[str, Any],
) -> str:
    sections = [f"Neuron {neuron}."]
    for pole, ev in evidence.items():
        s = ev.stats
        sections.append(
            f"\n=== {pole.upper()} POLE ===\n"
            f"Strongest gate-open activation: {s.max_act:.3g}. "
            f"Share of tokens above 10% of that: {s.density:.2%}.\n"
            + _examples_block("Strongest examples (one per document)", ev.top)
            + _examples_block("Weaker examples (20-50% of max, random sample)", ev.weaker)
            + _examples_block(
                f"Gate-closed leak firings in this direction (ignore for the job; strongest "
                f"is {s.leak_max_ratio:.0%} of max, {s.leak_share_top:.0%} of a naive top-50)",
                ev.leaks,
            )
            + f"Next tokens pushed up: {json.dumps([t for t, _ in logits[pole]['promotes']], ensure_ascii=False)}\n"
            + f"Next tokens pushed down: {json.dumps([t for t, _ in logits[pole]['suppresses']], ensure_ascii=False)}\n"
            + "Steering (greedy continuation, pole forced on):\n"
            + "\n".join(
                f"  prompt {p!r}\n    normal:  {steering['baseline'][i]!r}\n"
                f"    steered: {steering[pole][i]!r}"
                for i, p in enumerate(steering["prompts"])
            )
        )
    return "\n".join(sections)


def format_test_results(results: dict[str, dict[str, list[dict[str, Any]]]]) -> str:
    lines = [
        "Test results (score 0-10 relative to the pole's corpus maximum, at the strongest token):"
    ]
    for pole, groups in results.items():
        lines.append(f"\n{pole.upper()} POLE")
        for group, rows in groups.items():
            lines.append(f"  {group}:")
            lines += [f"    {r['score']:4.1f} at {r['token']!r}: {r['text']}" for r in rows]
    return "\n".join(lines)
