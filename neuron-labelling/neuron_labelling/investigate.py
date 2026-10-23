"""Label one neuron: gather evidence, run a conjecture-and-test loop, then verify blind.

1. Evidence: per-pole strong and weaker gate-open examples, leak examples,
   direct logit effects, and steered generations.
2. Investigation: the investigator proposes hypotheses with should-fire texts
   and near misses; we run them through the model and report the scores; it
   revises for ``rounds`` rounds, then writes the final explanation.
3. Verification: a fresh verifier sees only each pole's final explanation and
   writes new examples, near misses and one "blind maximum" text, which are
   scored against the pole's corpus maximum.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from neuron_labelling.examples import POLES, SIGN, PoleEvidence, make_decoder, pole_strength
from neuron_labelling.llm import ChatClient, Message, parse_json
from neuron_labelling.model import NeuronRef, Probe
from neuron_labelling.prompts import (
    FINAL_REQUEST,
    INVESTIGATOR_SYSTEM,
    RETEST_REQUEST,
    TEST_REQUEST,
    VERIFIER_REQUEST,
    VERIFIER_SYSTEM,
    format_evidence,
    format_test_results,
)

DEFAULT_STEER_PROMPTS = (
    "My plan for this weekend is",
    "The most important thing to remember is",
    "She opened the letter and",
)


@dataclass
class InvestigationConfig:
    rounds: int = 1
    steer_scale: float = 8.0  # steering delta, in multiples of the pole's corpus maximum
    steer_prompts: Sequence[str] = DEFAULT_STEER_PROMPTS
    max_new_tokens: int = 40
    strong: float = 0.5  # a text "fires strongly" at this fraction of the corpus maximum


def score_texts(
    probe: Probe, neuron: NeuronRef, texts: Sequence[str], pole: str, max_act: float
) -> list[dict[str, Any]]:
    """Strongest gate-open firing of ``pole`` in each text, on a 0-10 scale."""
    column = probe.column(neuron)
    decode = make_decoder(probe.tokenizer)
    rows = []
    for text, measured in zip(texts, probe.measure(texts), strict=True):
        strength = pole_strength(measured.gate[column], measured.value[column], pole, True)
        best = int(np.argmax(strength))
        rows.append(
            {
                "text": text,
                "score": round(10 * float(strength[best]) / max_act, 2) if max_act > 0 else 0.0,
                "token": decode(measured.tokens[best]),
            }
        )
    return rows


def steering_samples(
    probe: Probe, neuron: NeuronRef, evidence: dict[str, PoleEvidence], config: InvestigationConfig
) -> dict[str, Any]:
    prompts = list(config.steer_prompts)
    samples: dict[str, Any] = {
        "prompts": prompts,
        "baseline": probe.generate(prompts, max_new_tokens=config.max_new_tokens),
    }
    for pole in POLES:
        delta = SIGN[pole] * config.steer_scale * evidence[pole].stats.max_act
        samples[pole] = probe.generate(prompts, neuron, delta, config.max_new_tokens)
        samples[f"{pole}_delta"] = delta
    return samples


def pole_logits(logits: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, Any]]:
    """A negative activation reverses the direct logit effect."""
    return {
        "positive": logits,
        "negative": {"promotes": logits["suppresses"], "suppresses": logits["promotes"]},
    }


@dataclass
class NeuronReport:
    neuron: str
    evidence: dict[str, Any]
    logit_effects: dict[str, Any]
    steering: dict[str, Any]
    test_rounds: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    transcript: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ask(client: ChatClient, messages: list[Message], request: str) -> dict[str, Any]:
    messages.append({"role": "user", "content": request})
    reply = client.complete(messages)
    messages.append({"role": "assistant", "content": reply})
    return parse_json(reply)


def run_tests(
    probe: Probe,
    neuron: NeuronRef,
    proposal: dict[str, Any],
    evidence: dict[str, PoleEvidence],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pole in POLES:
        plan = proposal.get(pole) or {}
        max_act = evidence[pole].stats.max_act
        results[pole] = {
            group: score_texts(probe, neuron, [str(t) for t in plan.get(group, [])], pole, max_act)
            for group in ("should_fire", "near_misses")
        }
    return results


def verify_pole(
    client: ChatClient,
    probe: Probe,
    neuron: NeuronRef,
    pole: str,
    explanation: str,
    max_act: float,
    strong: float,
) -> dict[str, Any]:
    """Blind test of one explanation: new examples vs. near misses, plus a blind maximum."""
    messages: list[Message] = [{"role": "system", "content": VERIFIER_SYSTEM}]
    reply = _ask(client, messages, VERIFIER_REQUEST.format(explanation=explanation))
    examples = score_texts(
        probe, neuron, [str(t) for t in reply.get("examples", [])], pole, max_act
    )
    misses = score_texts(
        probe, neuron, [str(t) for t in reply.get("near_misses", [])], pole, max_act
    )
    blind = score_texts(probe, neuron, [str(reply.get("blind_max", ""))], pole, max_act)[0]
    hits = sum(r["score"] >= 10 * strong for r in examples)
    false_hits = sum(r["score"] >= 10 * strong for r in misses)
    return {
        "examples": examples,
        "near_misses": misses,
        "blind_max": blind,
        "hit_rate": hits / len(examples) if examples else 0.0,
        "near_miss_hit_rate": false_hits / len(misses) if misses else 0.0,
        "outcome": "win" if hits > false_hits else "tie" if hits == false_hits else "loss",
        "blind_max_vs_corpus": blind["score"] / 10,
    }


def investigate(
    neuron: NeuronRef,
    evidence: dict[str, PoleEvidence],
    logits: dict[str, list[tuple[str, float]]],
    probe: Probe,
    investigator: ChatClient,
    verifier: ChatClient,
    config: InvestigationConfig,
) -> NeuronReport:
    by_pole_logits = pole_logits(logits)
    steering = steering_samples(probe, neuron, evidence, config)
    report = NeuronReport(
        neuron=str(neuron),
        evidence={pole: asdict(ev) for pole, ev in evidence.items()},
        logit_effects=by_pole_logits,
        steering=steering,
    )
    messages: list[Message] = [{"role": "system", "content": INVESTIGATOR_SYSTEM}]
    briefing = format_evidence(str(neuron), evidence, by_pole_logits, steering)

    request = briefing + "\n\n" + TEST_REQUEST
    for _ in range(config.rounds):
        proposal = _ask(investigator, messages, request)
        results = run_tests(probe, neuron, proposal, evidence)
        report.test_rounds.append({"proposal": proposal, "results": results})
        request = format_test_results(results) + "\n\n" + RETEST_REQUEST
    final_request = (
        (format_test_results(report.test_rounds[-1]["results"]) if report.test_rounds else briefing)
        + "\n\n"
        + FINAL_REQUEST
    )
    report.final = _ask(investigator, messages, final_request)
    report.transcript = messages

    for pole in POLES:
        explanation = str((report.final.get(pole) or {}).get("explanation", "")).strip()
        if explanation:
            report.verification[pole] = verify_pole(
                verifier,
                probe,
                neuron,
                pole,
                explanation,
                evidence[pole].stats.max_act,
                config.strong,
            )
    return report
