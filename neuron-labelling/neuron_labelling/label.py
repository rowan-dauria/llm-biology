"""Label the neurons in a collection written by ``neuron_labelling.collect``.

Writes ``<collection>/labels/<L_N>.json`` (full record, including the agent
transcript), ``<L_N>.md`` (a short neuron paper) and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neuron_labelling.examples import POLES, PoleEvidence, make_decoder, pole_evidence
from neuron_labelling.investigate import (
    DEFAULT_STEER_PROMPTS,
    InvestigationConfig,
    NeuronReport,
    investigate,
)
from neuron_labelling.llm import ChatClient, OpenAICompatibleClient
from neuron_labelling.model import NeuronRef, Probe, Tokenizer, load_model


def load_collection(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "activations.npz") as data:
        arrays = {key: data[key] for key in data.files}
    return meta, arrays


def build_evidence(
    meta: dict[str, Any],
    arrays: dict[str, np.ndarray],
    tokenizer: Tokenizer,
    neuron: NeuronRef,
    skip_positions: int,
) -> dict[str, PoleEvidence]:
    column = meta["neurons"].index(str(neuron))
    decode = make_decoder(tokenizer)
    return {
        pole: pole_evidence(
            arrays["tokens"],
            arrays["gate"][column],
            arrays["value"][column],
            pole,
            decode,
            skip_positions=skip_positions,
        )
        for pole in POLES
    }


def paper_markdown(report: NeuronReport) -> str:
    final = report.final
    lines = [f"# {report.neuron}", "", final.get("summary", ""), ""]
    for pole in POLES:
        stats = report.evidence[pole]["stats"]
        verdict = final.get(pole) or {}
        check = report.verification.get(pole)
        lines += [
            f"## {pole.capitalize()} pole",
            "",
            f"**{verdict.get('explanation', '(none)')}**",
            "",
            f"- Category: `{verdict.get('category', '?')}`, confidence: {verdict.get('confidence', '?')}",
            f"- Offshoots: {', '.join(verdict.get('offshoots') or []) or 'none listed'}",
            f"- Corpus max (gate open): {stats['max_act']:.3g}; density {stats['density']:.2%}",
            f"- Leak: strongest {stats['leak_max_ratio']:.0%} of max, "
            f"{stats['leak_share_top']:.0%} of a naive top-50",
            "- Pushes up: "
            + ", ".join(f"`{t}`" for t, _ in report.logit_effects[pole]["promotes"][:8]),
        ]
        if check:
            lines += [
                f"- Verification: **{check['outcome']}**; new examples fire "
                f"{check['hit_rate']:.0%}, near misses {check['near_miss_hit_rate']:.0%}; "
                f"blind max reaches {check['blind_max_vs_corpus']:.0%} of corpus max",
                f"  - blind max: {check['blind_max']['text']!r}",
            ]
        lines += ["", "Top examples:", ""]
        lines += [f"1. `{ex['text']}`" for ex in report.evidence[pole]["top"][:5]]
        lines += [""]
    lines += ["## Steering", ""]
    for i, prompt in enumerate(report.steering["prompts"]):
        lines += [
            f"- {prompt!r}",
            f"  - normal: {report.steering['baseline'][i]!r}",
            f"  - positive: {report.steering['positive'][i]!r}",
            f"  - negative: {report.steering['negative'][i]!r}",
        ]
    return "\n".join(lines) + "\n"


def summary_markdown(reports: Sequence[NeuronReport]) -> str:
    lines = [
        "| neuron | pole | category | explanation | verification | blind max |",
        "|---|---|---|---|---|---|",
    ]
    for report in reports:
        for pole in POLES:
            verdict = report.final.get(pole) or {}
            check = report.verification.get(pole) or {}
            explanation = str(verdict.get("explanation", "")).replace("|", "\\|")
            lines.append(
                f"| {report.neuron} | {pole} | {verdict.get('category', '?')} | {explanation} | "
                f"{check.get('outcome', '-')} | {check.get('blind_max_vs_corpus', 0):.0%} |"
            )
    return "\n".join(lines) + "\n"


def label_collection(
    collection: Path,
    model: Any,
    tokenizer: Tokenizer,
    investigator: ChatClient,
    verifier: ChatClient,
    config: InvestigationConfig,
    neurons: Sequence[NeuronRef] | None = None,
    skip_positions: int = 4,
) -> list[NeuronReport]:
    meta, arrays = load_collection(collection)
    collected = [NeuronRef.parse(n) for n in meta["neurons"]]
    targets = list(neurons) if neurons else collected
    probe = Probe(model, tokenizer, collected)
    out_dir = collection / "labels"
    out_dir.mkdir(exist_ok=True)
    reports = []
    for neuron in targets:
        print(f"labelling {neuron}")
        evidence = build_evidence(meta, arrays, tokenizer, neuron, skip_positions)
        logits = meta["logit_effects"][str(neuron)]
        report = investigate(neuron, evidence, logits, probe, investigator, verifier, config)
        (out_dir / f"{neuron.slug}.json").write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        )
        (out_dir / f"{neuron.slug}.md").write_text(paper_markdown(report))
        reports.append(report)
    (out_dir / "summary.md").write_text(summary_markdown(reports))
    return reports


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path, help="directory written by collect")
    parser.add_argument("--neurons", help="subset to label, e.g. L12/N345")
    parser.add_argument("--llm-model", help="investigator model (default: $LLM_MODEL)")
    parser.add_argument("--verifier-model", help="verifier model (default: investigator)")
    parser.add_argument("--api-base", help="default: $LLM_API_BASE or OpenRouter")
    parser.add_argument("--rounds", type=int, default=1, help="conjecture-and-test rounds")
    parser.add_argument("--steer-scale", type=float, default=8.0)
    parser.add_argument("--steer-prompts", nargs="+", default=list(DEFAULT_STEER_PROMPTS))
    parser.add_argument("--strong", type=float, default=0.5)
    parser.add_argument("--skip-positions", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args(argv)

    meta, _ = load_collection(args.collection)
    model, tokenizer = load_model(meta["model_id"], args.device, args.dtype)
    investigator = OpenAICompatibleClient.from_env(args.llm_model, args.api_base)
    verifier = OpenAICompatibleClient.from_env(
        args.verifier_model or investigator.model, args.api_base
    )
    config = InvestigationConfig(
        rounds=args.rounds,
        steer_scale=args.steer_scale,
        steer_prompts=args.steer_prompts,
        strong=args.strong,
    )
    neurons = [NeuronRef.parse(n) for n in args.neurons.split(",")] if args.neurons else None
    label_collection(
        args.collection,
        model,
        tokenizer,
        investigator,
        verifier,
        config,
        neurons,
        args.skip_positions,
    )
    print(f"wrote {args.collection / 'labels'}")


if __name__ == "__main__":
    main()
