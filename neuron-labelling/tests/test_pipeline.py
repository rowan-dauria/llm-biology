import json

import pytest
from neuron_labelling.collect import pick_random_neurons, run_collect
from neuron_labelling.investigate import InvestigationConfig
from neuron_labelling.label import label_collection
from neuron_labelling.llm import parse_json
from neuron_labelling.model import NeuronRef
from neuron_labelling.prompts import FINAL_REQUEST, VERIFIER_SYSTEM


class ScriptedClient:
    """Stands in for the LLM: answers each request type with fixed JSON."""

    def __init__(self):
        self.requests: list[list[dict[str, str]]] = []

    def complete(self, messages):
        self.requests.append([dict(m) for m in messages])
        if messages[0]["content"] == VERIFIER_SYSTEM:
            reply = {
                "examples": ["the cat sat", "a dog ran", "birds fly", "fish swim", "cows moo"],
                "near_misses": ["the cat", "a dog", "birds", "fish", "cows"],
                "blind_max": "cats and dogs",
            }
        elif FINAL_REQUEST in messages[-1]["content"]:
            reply = {
                pole: {
                    "explanation": f"{pole} pole: animals",
                    "category": "one_job",
                    "offshoots": [],
                    "confidence": "low",
                }
                for pole in ("positive", "negative")
            } | {"summary": "an animal neuron"}
        else:
            plan = {"hypothesis": "animals", "should_fire": ["cat", "dog"], "near_misses": ["car"]}
            reply = {"positive": plan, "negative": plan}
        return "Here you go:\n```json\n" + json.dumps(reply) + "\n```"


CORPUS = [
    "The cat sat on the mat, and the dog barked at the mailman.",
    "In 1912 the Titanic sank after striking an iceberg in the North Atlantic.",
    "Denver, Colorado has a 303 area code, green chile, and Red Rocks.",
    "def add(a, b):\n    return a + b\n",
] * 6


@pytest.fixture
def collection(tmp_path, tiny_model, tokenizer):
    neurons = [NeuronRef(0, 7), NeuronRef(1, 30)]
    run_collect(
        tiny_model,
        tokenizer,
        CORPUS,
        neurons,
        tmp_path / "run",
        seq_len=32,
        n_chunks=20,
        batch_size=4,
        meta={"model_id": "tiny", "corpus": "test"},
    )
    return tmp_path / "run"


def test_collect_writes_arrays(collection):
    meta = json.loads((collection / "meta.json").read_text())
    assert meta["neurons"] == ["L0/N7", "L1/N30"]
    assert set(meta["logit_effects"]["L0/N7"]) == {"promotes", "suppresses"}
    import numpy as np

    with np.load(collection / "activations.npz") as data:
        assert data["tokens"].shape == (20, 32)
        assert data["gate"].shape == data["value"].shape == (2, 20, 32)


def test_label_collection_end_to_end(collection, tiny_model, tokenizer):
    investigator, verifier = ScriptedClient(), ScriptedClient()
    config = InvestigationConfig(rounds=2, steer_prompts=["Once upon"], max_new_tokens=4)
    reports = label_collection(
        collection, tiny_model, tokenizer, investigator, verifier, config, [NeuronRef(1, 30)]
    )

    (report,) = reports
    assert report.neuron == "L1/N30"
    assert len(report.test_rounds) == 2
    assert report.test_rounds[0]["results"]["positive"]["should_fire"][0]["text"] == "cat"
    assert report.final["summary"] == "an animal neuron"
    for pole in ("positive", "negative"):
        check = report.verification[pole]
        assert check["outcome"] in {"win", "tie", "loss"}
        assert len(check["examples"]) == 5
        assert check["blind_max_vs_corpus"] >= 0
    # investigator: two test rounds + final, in one growing conversation
    assert len(investigator.requests) == 3
    assert "Strongest examples" in investigator.requests[0][1]["content"]
    assert "Test results" in investigator.requests[2][-1]["content"]
    # verifier never sees the evidence, only the explanation
    assert all("Strongest examples" not in m["content"] for r in verifier.requests for m in r)

    out = collection / "labels"
    saved = json.loads((out / "L1_N30.json").read_text())
    assert saved["final"]["positive"]["category"] == "one_job"
    assert "# L1/N30" in (out / "L1_N30.md").read_text()
    assert "L1/N30" in (out / "summary.md").read_text()


def test_pick_random_neurons(tiny_model):
    neurons = pick_random_neurons(tiny_model, 3, seed=1)
    assert neurons == pick_random_neurons(tiny_model, 3, seed=1)
    assert all(0 <= n.index < 64 for n in neurons)


def test_parse_json_tolerates_prose():
    assert parse_json('Sure! {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    assert parse_json('```json\n{"x": [1, 2]}\n```') == {"x": [1, 2]}
    with pytest.raises(ValueError):
        parse_json("no json here")
