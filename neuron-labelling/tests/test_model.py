import numpy as np
import pytest
import torch
from neuron_labelling.model import NeuronRecorder, NeuronRef, NeuronSteerer, Probe, logit_effects


def test_parse_neuron_ref():
    assert NeuronRef.parse("L12/N345") == NeuronRef(12, 345)
    assert NeuronRef.parse("3:7") == NeuronRef(3, 7)
    assert str(NeuronRef(1, 2)) == "L1/N2"
    with pytest.raises(ValueError):
        NeuronRef.parse("layer twelve")


def _down_proj_inputs(model, layer, input_ids):
    captured = {}

    def hook(_module, args):
        captured["h"] = args[0].detach()

    handle = model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    return captured["h"]


def test_recorder_matches_mlp_activation(tiny_model):
    neurons = [NeuronRef(1, 5), NeuronRef(0, 3), NeuronRef(1, 40)]
    input_ids = torch.randint(0, 256, (2, 9))
    with NeuronRecorder(tiny_model, neurons) as recorder, torch.no_grad():
        tiny_model(input_ids=input_ids)
        gate, value = recorder.read()
    assert gate.shape == value.shape == (3, 2, 9)

    act = torch.nn.functional.silu(torch.from_numpy(gate)) * torch.from_numpy(value)
    for column, neuron in enumerate(neurons):
        hidden = _down_proj_inputs(tiny_model, neuron.layer, input_ids)
        torch.testing.assert_close(act[column], hidden[..., neuron.index], atol=1e-5, rtol=1e-4)


def test_recorder_removes_hooks(tiny_model):
    mlp = tiny_model.model.layers[0].mlp
    with NeuronRecorder(tiny_model, [NeuronRef(0, 1)]):
        assert mlp._forward_pre_hooks
    assert not mlp._forward_pre_hooks


def test_recorder_rejects_out_of_range(tiny_model):
    with pytest.raises(ValueError):
        NeuronRecorder(tiny_model, [NeuronRef(0, 64)])


def test_steerer_adds_delta(tiny_model):
    neuron = NeuronRef(0, 11)
    input_ids = torch.randint(0, 256, (1, 6))
    base = _down_proj_inputs(tiny_model, 0, input_ids)
    with NeuronSteerer(tiny_model, neuron, 2.5):
        steered = _down_proj_inputs(tiny_model, 0, input_ids)
    torch.testing.assert_close(steered[..., 11], base[..., 11] + 2.5)
    torch.testing.assert_close(steered[..., :11], base[..., :11])


def test_probe_measure_ignores_padding(tiny_model, tokenizer):
    probe = Probe(tiny_model, tokenizer, [NeuronRef(1, 2)], batch_size=4)
    short, long = probe.measure(["hi", "a longer text"])
    alone = probe.measure(["hi"])[0]
    assert len(short.tokens) == 2 and short.act.shape == (1, 2)
    assert long.act.shape == (1, len("a longer text"))
    np.testing.assert_allclose(short.act, alone.act, atol=1e-5)


def test_generate_changes_with_steering(tiny_model, tokenizer):
    probe = Probe(tiny_model, tokenizer, [NeuronRef(1, 2)])
    (normal,) = probe.generate(["abc"], max_new_tokens=8)
    (steered,) = probe.generate(["abc"], NeuronRef(1, 2), 500.0, max_new_tokens=8)
    assert normal != steered


def test_logit_effects_signs(tiny_model, tokenizer):
    effects = logit_effects(tiny_model, tokenizer, NeuronRef(1, 2), k=5)
    promotes = [v for _, v in effects["promotes"]]
    suppresses = [v for _, v in effects["suppresses"]]
    assert len(promotes) == len(suppresses) == 5
    assert promotes == sorted(promotes, reverse=True)
    assert suppresses == sorted(suppresses)
    assert promotes[0] > suppresses[0]
