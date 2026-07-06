from __future__ import annotations

from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder, load_transcoder


def test_circuit_tracer_exposes_required_transcoder_api() -> None:
    assert SingleLayerTranscoder is not None
    assert callable(load_transcoder)
