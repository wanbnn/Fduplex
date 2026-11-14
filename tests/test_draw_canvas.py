"""Unit tests for the isolated draw protocol and browser; no GPU or audio needed."""
import json
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pytest

from draw_canvas import apply_update, atomic_json, initial_state, launch_viewer, read_json


def test_patch_preserves_ids_and_untouched_nodes():
    first = apply_update(initial_state(), title="Cadastro",
                         nodes=[{"id": "inicio", "label": "Início", "kind": "start"},
                                {"id": "validar", "label": "Validar CPF", "kind": "decision"},
                                {"id": "fim", "label": "Fim", "kind": "end"}],
                         edges=[{"id": "a1", "source": "inicio", "target": "validar", "label": ""},
                                {"id": "a2", "source": "validar", "target": "fim", "label": "Sim"}])
    assert first["version"] == 1
    second = apply_update(first, nodes=[{"id": "validar", "label": "CPF válido?"}])
    assert len(second["nodes"]) == 3 and len(second["edges"]) == 2
    assert [n for n in second["nodes"] if n["id"] == "validar"][0]["label"] == "CPF válido?"
    assert second["title"] == "Cadastro"


def test_remove_node_cascades_edges():
    state = apply_update(initial_state(),
                         nodes=[{"id": "n1", "label": "1"}, {"id": "n2", "label": "2"}],
                         edges=[{"id": "e1", "source": "n1", "target": "n2"}])
    after = apply_update(state, remove_nodes=["n1"])
    assert len(after["nodes"]) == 1 and not after["edges"]


def test_reject_invalid_references_and_oversized_content_without_mutation():
    state = apply_update(initial_state(), nodes=[{"id": "ok", "label": "Mantido"}])
    before = json.dumps(state)
    with pytest.raises(ValueError):
        apply_update(state, edges=[{"id": "e", "source": "ok", "target": "ausente"}])
    with pytest.raises(ValueError):
        apply_update(state, nodes=[{"id": "script", "label": "<script>" * 100}])
    with pytest.raises(ValueError):
        apply_update(state, nodes=[{"id": "../escape", "label": "No"}])
    assert json.dumps(state) == before


def test_json_atomicity_and_localhost_read_only_server():
    with tempfile.TemporaryDirectory() as tmp:
        state, transcript = Path(tmp) / "canvas.json", Path(tmp) / "transcript.json"
        atomic_json(state, initial_state())
        atomic_json(transcript, {"text": "Criar início", "status": "Ouvindo"})
        server, address = launch_viewer(state, transcript)
        try:
            assert server.server_address[0] == "127.0.0.1"
            with urlopen(address + "/api/state", timeout=3) as reply:
                body = json.load(reply)
            assert body["transcript"]["text"] == "Criar início"
            assert body["version"] == 0
            assert read_json(state, {})["nodes"] == []
            with urlopen(address, timeout=3) as reply:
                assert b"Fduplex Draw" in reply.read()
        finally:
            server.shutdown()
            server.server_close()
