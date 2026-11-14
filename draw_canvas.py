"""Fluxogram canvas state, restricted MCP stdio server and localhost-only viewer."""
from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fduplex Draw Canvas")
KINDS = {"start", "end", "process", "decision", "input", "note"}
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,59}$")
MAX_NODES = 180
MAX_EDGES = 320
APP_DIR = Path(__file__).resolve().parent


def state_file() -> Path:
    value = os.getenv("FDUPLEX_DRAW_STATE")
    if not value:
        raise RuntimeError("FDUPLEX_DRAW_STATE não configurado")
    return Path(value).expanduser().resolve()


def initial_state() -> dict:
    return {"version": 0, "title": "Fluxograma ao vivo", "nodes": [], "edges": []}


def read_json(path: Path, fallback: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fallback


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".draw-", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def identifier(value: object) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("ID inválido (letras, dígitos, _ e -; inicia com letra)")
    return value


def label(value: object, limit: int = 120) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Texto obrigatório, limite de " + str(limit) + " caracteres")
    return value.strip()


def apply_update(
    data: dict, *, title: str = "", nodes: list[dict] | None = None,
    edges: list[dict] | None = None, remove_nodes: list[str] | None = None,
    remove_edges: list[str] | None = None,
) -> dict:
    """Pure, transactional patch. Retain previous graph elements not mentioned."""
    result = {"version": data.get("version", 0), "title": data.get("title", "Fluxograma ao vivo"),
              "nodes": [dict(n) for n in data.get("nodes", [])],
              "edges": [dict(e) for e in data.get("edges", [])]}
    if title:
        result["title"] = label(title, 100)
    removed_nodes = set(map(identifier, remove_nodes or []))
    removed_edges = set(map(identifier, remove_edges or []))
    node_map = {n["id"]: n for n in result["nodes"] if n["id"] not in removed_nodes}
    edge_map = {e["id"]: e for e in result["edges"]
                if e["id"] not in removed_edges
                and e["source"] not in removed_nodes and e["target"] not in removed_nodes}
    for node in nodes or []:
        if not isinstance(node, dict):
            raise ValueError("Nó precisa ser objeto")
        nid = identifier(node.get("id"))
        kind = node.get("kind", node_map.get(nid, {}).get("kind", "process"))
        if kind not in KINDS:
            raise ValueError("Tipo de nó inválido: " + str(kind))
        text = label(node.get("label", node_map.get(nid, {}).get("label")))
        node_map[nid] = {"id": nid, "label": text, "kind": kind}
    if len(node_map) > MAX_NODES:
        raise ValueError("Limite de nós excedido")
    for edge in edges or []:
        if not isinstance(edge, dict):
            raise ValueError("Aresta precisa ser objeto")
        eid = identifier(edge.get("id"))
        previous = edge_map.get(eid, {})
        source = identifier(edge.get("source", previous.get("source")))
        target = identifier(edge.get("target", previous.get("target")))
        if source not in node_map or target not in node_map:
            raise ValueError("Conexões precisam apontar para nós existentes")
        edge_label = edge.get("label", previous.get("label", ""))
        if not isinstance(edge_label, str) or len(edge_label) > 60:
            raise ValueError("Rótulo da conexão excede 60 caracteres")
        edge_map[eid] = {"id": eid, "source": source, "target": target, "label": edge_label}
    if len(edge_map) > MAX_EDGES:
        raise ValueError("Limite de conexões excedido")
    result["nodes"] = list(node_map.values())
    result["edges"] = list(edge_map.values())
    result["version"] += 1
    return result


@mcp.tool()
def get_canvas() -> dict:
    """Read the current drawing: stable node/edge IDs, shapes, labels and connections."""
    return read_json(state_file(), initial_state())


@mcp.tool()
def update_canvas(
    title: str = "", nodes: list[dict] | None = None, edges: list[dict] | None = None,
    remove_nodes: list[str] | None = None, remove_edges: list[str] | None = None,
) -> dict:
    """Incrementally draw/update a flowchart. nodes: [{id,label,kind}], kind=start/end/process/decision/input/note. edges: [{id,source,target,label}]. Reuse IDs to edit, never recreate unrelated nodes. Remove by IDs when corrected."""
    path = state_file()
    updated = apply_update(read_json(path, initial_state()), title=title, nodes=nodes,
                           edges=edges, remove_nodes=remove_nodes, remove_edges=remove_edges)
    atomic_json(path, updated)
    return {"ok": True, "version": updated["version"], "nodes": len(updated["nodes"]),
            "edges": len(updated["edges"])}


def launch_viewer(state_path: Path, transcript_path: Path, port: int = 0):
    """Serve only the fixed canvas page and read-only JSON on loopback."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/state":
                graph = read_json(state_path, initial_state())
                graph["transcript"] = read_json(transcript_path, {"text": "", "status": "Ouvindo…"})
                payload = json.dumps(graph, ensure_ascii=False).encode("utf-8")
                mime = "application/json; charset=utf-8"
            elif self.path == "/":
                payload = (APP_DIR / "draw_web.html").read_bytes()
                mime = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, name="fduplex-draw-web", daemon=True)
    thread.start()
    return server, "http://127.0.0.1:" + str(server.server_port)


if __name__ == "__main__":
    mcp.run(transport="stdio")
