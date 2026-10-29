"""Local MCP tool server for Fduplex. Start with python -m fduplex_mcp."""
from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib.parse import quote_plus, urlsplit

from mcp.server.fastmcp import FastMCP

HOME = Path.home()
WORKSPACE = Path(os.getenv("FDUPLEX_WORKSPACE", str(HOME / "Fduplex-workspace"))).expanduser().resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
READ_ROOTS = [Path(p).expanduser().resolve() for p in os.getenv(
    "FDUPLEX_READ_ROOTS", str(HOME / "Documents") + os.pathsep + str(WORKSPACE)
).split(os.pathsep) if p]
WRITE_ROOTS = [Path(p).expanduser().resolve() for p in os.getenv(
    "FDUPLEX_WRITE_ROOTS", str(WORKSPACE)
).split(os.pathsep) if p]
SKILL_ROOT = Path(__file__).resolve().parent / "skills"
STATE_DIR = Path(os.getenv("FDUPLEX_STATE_DIR", str(WORKSPACE / ".fduplex"))).expanduser().resolve()
STATE_DIR.mkdir(parents=True, exist_ok=True)
DB = STATE_DIR / "index.sqlite3"
MAX_READ = 120_000
MAX_RESULTS = 30
DENIED = {".git", ".ssh", ".gnupg", ".aws", ".config", ".local", "node_modules", ".venv", "__pycache__", ".fduplex"}
mcp = FastMCP("Fduplex Local Assistant")


def allowed(path: str, roots: list[Path], *, must_exist: bool = False) -> Path:
    p = Path(path).expanduser().resolve(strict=False)
    if not any(p == root or root in p.parents for root in roots):
        raise ValueError("Path outside permitted roots: " + str(p))
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return p


def valid_file(p: Path) -> bool:
    return not any(part in DENIED for part in p.parts) and p.is_file() and p.stat().st_size < 1_000_000


def safe_text(p: Path, limit: int = MAX_READ) -> str:
    raw = p.read_bytes()[:limit]
    if b"\x00" in raw:
        raise ValueError("Binary file")
    return raw.decode("utf-8", errors="replace")


@mcp.tool()
def list_files(directory: str = "", pattern: str = "*", limit: int = 100) -> list[dict]:
    """List files in a permitted directory, including sizes."""
    root = allowed(directory or str(WORKSPACE), READ_ROOTS + WRITE_ROOTS, must_exist=True)
    if not root.is_dir():
        raise ValueError("Not a directory")
    results = []
    for p in root.glob(pattern):
        if len(results) >= min(limit, 200):
            break
        if any(part in DENIED for part in p.relative_to(root).parts):
            continue
        results.append({"path": str(p), "directory": p.is_dir(), "size": p.stat().st_size if p.is_file() else None})
    return results


@mcp.tool()
def read_file(path: str, start_line: int = 1, max_lines: int = 250) -> str:
    """Read text from a file in the permitted read roots."""
    p = allowed(path, READ_ROOTS + WRITE_ROOTS, must_exist=True)
    lines = safe_text(p).splitlines()
    a = max(0, start_line - 1)
    return "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines[a:a+min(max_lines, 500)], a))


@mcp.tool()
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a UTF-8 file in the authorized workspace. Replacing an existing file requires overwrite."""
    p = allowed(path, WRITE_ROOTS)
    if len(content.encode()) > 1_000_000:
        raise ValueError("File is too large")
    p.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        p.write_text(content, encoding="utf-8")
    else:
        with p.open("x", encoding="utf-8") as f:
            f.write(content)
    return "Written: " + str(p)


@mcp.tool()
def apply_patch(path: str, patch: str) -> str:
    """Apply a unified diff to a writable UTF-8 file. Strict context matching; fails atomically on mismatch."""
    p = allowed(path, WRITE_ROOTS, must_exist=True)
    original = p.read_text(encoding="utf-8")
    old = original.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0
    lines = patch.splitlines(keepends=True)
    i = 0
    pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    while i < len(lines):
        line = lines[i]
        if line.startswith(("--- ", "+++ ", "diff ", "index ")):
            i += 1
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError("Invalid unified diff hunk")
        start = int(match.group(1)) - 1
        if start < cursor:
            raise ValueError("Overlapping hunks")
        result.extend(old[cursor:start])
        cursor = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@ "):
            item = lines[i]
            if item.startswith("\\ No newline"):
                i += 1
                continue
            if not item or item[0] not in " +-":
                raise ValueError("Malformed patch")
            kind, value = item[0], item[1:]
            if kind in " -":
                if cursor >= len(old) or old[cursor].rstrip("\n") != value.rstrip("\n"):
                    raise ValueError(f"Patch context mismatch at source line {cursor+1}")
                cursor += 1
            if kind in " +":
                result.append(value)
            i += 1
    if not lines or not any(l.startswith("@@ ") for l in lines):
        raise ValueError("No patch hunks")
    result.extend(old[cursor:])
    updated = "".join(result)
    p.write_text(updated, encoding="utf-8")
    return "\n".join(difflib.unified_diff(original.splitlines(), updated.splitlines(), fromfile=str(p), tofile=str(p)))


@mcp.tool()
async def run_terminal(command: str, timeout: int = 20) -> dict:
    """Execute an allowlisted program, without a shell, inside the workspace. No pipes or shell expansion."""
    argv = shlex.split(command)
    # Only metadata commands or strictly scoped file reads are accepted here.
    metadata_commands = {"pwd", "date", "uname", "df", "free", "whoami"}
    file_commands = {"ls", "cat", "head", "tail"}
    if argv[0] not in metadata_commands | file_commands:
        raise ValueError("Executable not permitted")
    if argv[0] in metadata_commands and len(argv) != 1:
        raise ValueError("Arguments disabled for metadata commands")
    if argv[0] in file_commands:
        if not 1 <= len(argv) <= 6:
            raise ValueError("Provide 1 to 5 paths")
        for argument in argv[1:]:
            if argument.startswith("-"):
                raise ValueError("Flags disabled for filesystem commands")
            target = allowed(str(WORKSPACE / argument), WRITE_ROOTS, must_exist=True)
            if any(part in DENIED or part.startswith(".env") for part in target.parts):
                raise ValueError("Protected path")
            if argv[0] in {"cat", "head", "tail"} and not target.is_file():
                raise ValueError("Expected file")
    proc = await asyncio.create_subprocess_exec(*argv, cwd=str(WORKSPACE), stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                env={"PATH": "/usr/bin:/bin", "HOME": str(WORKSPACE), "LANG": "C.UTF-8"})
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), min(max(timeout, 1), 30))
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return {"error": "timeout"}
    return {"returncode": proc.returncode, "stdout": stdout[:16000].decode(errors="replace"),
            "stderr": stderr[:4000].decode(errors="replace")}


def db_connect():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, name TEXT, content TEXT, modified REAL)")
    return c


@mcp.tool()
def index_files(max_files: int = 5000) -> dict:
    """Index permitted document folders by filename and text content into local SQLite."""
    scanned = 0
    with db_connect() as db:
        for root in READ_ROOTS + WRITE_ROOTS:
            if not root.is_dir():
                continue
            for base, dirs, files in os.walk(root):
                dirs[:] = [x for x in dirs if x not in DENIED]
                for name in files:
                    if scanned >= min(max_files, 20000):
                        return {"indexed": scanned, "truncated": True}
                    p = Path(base) / name
                    try:
                        if not valid_file(p):
                            continue
                        content = safe_text(p, 32_000)
                        db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?)", (str(p), name, content, p.stat().st_mtime))
                        scanned += 1
                    except (OSError, ValueError):
                        continue
    return {"indexed": scanned, "truncated": False}


@mcp.tool()
def search_files(query: str, limit: int = 12) -> list[dict]:
    """Rank indexed files using filename matches, content word hits and modified time. Run index_files first."""
    words = re.findall(r"[\w-]+", query.casefold())[:12]
    if not words:
        return []
    scored = []
    with db_connect() as db:
        for path, name, content, modified in db.execute("SELECT path, name, content, modified FROM files"):
            try:
                allowed(path, READ_ROOTS + WRITE_ROOTS, must_exist=True)
            except (OSError, ValueError):
                continue
            low_name, low_content = name.casefold(), content.casefold()
            score = sum(12 * (w in low_name) + min(4, low_content.count(w)) for w in words)
            if score:
                idx = next((low_content.find(w) for w in words if w in low_content), -1)
                excerpt = content[max(0, idx-70):idx+220] if idx >= 0 else ""
                scored.append((score, modified, {"path": path, "score": score, "snippet": excerpt}))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored[:min(max(limit, 1), MAX_RESULTS)]]


@mcp.tool()
def list_skills() -> list[dict]:
    """Discover specialist skills; invoke read_skill only if a skill matches the user's task."""
    output = []
    for p in sorted(SKILL_ROOT.glob("*/SKILL.md")):
        text = p.read_text(encoding="utf-8")
        first = next((x.lstrip("# ").strip() for x in text.splitlines() if x.startswith("# ")), p.parent.name)
        output.append({"name": p.parent.name, "title": first})
    return output


@mcp.tool()
def read_skill(name: str) -> str:
    """Load the full instructions for a discovered skill."""
    if not re.fullmatch(r"[a-z0-9_-]{1,50}", name):
        raise ValueError("Invalid skill name")
    p = SKILL_ROOT / name / "SKILL.md"
    if not p.is_file():
        raise FileNotFoundError(name)
    return safe_text(p, 30_000)


@mcp.tool()
async def browse_page(url: str, wait_ms: int = 600) -> dict:
    """Use isolated Chromium via Playwright to read page title, URL, links and visible text."""
    from playwright.async_api import async_playwright
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Only http(s) URLs allowed")
    import ipaddress
    import socket
    host = parsed.hostname or ""
    if host.lower() in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Local network access not allowed")
    def check_host():
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, parsed.port or 443):
            address = ipaddress.ip_address(sockaddr[0])
            if not address.is_global:
                raise ValueError("Private, loopback, or reserved IP target rejected")
    await asyncio.to_thread(check_host)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            page = await browser.new_page(accept_downloads=False)
            # Block private redirects and subresources as well as top-level navigation.
            async def guard(route):
                dest = urlsplit(route.request.url)
                if dest.scheme not in {"http", "https"}:
                    await route.abort()
                    return
                try:
                    await asyncio.to_thread(lambda: validate_host(dest.hostname or "", dest.port or 443))
                    await route.continue_()
                except Exception:
                    await route.abort()
            await page.route("**/*", guard)
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(min(max(wait_ms, 0), 3000))
            return {"url": page.url, "title": await page.title(), "text": (await page.locator("body").inner_text())[:12000],
                    "links": (await page.locator("a[href]").evaluate_all("(els)=>els.slice(0,30).map(e=>({text:e.innerText.slice(0,80),href:e.href}))"))}
        finally:
            await browser.close()


def validate_host(host: str, port: int) -> None:
    import ipaddress
    import socket
    if host.lower() == "localhost" or host.endswith(".local"):
        raise ValueError("Forbidden host")
    for _, _, _, _, addr in socket.getaddrinfo(host, port):
        if not ipaddress.ip_address(addr[0]).is_global:
            raise ValueError("Forbidden IP")


@mcp.tool()
async def web_search(query: str) -> dict:
    """Search the internet using a browser search-results page (not a paid API)."""
    return await browse_page("https://www.google.com/search?q=" + quote_plus(query))


@mcp.tool()
def system_info() -> dict:
    """Return non-sensitive information about the host and configured workspace."""
    import platform
    import shutil
    total, used, free = shutil.disk_usage(WORKSPACE)
    return {"os": platform.platform(), "python": platform.python_version(), "workspace": str(WORKSPACE),
            "disk_free_gb": round(free / 1e9, 2), "read_roots": list(map(str, READ_ROOTS)),
            "write_roots": list(map(str, WRITE_ROOTS))}


if __name__ == "__main__":
    mcp.run(transport="stdio")
