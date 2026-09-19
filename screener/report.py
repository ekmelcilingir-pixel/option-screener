"""Self-contained HTML report rendering (Jinja2)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .strategies import ALL_STRATEGIES

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def _pct(x, d=1):
    try:
        return f"{100 * float(x):.{d}f}%"
    except (TypeError, ValueError):
        return "—"


def _num(x, d=2):
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return "—"


def render_report(results, feats, meta, cfg) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = _pct
    env.filters["num"] = _num
    tpl = env.get_template("report.html.j2")
    top = cfg["report"]["top_per_strategy"]
    sections = []
    for s in ALL_STRATEGIES:
        cands = results.get(s.key, [])
        sections.append({"key": s.key, "label": s.label, "description": s.description, "total": len(cands), "cands": cands[:top]})
    # Names that qualify for more than one strategy
    multi = {}
    for k, cands in results.items():
        for c in cands[:top]:
            multi.setdefault(c.symbol, []).append((k, c.score))
    multi = {s: v for s, v in multi.items() if len(v) > 1}
    return tpl.render(title=cfg["report"]["title"], meta=meta, sections=sections, multi=multi, n_feats=len(feats))
