"""Parse Python modules with ast (no import side effects)."""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    methods: List[str]
    lineno: int


@dataclass
class FunctionInfo:
    name: str
    lineno: int


@dataclass
class ModuleStructure:
    path: Path
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    assigns: List[tuple[str, int]]  # top-level NAME = value


def analyze_module(path: Path) -> ModuleStructure:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    classes: List[ClassInfo] = []
    functions: List[FunctionInfo] = []
    assigns: List[tuple[str, int]] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
                elif isinstance(b, ast.Attribute):
                    parts = []
                    cur: ast.AST = b
                    while isinstance(cur, ast.Attribute):
                        parts.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        parts.append(cur.id)
                    bases.append(".".join(reversed(parts)))
            methods = [
                n.name
                for n in node.body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")
            ]
            classes.append(
                ClassInfo(
                    name=node.name,
                    bases=bases,
                    methods=methods,
                    lineno=node.lineno,
                )
            )
        elif isinstance(node, ast.FunctionDef):
            functions.append(FunctionInfo(name=node.name, lineno=node.lineno))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    assigns.append((t.id, node.lineno))

    return ModuleStructure(
        path=path,
        classes=classes,
        functions=functions,
        assigns=assigns,
    )


def registry_dict_from_source(path: Path, registry_name: str) -> Optional[dict]:
    """Best-effort extract a dict literal assigned to registry_name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == registry_name:
                if isinstance(node.value, ast.Dict):
                    out = {}
                    for k, v in zip(node.value.keys, node.value.values):
                        key = None
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            key = k.value
                        if key and isinstance(v, ast.Name):
                            out[key] = v.id
                    return out
    return None


def get_registry_map(path: Path) -> Optional[dict]:
    for name in ("_REGISTRY", "REGISTRY"):
        m = registry_dict_from_source(path, name)
        if m:
            return m
    return None
