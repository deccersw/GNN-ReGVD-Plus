"""Parser backend contract.

A backend turns one source file into :class:`FileFacts`. Two backends exist:
a dependency-free lexical one (always available) and a tree-sitter one (more
accurate, optional). Both must honour the coordinate convention described in
``interproc.models``: every offset on a CallSite is relative to the owning
FunctionDef.func_text.
"""

from typing import Protocol

from ..models import FileFacts


class ParserBackend(Protocol):
    name: str
    version: str

    def parse(self, path: str, rel_path: str, text: str,
              language: str) -> FileFacts:
        ...


def select_backend(preference: str = "auto"):
    """Resolve a backend name to an instance, degrading gracefully."""
    from .lex_backend import LexerBackend

    if preference in ("lexer", "regex"):
        return LexerBackend()

    if preference in ("auto", "treesitter"):
        try:
            from .treesitter_backend import TreeSitterBackend
            return TreeSitterBackend()
        except Exception:
            if preference == "treesitter":
                raise
            return LexerBackend()

    raise ValueError(f"unknown parser backend: {preference}")
