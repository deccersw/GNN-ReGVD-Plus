"""
Dependency-free parser backend built on the lexical scanner (decision D-007).

It finds function definitions by brace matching rather than by regex, which is
what makes the byte offsets it reports trustworthy enough to splice on.
Accuracy is deliberately traded for robustness: anything it cannot understand
is skipped and recorded, never guessed at.
"""

import hashlib
import logging
from typing import List, Optional, Set, Tuple

from .. import clex
from ..clex import (CODE, COMMENT, PREPROC, NON_CALL_KEYWORDS,
                    POST_PARAM_TOKENS, TYPE_KEYWORDS)
from ..models import (CallSite, FileFacts, FuncId, FunctionDef, GlobalDef,
                      Include, MacroDef, Param, TypeDef)

logger = logging.getLogger(__name__)

DECL_STOP_WORDS = NON_CALL_KEYWORDS | {"return", "else", "case", "goto",
                                       "break", "continue", "default"}


class LexerBackend:
    name = "lexer"
    version = "1"

    def parse(self, path: str, rel_path: str, text: str,
              language: str) -> FileFacts:
        mask = clex.code_mask(text)
        facts = FileFacts(
            path=path,
            rel_path=rel_path,
            language=language,
            text=text,
            sha1=hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16],
        )

        functions, ranges = self._find_functions(text, mask, rel_path)
        facts.functions = functions
        self._collect_preproc(text, mask, facts)
        self._collect_top_level(text, mask, ranges, facts)
        return facts

    # ------------------------------------------------------------------
    # Function definitions
    # ------------------------------------------------------------------

    def _find_functions(self, text: str, mask: bytearray,
                        rel_path: str) -> Tuple[List[FunctionDef], List[Tuple[int, int]]]:
        funcs: List[FunctionDef] = []
        ranges: List[Tuple[int, int]] = []
        n = len(text)
        i = 0

        while i < n:
            if mask[i] != CODE or text[i] != '(':
                i += 1
                continue

            close = clex.match_pair(text, mask, i, '(', ')')
            if close < 0:
                break

            brace = self._skip_post_params(text, mask, close)
            if brace < 0:
                i = close
                continue

            ident = clex.ident_before(text, mask, i)
            if ident is None or text[ident[0]:ident[1]] in NON_CALL_KEYWORDS:
                i = close
                continue

            # An unbalanced body means the file (or the snippet) is cut off.
            # Recover by taking the rest of the text and flagging it, rather
            # than dropping the function entirely (decision D-018).
            body_end = clex.match_pair(text, mask, brace, '{', '}')
            truncated_body = body_end < 0
            if truncated_body:
                body_end = len(text)

            built = self._build_function(text, mask, rel_path, ident, i, close,
                                         brace, body_end, truncated_body)
            if built is not None:
                fn, decl_start = built
                funcs.append(fn)
                ranges.append((decl_start, body_end))
            i = body_end

        return funcs, ranges

    def _skip_post_params(self, text: str, mask: bytearray, pos: int) -> int:
        """Index of the '{' that opens a body, or -1 if this is not a definition."""
        n = len(text)
        j = clex.next_code(text, mask, pos)
        while j < n:
            c = text[j]
            if c == '{':
                return j
            if c == ':' and not (j + 1 < n and text[j + 1] == ':'):
                # C++ constructor initialiser list
                return self._scan_to_brace(text, mask, j)
            if c == '-' and j + 1 < n and text[j + 1] == '>':
                # trailing return type
                return self._scan_to_brace(text, mask, j + 2)
            if c == '(':
                k = clex.match_pair(text, mask, j, '(', ')')
                if k < 0:
                    return -1
                j = clex.next_code(text, mask, k)
                continue
            if c.isalpha() or c == '_':
                e = j
                while e < n and mask[e] == CODE and (text[e].isalnum() or text[e] == '_'):
                    e += 1
                if text[j:e] not in POST_PARAM_TOKENS:
                    return -1
                j = clex.next_code(text, mask, e)
                continue
            return -1
        return -1

    @staticmethod
    def _scan_to_brace(text: str, mask: bytearray, start: int) -> int:
        n = len(text)
        depth = 0
        k = start
        while k < n:
            if mask[k] == CODE:
                c = text[k]
                if c in '([':
                    depth += 1
                elif c in ')]':
                    depth -= 1
                elif c == '{' and depth == 0:
                    return k
                elif c == ';' and depth == 0:
                    return -1
            k += 1
        return -1

    def _build_function(self, text: str, mask: bytearray, rel_path: str,
                        ident: Tuple[int, int], lparen: int, rparen: int,
                        brace: int, body_end: int,
                        truncated_body: bool = False) -> Optional[Tuple[FunctionDef, int]]:
        name_start, name_end = ident
        name = text[name_start:name_end]
        decl_start = self._decl_start(text, mask, name_start)
        func_text = text[decl_start:body_end]
        prefix = text[decl_start:name_start]
        params, is_vararg = parse_params(text[lparen + 1:rparen - 1])

        qualifier = clex.qualifier_before(text, mask, name_start)
        qualified = f"{qualifier}::{name}" if qualifier else name

        fmask = clex.code_mask(func_text)
        body_start_rel = brace - decl_start
        body_end_rel = body_end - decl_start

        fid = FuncId(file=rel_path, name=name,
                     start_line=clex.line_of(text, decl_start))

        fn = FunctionDef(
            fid=fid,
            name=name,
            qualified_name=qualified,
            return_type=" ".join(prefix.split()),
            params=params,
            func_text=func_text,
            body_start=body_start_rel,
            body_end=body_end_rel,
            start_line=fid.start_line,
            end_line=clex.line_of(text, body_end),
            is_static="static" in prefix.split(),
            is_vararg=is_vararg,
            is_virtual="virtual" in prefix.split(),
            is_template=self._looks_templated(text, decl_start),
        )

        fn.calls = self._find_calls(func_text, fmask, body_start_rel,
                                    body_end_rel, fn.start_line)
        fn.flags = self._function_flags(func_text, fmask, body_start_rel,
                                        body_end_rel)
        if truncated_body:
            fn.flags.add("truncated_body")
        fn.local_names = {p.name for p in params} | declared_names(
            func_text, fmask, body_start_rel, body_end_rel)
        return fn, decl_start

    @staticmethod
    def _decl_start(text: str, mask: bytearray, name_start: int) -> int:
        """Beginning of the declaration that introduces the function name."""
        j = name_start - 1
        while j >= 0:
            if mask[j] == PREPROC:
                break
            if mask[j] == CODE:
                c = text[j]
                if c in ';{}':
                    break
                if c == ':' and not (j > 0 and text[j - 1] == ':') \
                        and not (j + 1 < len(text) and text[j + 1] == ':'):
                    break
            j -= 1
        j += 1
        while j < name_start and (mask[j] in (COMMENT, PREPROC)
                                  or text[j].isspace()):
            j += 1
        return j

    @staticmethod
    def _looks_templated(text: str, decl_start: int) -> bool:
        window = text[max(0, decl_start - 200):decl_start]
        return "template" in window and window.rstrip().endswith('>')

    @staticmethod
    def _function_flags(func_text: str, fmask: bytearray,
                        body_start: int, body_end: int) -> Set[str]:
        body = func_text[body_start:body_end]
        bmask = fmask[body_start:body_end]
        flags: Set[str] = set()
        if clex.contains_word(body, bmask, ("goto",)):
            flags.add("has_goto")
        if clex.has_label(body, bmask):
            flags.add("has_label")
        if clex.contains_word(body, bmask, ("asm", "__asm", "__asm__")):
            flags.add("has_asm")
        if clex.contains_word(body, bmask, ("setjmp", "longjmp", "sigsetjmp")):
            flags.add("has_setjmp")
        if PREPROC in bmask:
            directives = {"#if", "#ifdef", "#ifndef", "#else", "#elif"}
            for line in body.splitlines():
                stripped = line.strip()
                if any(stripped.startswith(d) for d in directives):
                    flags.add("has_preproc_branch")
                    break
        return flags

    # ------------------------------------------------------------------
    # Call sites
    # ------------------------------------------------------------------

    def _find_calls(self, func_text: str, fmask: bytearray,
                    body_start: int, body_end: int,
                    func_start_line: int) -> List[CallSite]:
        calls: List[CallSite] = []
        i = body_start
        n = body_end

        while i < n:
            if fmask[i] != CODE or not (func_text[i].isalpha() or func_text[i] == '_'):
                i += 1
                continue

            j = i
            while j < n and fmask[j] == CODE and (func_text[j].isalnum() or func_text[j] == '_'):
                j += 1
            name = func_text[i:j]

            nxt = clex.next_code(func_text, fmask, j)
            if nxt >= n or func_text[nxt] != '(' or name in NON_CALL_KEYWORDS \
                    or name in TYPE_KEYWORDS:
                i = j
                continue

            close = clex.match_pair(func_text, fmask, nxt, '(', ')')
            if close < 0:
                i = j
                continue

            inner = func_text[nxt + 1:close - 1]
            args = [a.strip() for a in clex.split_top_level(inner)] \
                if inner.strip() else []

            kind, receiver, site_start = self._call_receiver(
                func_text, fmask, i)

            calls.append(CallSite(
                callee_name=name,
                arg_texts=args,
                start=site_start,
                end=close,
                line=func_start_line + func_text.count('\n', 0, i),
                kind=kind,
                receiver=receiver,
                result_used=self._result_used(func_text, fmask, site_start, close),
            ))
            # keep scanning inside the arguments so nested calls are found too
            i = nxt + 1

        return calls

    @staticmethod
    def _call_receiver(func_text: str, fmask: bytearray,
                       ident_start: int) -> Tuple[str, str, int]:
        prev = clex.prev_code(func_text, fmask, ident_start - 1)
        if prev < 0:
            return "direct", "", ident_start
        c = func_text[prev]
        if c == '.':
            recv = clex.ident_before(func_text, fmask, prev)
            start = recv[0] if recv else ident_start
            return "method", func_text[recv[0]:recv[1]] if recv else "", start
        if c == '>' and prev > 0 and func_text[prev - 1] == '-':
            recv = clex.ident_before(func_text, fmask, prev - 1)
            start = recv[0] if recv else ident_start
            return "method", func_text[recv[0]:recv[1]] if recv else "", start
        if c == ':' and prev > 0 and func_text[prev - 1] == ':':
            return "direct", "", ident_start
        return "direct", "", ident_start

    @staticmethod
    def _result_used(func_text: str, fmask: bytearray,
                     start: int, close: int) -> bool:
        after = clex.next_code(func_text, fmask, close)
        if after >= len(func_text) or func_text[after] != ';':
            return True
        prev = clex.prev_code(func_text, fmask, start - 1)
        return not (prev < 0 or func_text[prev] in ';{}:)')

    # ------------------------------------------------------------------
    # Preprocessor and top-level declarations (needed by the bundler)
    # ------------------------------------------------------------------

    def _collect_preproc(self, text: str, mask: bytearray,
                         facts: FileFacts) -> None:
        n = len(text)
        i = 0
        while i < n:
            if mask[i] != PREPROC:
                i += 1
                continue
            j = i
            while j < n and mask[j] == PREPROC:
                j += 1
            line = text[i:j]
            i = j

            body = line.lstrip()[1:].lstrip()
            if body.startswith("include"):
                rest = body[len("include"):].strip()
                if rest.startswith('<') and '>' in rest:
                    facts.includes.append(
                        Include(rest[1:rest.index('>')], True))
                elif rest.startswith('"') and rest.count('"') >= 2:
                    facts.includes.append(
                        Include(rest[1:rest.index('"', 1)], False))
            elif body.startswith("define"):
                rest = body[len("define"):].strip()
                k = 0
                while k < len(rest) and (rest[k].isalnum() or rest[k] == '_'):
                    k += 1
                if k:
                    facts.macros.append(
                        MacroDef(rest[:k], k < len(rest) and rest[k] == '('))

    def _collect_top_level(self, text: str, mask: bytearray,
                           ranges: List[Tuple[int, int]],
                           facts: FileFacts) -> None:
        n = len(text)
        skip = sorted(ranges)
        i = 0
        stmt_start: Optional[int] = None
        si = 0

        while i < n:
            while si < len(skip) and skip[si][1] <= i:
                si += 1
            if si < len(skip) and skip[si][0] <= i < skip[si][1]:
                i = skip[si][1]
                stmt_start = None
                continue

            if mask[i] != CODE:
                i += 1
                continue

            c = text[i]
            if stmt_start is None and not c.isspace():
                stmt_start = i

            if c == '{':
                head = " ".join(text[stmt_start:i].split()) \
                    if stmt_start is not None else ""
                if _is_transparent_block(head):
                    # `extern "C" { ... }` and `namespace X { ... }` do not
                    # introduce a scope for declarations -- everything inside
                    # is still top level. Skipping the block wholesale lost
                    # every prototype in a C header with the C++ guard, which
                    # is nearly all of them: tiffio.h yielded 1 prototype
                    # instead of ~200.
                    i += 1
                    stmt_start = None
                    continue
                close = clex.match_pair(text, mask, i, '{', '}')
                if close < 0:
                    break
                i = close
                continue
            if c == ';':
                if stmt_start is not None:
                    self._classify_decl(text[stmt_start:i + 1],
                                        clex.line_of(text, stmt_start), facts)
                stmt_start = None
                i += 1
                continue
            if c == '}':
                stmt_start = None
            i += 1

    @staticmethod
    def _classify_decl(stmt: str, line: int, facts: FileFacts) -> None:
        flat = " ".join(stmt.split())
        if not flat or flat == ';':
            return
        head = flat.split(None, 1)[0]

        if head == "typedef":
            name = _last_identifier(flat[:-1])
            if name:
                facts.types.append(TypeDef(name, "typedef", stmt, line))
            return

        if head in ("struct", "union", "enum", "class") and '{' in flat:
            parts = flat.split()
            name = parts[1].rstrip('{') if len(parts) > 1 else ""
            if name and name != '{':
                facts.types.append(TypeDef(name, head, stmt, line))
            return

        if '(' in flat and '=' not in flat.split('(')[0]:
            mask = clex.code_mask(flat)
            lp = flat.index('(')
            ident = clex.ident_before(flat, mask, lp)
            if ident:
                facts.prototypes.add(flat[ident[0]:ident[1]])
            return

        name = _last_identifier(flat.split('=')[0])
        if name and name not in TYPE_KEYWORDS:
            facts.globals.append(GlobalDef(name, stmt, line))


# ----------------------------------------------------------------------
# Small helpers shared with the tree-sitter backend
# ----------------------------------------------------------------------

def _is_transparent_block(head: str) -> bool:
    """Does this `{` open a linkage/namespace block rather than a type body?"""
    if not head:
        return False
    if head.startswith('extern "C"') or head.startswith('extern "C++"'):
        return True
    first = head.split(None, 1)[0] if head.split() else ""
    return first == "namespace"


def _last_identifier(text: str) -> str:
    mask = clex.code_mask(text)
    last = ""
    for _s, _e, name in clex.iter_identifiers(text, mask):
        if name not in TYPE_KEYWORDS:
            last = name
    return last


def parse_params(param_text: str) -> Tuple[List[Param], bool]:
    """Split a parameter list into (params, is_vararg)."""
    params: List[Param] = []
    is_vararg = False
    raw = param_text.strip()
    if not raw or raw == "void":
        return params, is_vararg

    for idx, part in enumerate(clex.split_top_level(raw)):
        p = part.strip()
        if not p:
            continue
        if p == "...":
            is_vararg = True
            continue

        p_nodefault = p.split('=')[0].strip()
        array_suffix = ""
        while p_nodefault.endswith(']'):
            open_idx = p_nodefault.rfind('[')
            if open_idx < 0:
                break
            array_suffix = p_nodefault[open_idx:] + array_suffix
            p_nodefault = p_nodefault[:open_idx].strip()

        name = ""
        type_text = p_nodefault

        if "(*" in p_nodefault.replace(" ", ""):
            # function pointer parameter: name sits right after "(*"
            compact = p_nodefault
            k = compact.find("(*")
            if k >= 0:
                m = k + 2
                while m < len(compact) and (compact[m].isalnum() or compact[m] == '_'):
                    m += 1
                name = compact[k + 2:m]
                type_text = p_nodefault
        else:
            mask = clex.code_mask(p_nodefault)
            idents = [(s, e, nm) for s, e, nm in clex.iter_identifiers(p_nodefault, mask)]
            if idents and idents[-1][2] not in TYPE_KEYWORDS:
                s, e, nm = idents[-1]
                name = nm
                type_text = (p_nodefault[:s] + p_nodefault[e:]).strip()

        if not name:
            name = f"_p{idx}"
            type_text = p_nodefault

        params.append(Param(type=(type_text + array_suffix).strip() or "int",
                            name=name))

    return params, is_vararg


def declared_names(func_text: str, fmask: bytearray,
                   start: int, end: int) -> Set[str]:
    """Names declared inside a function body (approximate, deliberately tight).

    Used only to decide whether an inlined callee needs alpha-renaming, so
    over-approximating costs extra renames (which we want to avoid, D-011) and
    under-approximating costs a name clash. The rule below requires an
    identifier to be directly preceded by another identifier (its type) and
    followed by one of ``= ; , [ )``.
    """
    names: Set[str] = set()
    for s, e, name in clex.iter_identifiers(func_text, fmask, start, end):
        if name in TYPE_KEYWORDS or name in DECL_STOP_WORDS:
            continue
        prev = clex.prev_code(func_text, fmask, s - 1)
        if prev < 0:
            continue
        pc = func_text[prev]
        anchor = prev
        if pc in '*&':
            anchor = clex.prev_code(func_text, fmask, prev - 1)
            if anchor < 0:
                continue
        elif not (pc.isalnum() or pc == '_' or pc == '>'):
            continue

        preceding = clex.ident_before(func_text, fmask, anchor + 1)
        if preceding is None:
            continue
        prev_word = func_text[preceding[0]:preceding[1]]
        if prev_word in DECL_STOP_WORDS:
            continue

        nxt = clex.next_code(func_text, fmask, e)
        if nxt < len(func_text) and func_text[nxt] in '=;,[)':
            names.add(name)
    return names
