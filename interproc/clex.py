"""
Minimal C/C++ lexical scanner used by the fallback parser backend.

The whole module exists to answer one question accurately: *is byte i part of
real code, or is it inside a string / comment / preprocessor directive?*

Everything downstream (brace matching, function extraction, call-site
detection, identifier renaming) is byte-range based, so a wrong answer here
does not merely lose a function -- it produces corrupted spliced text.
That is why this is a scanner and not a pile of regexes (decision D-007).

Character classes in the mask:
    'c'  code
    's'  string or char literal (including quotes)
    'm'  comment
    'p'  preprocessor directive (whole logical line)
"""

from typing import Dict, Iterator, List, Optional, Tuple

CODE = ord('c')
STR = ord('s')
COMMENT = ord('m')
PREPROC = ord('p')

# Identifiers that look like calls but are not.
NON_CALL_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "catch", "defined",
    "alignof", "_Alignof", "typeof", "__typeof__", "decltype", "static_assert",
    "_Static_assert", "__attribute__", "__declspec", "noexcept", "throw",
    "do", "else", "case", "goto", "new", "delete", "and", "or", "not",
}

# Tokens allowed between the parameter list and the opening brace of a
# function definition (C++ qualifiers, GNU attributes, trailing return types).
POST_PARAM_TOKENS = {
    "const", "noexcept", "override", "final", "volatile", "mutable",
    "__restrict", "__restrict__", "throw", "try",
}

TYPE_KEYWORDS = {
    "void", "char", "short", "int", "long", "float", "double", "signed",
    "unsigned", "const", "volatile", "struct", "union", "enum", "static",
    "extern", "register", "inline", "restrict", "__restrict", "_Bool",
    "size_t", "ssize_t", "auto", "class", "typename", "constexpr",
}


def code_mask(text: str) -> bytearray:
    """Classify every byte of ``text``. See module docstring for classes."""
    n = len(text)
    mask = bytearray(b'c' * n)
    i = 0
    line_start = True

    while i < n:
        ch = text[i]

        if ch == '\n':
            line_start = True
            i += 1
            continue

        if ch in ' \t\r':
            i += 1
            continue

        if ch == '#' and line_start:
            end = _preproc_line_end(text, i)
            mask[i:end] = b'p' * (end - i)
            i = end
            line_start = True
            continue

        line_start = False

        if ch == '/' and i + 1 < n and text[i + 1] == '/':
            end = text.find('\n', i)
            end = n if end < 0 else end
            mask[i:end] = b'm' * (end - i)
            i = end
            continue

        if ch == '/' and i + 1 < n and text[i + 1] == '*':
            end = text.find('*/', i + 2)
            end = n if end < 0 else end + 2
            mask[i:end] = b'm' * (end - i)
            i = end
            continue

        if ch == '"' or ch == "'":
            end = _literal_end(text, i)
            mask[i:end] = b's' * (end - i)
            i = end
            continue

        i += 1

    return mask


def _preproc_line_end(text: str, start: int) -> int:
    """End of a preprocessor logical line, honouring backslash continuation."""
    n = len(text)
    j = start
    while j < n:
        nl = text.find('\n', j)
        if nl < 0:
            return n
        e = nl
        while e > j and text[e - 1] in ' \t\r':
            e -= 1
        if e > j and text[e - 1] == '\\':
            j = nl + 1
            continue
        return nl
    return n


def _literal_end(text: str, start: int) -> int:
    """Index just past a string/char literal that begins at ``start``."""
    quote = text[start]
    n = len(text)
    j = start + 1
    while j < n:
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == quote:
            return j + 1
        if c == '\n':          # unterminated literal: stop at end of line
            return j
        j += 1
    return n


# --------------------------------------------------------------------------
# Navigation helpers. All of them operate on (text, mask) pairs and only ever
# consider bytes classified as CODE.
# --------------------------------------------------------------------------

def is_code(mask: bytearray, i: int) -> bool:
    return 0 <= i < len(mask) and mask[i] == CODE


def next_code(text: str, mask: bytearray, i: int) -> int:
    """First index >= i that is code and not whitespace (len(text) if none)."""
    n = len(text)
    while i < n:
        if mask[i] == CODE and not text[i].isspace():
            return i
        i += 1
    return n


def prev_code(text: str, mask: bytearray, i: int) -> int:
    """Last index <= i that is code and not whitespace (-1 if none)."""
    while i >= 0:
        if mask[i] == CODE and not text[i].isspace():
            return i
        i -= 1
    return -1


def match_pair(text: str, mask: bytearray, i: int,
               open_ch: str, close_ch: str) -> int:
    """Given ``i`` at ``open_ch``, return the index just past its match.

    Returns -1 when unbalanced (truncated file, macro trickery).
    """
    n = len(text)
    depth = 0
    j = i
    while j < n:
        if mask[j] == CODE:
            c = text[j]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return j + 1
        j += 1
    return -1


def ident_before(text: str, mask: bytearray, i: int) -> Optional[Tuple[int, int]]:
    """Identifier immediately preceding position ``i`` (skipping whitespace)."""
    j = prev_code(text, mask, i - 1)
    if j < 0 or not (text[j].isalnum() or text[j] == '_'):
        return None
    end = j + 1
    start = j
    while start > 0 and mask[start - 1] == CODE and \
            (text[start - 1].isalnum() or text[start - 1] == '_'):
        start -= 1
    if text[start].isdigit():
        return None
    return start, end


def qualifier_before(text: str, mask: bytearray, i: int) -> str:
    """C++ ``A::B::`` prefix immediately before position ``i``."""
    parts: List[str] = []
    j = i
    while True:
        k = prev_code(text, mask, j - 1)
        if k < 1 or text[k] != ':' or text[k - 1] != ':':
            break
        ident = ident_before(text, mask, k - 1)
        if ident is None:
            break
        parts.append(text[ident[0]:ident[1]])
        j = ident[0]
    return "::".join(reversed(parts))


def iter_identifiers(text: str, mask: bytearray,
                     start: int = 0,
                     end: Optional[int] = None) -> Iterator[Tuple[int, int, str]]:
    """Yield (start, end, name) for every identifier in a code range."""
    n = len(text) if end is None else end
    i = start
    while i < n:
        if mask[i] != CODE or not (text[i].isalpha() or text[i] == '_'):
            i += 1
            continue
        j = i
        while j < n and mask[j] == CODE and (text[j].isalnum() or text[j] == '_'):
            j += 1
        yield i, j, text[i:j]
        i = j


def split_top_level(text: str, sep: str = ',') -> List[str]:
    """Split on ``sep`` occurrences that are not nested in (), [], {} or <>."""
    mask = code_mask(text)
    parts: List[str] = []
    depth = 0
    angle = 0
    last = 0
    for i, ch in enumerate(text):
        if mask[i] != CODE:
            continue
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif ch == '<':
            angle += 1
        elif ch == '>' and angle > 0:
            angle -= 1
        elif ch == sep and depth == 0 and angle == 0:
            parts.append(text[last:i])
            last = i + 1
    parts.append(text[last:])
    return parts


def substitute_identifiers(text: str, mapping: Dict[str, str]) -> str:
    """Rename identifiers in ``text``, never touching strings or comments."""
    if not mapping:
        return text
    mask = code_mask(text)
    out: List[str] = []
    pos = 0
    for s, e, name in iter_identifiers(text, mask):
        if name in mapping:
            out.append(text[pos:s])
            out.append(mapping[name])
            pos = e
    out.append(text[pos:])
    return "".join(out)


def find_return_statements(text: str, mask: bytearray) -> List[Tuple[int, int, bool]]:
    """Locate ``return`` keywords.

    Returns (kw_start, kw_end, has_expression) triples. ``has_expression`` is
    False for a bare ``return;``.
    """
    found: List[Tuple[int, int, bool]] = []
    for s, e, name in iter_identifiers(text, mask):
        if name != "return":
            continue
        nxt = next_code(text, mask, e)
        has_expr = nxt < len(text) and text[nxt] != ';'
        found.append((s, e, has_expr))
    return found


def contains_word(text: str, mask: bytearray, words) -> bool:
    wanted = set(words)
    for _s, _e, name in iter_identifiers(text, mask):
        if name in wanted:
            return True
    return False


def has_label(text: str, mask: bytearray) -> bool:
    """True when the code contains a goto label (excluding case/default/``::``)."""
    for s, e, name in iter_identifiers(text, mask):
        if name in ("case", "default", "public", "private", "protected"):
            continue
        nxt = next_code(text, mask, e)
        if nxt >= len(text) or text[nxt] != ':':
            continue
        if nxt + 1 < len(text) and text[nxt + 1] == ':':
            continue
        prev = prev_code(text, mask, s - 1)
        if prev >= 0 and text[prev] not in ';{}:':
            continue
        # ternary "a ? b : c" — the ':' is preceded by an expression, and the
        # label heuristic above already required a statement boundary before.
        return True
    return False


def line_of(text: str, offset: int) -> int:
    """1-based line number of a byte offset."""
    return text.count('\n', 0, max(0, offset)) + 1
