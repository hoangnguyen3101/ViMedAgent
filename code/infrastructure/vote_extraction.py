"""
Vote extraction rule of the paper (Section 3.6): the option letter of an output is the
LAST explicit, non-negated statement of an answer in it.

Explicit statements recognised (Markdown emphasis and <think> blocks are ignored):
  * a declaration:      "the (correct/best/final) answer is X", "my final answer remains X",
                        "answer: X", "đáp án (đúng) là X", "I choose/maintain/select X"
  * a verdict on a letter: "option X is the correct answer / the best choice / correct"
  * a letter with its option text in an answering sentence:
                        "the most likely pathogen is B. Coxsackie A virus"
  * a line that is only a label:  "B", "B.", "(B)", "B. Coxsackie A virus", "Option B"

A statement is negated, and skipped, when it is followed by "is not / incorrect / wrong /
excluded / ruled out" or preceded by "not / not the" / "rule out" / "unlikely".
The statement that comes last decides. If it names two options ("A or B"), a letter outside
the option range, or an article ("the answer is a beta-blocker"), the vote is missing (None).
Nothing here consults the gold answer.

extract_vote(text, n_options)            letter rule only — usable as the run-time parser.
extract_vote(text, n_options, options)   additionally falls back to the option TEXT when no
                                         letter statement exists (options: list of option strings,
                                         in A, B, C, D order). Reported separately in the analysis.
"""

import re
from infrastructure.choices import choice_letters

VERSION = "last-explicit-v1"

_LABEL = r'[\(\[\"\']?((?-i:[A-Z]))[\)\]\"\']?(?!\w)'
_CUE_NOUN = (r'(?:(?:the|my|our)\s+)?(?:(?:final|correct|best|chosen|revised|updated|most\s+appropriate)\s+)*'
             r'(?:answer|response)|(?:(?:the|my|our)\s+)?(?:final|correct|best|chosen|revised|updated|most\s+appropriate)\s+'
             r'(?:choice|option)'
             # Vietnamese (VM14K): "đáp án (đúng) là", "câu trả lời đúng là", "lựa chọn đúng đắn:", "kết luận:" ...
             r'|(?:câu\s+trả\s+lời|đáp\s+án|lựa\s+chọn|phương\s+án|sự\s+lựa\s+chọn)'
             r'(?:\s+(?:cuối\s+cùng|chính\s+xác|đúng\s+đắn|đúng|tốt\s+nhất|phù\s+hợp\s+nhất|phù\s+hợp|hợp\s+lý\s+nhất|của\s+tôi))*'
             r'|(?:kết\s+luận|lời\s+kết)')
_LINK = (r'\s*(?:(?:is|would\s+be|remains?|stays?|stay|continues?\s+to\s+be|should\s+be|là|vẫn\s+là)\b\s*[:：=\-–—]?\s*'
         r'|[:：=\-–—]\s*)(?:[✅✔✓☑👉➡️]\s*)?')

# declaration: "<cue noun> (is|remains|:) X"  /  "đáp án (đúng) (là) X"
_DECL = re.compile(
    r'(?:\b(?:' + _CUE_NOUN + r')(?:\s+remains?|\s+stays?)?' + _LINK +
    r'|đáp\s+án\s*(?:đúng\s*)?(?:là\s*)?[:：=\-]?\s*|\bconclusion\b\s*[:：=\-]?\s*)'
    r'(?:option\s+|choice\s+|answer\s+)?' + _LABEL, re.IGNORECASE)
# a lowercase label is accepted after a declaration only when nothing follows it: "The answer: e"
_DECL_LOWER = re.compile(
    r'(?:\b(?:' + _CUE_NOUN + r')(?:\s+remains?|\s+stays?)?' + _LINK + r'|\bconclusion\b\s*[:：=\-]?\s*)'
    r'(?:option\s+|choice\s+)?[\(\[\"\']?((?-i:[a-e]))[\)\]\"\']?(?=\s*(?:[.!,;]|$))', re.IGNORECASE)
# first person: "I choose / maintain / select / stand by / stay with / agree with (option) X"
_FIRST_VI = re.compile(
    r'\b(?:tôi|chúng\s+tôi)\s+(?:sẽ\s+|vẫn\s+|xin\s+|quyết\s+định\s+)*(?:chọn|giữ\s+nguyên|lựa\s+chọn|đồng\s+ý\s+với|đồng\s+ý)\s+'
    r'(?:(?:đáp\s+án|phương\s+án|lựa\s+chọn|câu\s+trả\s+lời|đáp\s+án\s+là|là)\s+)*' + _LABEL, re.IGNORECASE)
_FIRST = re.compile(
    r'\b(?:I|we)\s+(?:will\s+|would\s+|still\s+|therefore\s+|thus\s+)*'
    r'(?:choose|select|pick|maintain|stand\s+by|stay\s+with|stick\s+(?:with|to)|agree\s+(?:with|that)|'
    r'update(?:\s+my\s+answer)?\s+to|change(?:\s+my\s+answer)?\s+to|revise(?:\s+my\s+answer)?\s+to)\s+'
    r'(?:(?:that|the|my)\s+)*(?:(?:answer|option|choice)\s+(?:is\s+|remains\s+|to\s+)?)?(?:option\s+|choice\s+)?'
    + _LABEL, re.IGNORECASE)
# verdict: "option X is the correct answer / the best choice / correct / right"
_VERDICT_VI = re.compile(
    r'\b(?:lựa\s+chọn|đáp\s+án|phương\s+án|câu)\s+' + _LABEL +
    r'(?:\s*\([^)]{0,80}\))?\s*(?:mới\s+)?(?:là\s+)?(?:đúng\b|chính\s+xác\b|phù\s+hợp\s+nhất|tốt\s+nhất|hợp\s+lý\s+nhất|'
    r'(?:đáp\s+án|lựa\s+chọn|câu\s+trả\s+lời)\s+(?:đúng|chính\s+xác))', re.IGNORECASE)
# a bare label as subject: "**A** would be the choice", "B is the best answer"
_VERDICT_BARE = re.compile(
    r'(?<![A-Za-z0-9])' + _LABEL + r'\s+(?:would\s+be|is|remains)\s+(?:still\s+)?(?:the\s+)?'
    r'(?:(?:single\s+)?(?:best|correct|right|most\s+(?:appropriate|likely|reasonable))\s+)?(?:choice|answer|option)\b', re.IGNORECASE)
# "B. Tuần thứ 32 is selected as the answer", "A. Image of stones seems the most likely correct answer"
_LETTER_TEXT_VERDICT = re.compile(
    r'(?<![A-Za-z0-9])' + _LABEL + r'[.)]\s+[^\n.!?]{1,140}?\s+(?:is|are|seems?|appears?|would\s+be|being)\s+'
    r'(?:selected|chosen|picked|(?:the\s+)?(?:most\s+)?(?:likely|reasonable|appropriate|plausible|best|correct|right)'
    r'(?:\s+correct)?(?:\s+(?:answer|choice|option))?\b)', re.IGNORECASE)
# Vietnamese tail: "C Trường phổi dãn rộng là câu trả lời chính xác nhất", "**D. Tất cả đúng** là đáp án đúng"
_LETTER_TEXT_VI = re.compile(
    r'(?<![A-Za-z0-9])' + _LABEL + r'[.)]?\s+[^\n.!?]{1,140}?\s+là\s+(?:câu\s+trả\s+lời|đáp\s+án|lựa\s+chọn|phương\s+án)\s+'
    r'(?:đúng|chính\s+xác|tốt\s+nhất|phù\s+hợp)', re.IGNORECASE)
# "... would be **D**, focusing on ..." (a single label after "would be / should be / is selected")
_BE_LABEL = re.compile(r'\b(?:would\s+be|should\s+be|is\s+selected|will\s+be)\s+(?:option\s+)?' + _LABEL + r'(?=\s*[,.;:]|\s*$)')
# label first: "A là lựa chọn đúng", "C là đáp án chính xác nhất"
_VERDICT_BARE_VI = re.compile(
    r'(?<![A-Za-z0-9])' + _LABEL + r'\s+là\s+(?:câu\s+trả\s+lời|đáp\s+án|lựa\s+chọn|phương\s+án)\s+'
    r'(?:đúng|chính\s+xác|tốt\s+nhất|phù\s+hợp)', re.IGNORECASE)
_VERDICT_NP = re.compile(
    r'\b(?:option|choice)\s+' + _LABEL + r'\s+(?:the|as\s+the)\s+(?:single\s+)?(?:most\s+)?'
    r'(?:correct|best|right|appropriate|plausible|likely|accurate|suitable)\b', re.IGNORECASE)
_VERDICT = re.compile(
    r'\b(?:option|choice|answer)\s+' + _LABEL + r'(?:\s*\([^)]{0,80}\))?\s*(?:is|remains|stays|would\s+be)\s+'
    r'(?:still\s+|also\s+|indeed\s+|therefore\s+)?(?:the\s+)?(?:(?:single\s+)?(?:most\s+)?(?:correct|best|right)\b|'
    r'(?:most|more\s+than\s+any)\s+(?:appropriate|plausible|likely|accurate|suitable|fitting)\b)', re.IGNORECASE)
# answering sentence + "X. text" or "X) text" : "... is B. Coxsackie A virus"
_LETTER_TEXT = re.compile(
    r'\b(?:answer|choice|option|diagnosis|most\s+likely|most\s+appropriate|best|correct|appropriate|'
    r'indicated|recommended|next\s+step|treatment|therapy|explanation|pathogen|mechanism|cause|'
    r'is|are|be|remains|choose|select|là|chọn)\b[^.\n]{0,80}?(?<![A-Za-z0-9])' + _LABEL +
    r'(?:[.)]\s+(?=[A-Z0-9(↑↓])|\s*[-–—:]\s+(?=[A-Z0-9(↑↓]))')
_LINE = re.compile(r'^\s*(?:[✅✔✓☑👉➡️]\s*)?(?:option\s+|choice\s+|answer\s*[:：]?\s*|đáp\s+án\s*[:：]?\s*)?' + _LABEL + r'\s*(?:[.:)!]|$|\s*[-–—]\s+)', re.IGNORECASE)

_NEG_AFTER = re.compile(
    r'^[\s\w\'’,()\-/]{0,60}?\b(?:is|are|was|would\s+be|seems?|appears?)\s+(?:also\s+|clearly\s+|definitely\s+)?'
    r'(?:not|incorrect|wrong|false|excluded|ruled\s+out|unlikely|inappropriate|less\s+likely)\b|'
    r'^[\s\w\'’,()\-/]{0,60}?\b(?:không\s+(?:đúng|phải|chính\s+xác|phù\s+hợp)|sai)\b|'
    r'^\s*(?:\([^)]*\)\s*)?(?:is\s+)?(?:not|incorrect|wrong)\b|^\s*(?:isn\'t|isn’t|does\s+not|doesn\'t)\b',
    re.IGNORECASE)
_HEDGE = re.compile(r'^\s*(?:và|hoặc|and|or|,)?\s*(?:[A-E]\s*)?(?:và|hoặc|and|or)?\s*(?:[A-E]\s*)?'
                    r'(?:cũng\s+có\s+thể|có\s+thể\s+(?:cũng\s+)?đúng|may\s+also|could\s+also|might\s+also|can\s+also)', re.IGNORECASE)
_NEG_BEFORE = re.compile(r'(?:\bnot(?:\s+the)?(?:\s+(?:answer|option|choice))?|\brule[sd]?\s+out|\bexclude[sd]?|'
                         r'\bunlikely|\bincorrect|\bwrong|\bneither|\bnor|\bkhông(?:\s+phải)?)\W*(?:\w+\W+){0,2}$', re.IGNORECASE)
_THINK = re.compile(r'<think>.*?(?:</think>|$)', re.DOTALL | re.IGNORECASE)


def _strip(text: str) -> str:
    text = _THINK.sub('', text or '').strip()
    return re.sub(r'[*_`]', '', text)


def _statements(plain: str) -> list[tuple[int, str]]:
    """(end position, raw letter) of every non-negated explicit statement, in text order."""
    found = {}
    for rx in (_DECL, _DECL_LOWER, _FIRST, _FIRST_VI, _VERDICT, _VERDICT_NP, _VERDICT_VI, _VERDICT_BARE, _LETTER_TEXT_VERDICT, _LETTER_TEXT_VI, _BE_LABEL, _VERDICT_BARE_VI, _LETTER_TEXT):
        for m in rx.finditer(plain):
            letter, end = m.group(1).upper(), m.end(1)
            if rx is _LETTER_TEXT and re.search(r'\b(?:option|choice)\s*\(?$', plain[max(0, m.start(1) - 12):m.start(1)], re.I):
                continue                                         # "(option D) should be considered"
            if rx is _LETTER_TEXT and letter not in 'ABCDE':   # weak pattern: "oxidation of I- to I2"
                continue
            tail = plain[end:end + 90]
            neg_tail = tail.split('\n', 1)[0]            # negation only counts on the same line
            head = plain[max(0, m.start(1) - 40):m.start(1)]
            if rx not in (_VERDICT, _VERDICT_NP, _VERDICT_VI, _VERDICT_BARE, _LETTER_TEXT_VERDICT, _LETTER_TEXT_VI, _BE_LABEL, _VERDICT_BARE_VI) and _NEG_AFTER.match(neg_tail):
                continue
            if rx in (_VERDICT, _VERDICT_NP, _VERDICT_VI, _VERDICT_BARE, _LETTER_TEXT_VERDICT, _LETTER_TEXT_VI, _BE_LABEL, _VERDICT_BARE_VI) and re.match(r'\s*(?:\([^)]*\)\s*)?(?:is\s+)?not\b', tail):
                continue
            if _HEDGE.match(neg_tail):
                continue
            if _NEG_BEFORE.search(head) and rx is not _DECL:
                continue
            # an article, not a label: "answer is A patient ..." / "answer is a beta-blocker"
            if letter == 'A' and re.match(r'\s+[a-z]', tail) and not re.match(
                    r'\s+(?:because|since|as|due|which|given|based|is|would|should|will|seems?|remains?|là|vì|nên|mới|đúng|sai|-)\b', tail):
                continue
            # "A or B", "A/B", "A, B", "A and B": unresolved, the statement exists but names two options
            two = bool(re.match(r'\s*(?:or\b|and\b|và\b|hoặc\b|/|,)\s*\(?[A-Z]\)?(?![\w])', tail))
            found[m.start(1)] = (end, letter, two, False)
    for line_start, line in _lines(plain):
        m = _LINE.match(line)
        if m and not (m.group(1) == 'A' and re.match(r'\s+[a-z]', line[m.end(1):])):
            # "B. Coxsackie" as a line; but a bare list entry among 4 consecutive option lines is not an answer
            found.setdefault(line_start + m.start(1), (line_start + m.end(1), m.group(1), False, True))
    return [(end, letter, two, from_line) for _, (end, letter, two, from_line) in sorted(found.items())]


def _lines(plain: str):
    pos = 0
    for line in plain.split('\n'):
        yield pos, line
        pos += len(line) + 1


def _is_option_listing(plain: str) -> bool:
    """Lines 'A. x' 'B. y' 'C. z' 'D. w' restated as a list: those lines are not an answer."""
    heads = {m.group(1) for m in re.finditer(r'(?m)^\s*\(?([A-E])[.)]\s+\S', plain)}
    return len(heads) >= 3


def extract_vote(text: str, n_options: int = 4, options: list[str] | None = None) -> str | None:
    letters = choice_letters(n_options)
    plain = _strip(text)
    if not plain:
        return None
    stmts = _statements(plain)
    if stmts and _is_option_listing(plain):
        # a line "B. text" that only restates the option list is not an answer; a cue statement is
        listed = {m.start(1) for m in re.finditer(r'(?m)^\s*\(?([A-E])[.)]\s+\S', plain)}
        stmts = [x for x in stmts if not (x[3] and (x[0] - 1) in listed)]
    if stmts:
        _, letter, two, _ = stmts[-1]
        if two:
            return None
        return letter if letter in letters else None
    if options:
        return _from_option_text(plain, letters, options)
    return None


def _from_option_text(plain: str, letters: str, options: list[str]) -> str | None:
    """No letter statement: the option whose text the closing sentences name (exactly one)."""
    sentences = [s for s in re.split(r'(?<=[.!?])\s+|\n+', plain) if s.strip()]
    tail = ' '.join(sentences[-2:]).lower()
    hit = []
    for letter, opt in zip(letters, options):
        o = re.sub(r'\s+', ' ', opt.strip().lower().rstrip('.'))
        if len(o) >= 3 and re.search(r'(?<!\w)' + re.escape(o) + r'(?!\w)', tail):
            hit.append(letter)
    return hit[0] if len(hit) == 1 else None
