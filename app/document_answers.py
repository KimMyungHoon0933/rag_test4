"""Parse the grounding returned by the document-answer model."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


DOCUMENT_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["source", "quote"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string"},
    },
    "required": ["citations", "answer"],
    "additionalProperties": False,
}

TEMPLATE_MARKER = re.compile(r"(?:^|\n)\s*#{2,}\s*(?:Instruction|Input|Response)\b|\[/?INST\]", re.I)


@dataclass(frozen=True)
class Citation:
    document: Document
    quote: str


@dataclass(frozen=True)
class GroundedReply:
    answer: str
    citations: tuple[Citation, ...]


REPLY_PREFIX = re.compile(
    r"^(?:(?:\[(?:assistant|human|out|solut)\]\s*:?|(?:답변|정답|assistant|answer)\s*[:：])\s*)+", re.I,
)


def clean_reply(text: str) -> str:
    answer = TEMPLATE_MARKER.split(text.strip(), maxsplit=1)[0].strip()
    return REPLY_PREFIX.sub('', answer).strip()


def matching_excerpt(quote: str, text: str) -> str | None:
    content = " ".join(text.split())
    if quote in content:
        return quote
    # Extract the original span when PDF spacing and punctuation differ.
    target = "".join(char for char in quote if char.isalnum())
    if len(target) < 5:
        return None
    indexed = [(index, char) for index, char in enumerate(content) if char.isalnum()]
    match = "".join(char for _, char in indexed).find(target)
    if match < 0:
        return None
    start, end = indexed[match][0], indexed[match + len(target) - 1][0]
    return content[start:end + 1]


def decode_document_reply(text: str) -> dict | None:
    if text.lstrip().startswith("\x60\x60\x60"):
        text = re.sub(r"^\s*\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60\s*$", "", text, flags=re.I)
    try:
        reply = json.loads(text)
    except (TypeError, ValueError):
        return None
    return reply if isinstance(reply, dict) else None


def document_reply_abstained(text: str) -> bool:
    reply = decode_document_reply(text)
    if reply is None:
        return False
    answer = reply.get("answer")
    return isinstance(answer, str) and not answer.strip() and reply.get("citations") == []


FULL_DATE = re.compile(
    r"(?<!\d)(\d{4})\s*(?:년|[./-])\s*(\d{1,2})\s*(?:월|[./-])\s*(\d{1,2})(?!\d)"
)


def date_values(text: str) -> set[tuple[int, int, int]]:
    return {
        (int(year), int(month), int(day))
        for year, month, day in FULL_DATE.findall(text)
    }


def document_dates(documents: list[Document]) -> set[tuple[int, int, int]]:
    return {date for document in documents for date in date_values(document.page_content)}


def parse_grounded_reply(text: str, documents: list[Document]) -> GroundedReply | None:
    reply = decode_document_reply(text)
    if reply is None:
        return None

    answer, entries = reply.get("answer"), reply.get("citations")
    if not isinstance(answer, str) or not isinstance(entries, list) or not 1 <= len(entries) <= 3:
        return None
    answer = clean_reply(answer)
    if not answer or date_values(answer) - document_dates(documents):
        return None

    citations = []
    for item in entries:
        if not isinstance(item, dict):
            return None
        source, quote = item.get("source"), item.get("quote")
        if type(source) is not int or not 1 <= source <= len(documents) or not isinstance(quote, str):
            return None
        document = documents[source - 1]
        quote = " ".join(quote.split())
        if not 2 <= len(quote) <= 240:
            return None
        excerpt = matching_excerpt(quote, document.page_content)
        if excerpt is None:
            return None
        citations.append(Citation(document, excerpt))

    return GroundedReply(answer, tuple(citations))


INTRODUCTION = re.compile(r"(?<![가-힣])([가-힣]{2,5}?)[은는이가]\s+누구")
FOLLOW_UP = re.compile(r"^(?:그\s*(?:사람|분)|해당\s*인물)(?:의)?\s*")
IMPLIED_SUBJECT = re.compile(r"^(?:가족|소속|전공|취미|경력|연락처)(?:[은는이가을를]|\s|[?!.]|$)")


ABOUT_PERSON = re.compile(r"^([가-힣]{2,5})에\s*대해")
NAME_SUFFIX_TYPO = re.compile(r"^([가-힣]{2,5})으(?=\s)")
NAME_TOKEN = re.compile(r"(?<![가-힣])[가-힣]{2,5}(?![가-힣])")


def resolve_document_question(
    question: str, messages: list[dict], source_names: tuple[str, ...] = (),
) -> str:
    last_question = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"), "",
    )
    subject = INTRODUCTION.search(last_question) or ABOUT_PERSON.match(last_question)
    typo = NAME_SUFFIX_TYPO.match(question)
    if typo:
        entered = typo[1]
        candidates = {name for filename in source_names for name in NAME_TOKEN.findall(Path(filename).stem)}
        if subject:
            candidates.add(subject[1])
        if entered in candidates:
            return entered + "의" + question[typo.end():]
        # A missed final consonant changes only the last Hangul syllable's coda.
        matches = {
            name for name in candidates
            if name[:-1] == entered[:-1]
            and (ord(name[-1]) - 0xAC00) // 28 == (ord(entered[-1]) - 0xAC00) // 28
        }
        if len(matches) == 1:
            return matches.pop() + "의" + question[typo.end():]
        return question

    if not subject:
        return question
    if FOLLOW_UP.match(question):
        return FOLLOW_UP.sub(f"{subject[1]}의 ", question, count=1)
    if IMPLIED_SUBJECT.match(question):
        return f"{subject[1]}의 {question}"
    return question


TITLE_TOKEN = re.compile(r"[가-힣]{3,}|[a-z][a-z0-9-]{2,}")


def select_document_scope(question: str, documents: list[Document]) -> list[Document]:
    """Keep sources explicitly named in the question when retrieval found them."""
    question_words = TITLE_TOKEN.findall(question.casefold())
    focused = []
    for document in documents:
        file_title = Path(document.metadata.get("source", "")).stem.casefold()
        title_words = TITLE_TOKEN.findall(file_title)
        if any(word.startswith(title) for title in title_words for word in question_words):
            focused.append(document)
    return focused
