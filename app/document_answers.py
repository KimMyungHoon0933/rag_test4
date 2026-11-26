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
    r"^(?:(?:\[(?:assistant|human|solut)\]|(?:답변|정답|assistant|answer)\s*[:：])\s*)+", re.I,
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


def parse_grounded_reply(text: str, documents: list[Document]) -> GroundedReply | None:
    if text.lstrip().startswith("\x60\x60\x60"):
        text = re.sub(r"^\s*\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60\s*$", "", text, flags=re.I)
    try:
        reply = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(reply, dict):
        return None

    answer, entries = reply.get("answer"), reply.get("citations")
    if not isinstance(answer, str) or not isinstance(entries, list) or not 1 <= len(entries) <= 3:
        return None
    answer = clean_reply(answer)
    if not answer:
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
EXTRA_PROFILE_DETAILS = re.compile(r"대학교|학사\s*과정|생년|[12]\d{3}[년./-]|남자|여자|남성|여성|@")
ACADEMIC_DEPARTMENT = re.compile(r"(?<!\w)[가-힣A-Za-z]+?학과(?:(?!\w)|(?=(?:에서|에|의|로)\b))")


def concise_profile_reply(
    question: str, reply: GroundedReply | None, documents: list[Document],
) -> GroundedReply | None:
    """Limit a broad introduction to a verified role if the draft included extra details."""
    subject = INTRODUCTION.search(question)
    if not subject or (reply and not EXTRA_PROFILE_DETAILS.search(reply.answer)):
        return reply

    roles = {}
    for document in documents:
        text = document.page_content
        if subject[1] not in text or "재학" not in text:
            continue
        for department in set(ACADEMIC_DEPARTMENT.findall(text)):
            roles[department] = document

    if len(roles) != 1:
        return reply
    department, document = next(iter(roles.items()))
    name = subject[1]
    topic = "은" if (ord(name[-1]) - ord("가")) % 28 else "는"
    return GroundedReply(
        f"{name}{topic} {department}에 재학 중입니다.",
        (Citation(document, department),),
    )


FOLLOW_UP = re.compile(r"^(?:그\s*(?:사람|분)|해당\s*인물)(?:의)?\s*")
IMPLIED_SUBJECT = re.compile(r"^(?:가족|소속|전공|취미|경력|연락처)(?:[은는이가을를]|\s|[?!.]|$)")


ABOUT_PERSON = re.compile(r"^([가-힣]{2,5})에\s*대해")
FAMILY_NAME_TYPO = re.compile(r"^([가-힣]{2,5})으(?=\s+가족)")
NAME_TOKEN = re.compile(r"(?<![가-힣])[가-힣]{2,5}(?![가-힣])")


def resolve_document_question(
    question: str, messages: list[dict], source_names: tuple[str, ...] = (),
) -> str:
    last_question = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"), "",
    )
    subject = INTRODUCTION.search(last_question) or ABOUT_PERSON.match(last_question)
    typo = FAMILY_NAME_TYPO.match(question)
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


FAMILY_MEMBER_QUESTION = re.compile(
    r"(?<![가-힣])([가-힣]{2,5}?)의?\s+가족(?:\s*(?:관계|구성원))?(?:은|는|이|가)?(?:\s|[?!.]|$)"
)
FAMILY_LINE = re.compile(r"(?m)^[ \t]*가족\s*관계[ \t]*[:：]?[ \t]*(?:\r?\n[ \t]*)?([^\r\n]+)")


def requires_document_family_answer(
    question: str, documents: list[Document], source_names: tuple[str, ...],
) -> bool:
    if FAMILY_NAME_TYPO.match(question):
        return True
    subject = FAMILY_MEMBER_QUESTION.search(question)
    return bool(subject and (
        any(subject[1] in NAME_TOKEN.findall(Path(name).stem) for name in source_names)
        or any(subject[1] in document.page_content for document in documents)
    ))


def recover_document_reply(question: str, documents: list[Document]) -> GroundedReply | None:
    """Recover a short cited answer only when a named person's source has an explicit field."""
    subject = FAMILY_MEMBER_QUESTION.search(question)
    if not subject or any(term in question for term in ("연락처", "주소", "생년", "직업", "나이")):
        return None
    sources = {
        document.metadata.get("source") for document in documents
        if subject[1] in document.page_content or subject[1] in document.metadata.get("source", "")
    }
    matches = {}
    for document in documents:
        if document.metadata.get("source") not in sources:
            continue
        field = FAMILY_LINE.search(document.page_content)
        if not field:
            continue
        value = field[1].strip().rstrip(".")
        if not value or len(value) > 80 or "@" in value or "###" in value:
            continue
        matches[value] = document
    if len(matches) != 1:
        return None
    value, document = next(iter(matches.items()))
    return GroundedReply(f"문서에 따르면 가족은 {value}입니다.", (Citation(document, value),))
