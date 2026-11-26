"""업로드 파일을 디스크에 저장하지 않고 텍스트와 출처를 추출합니다."""
from io import BytesIO
from pathlib import Path


def extract_file(name: str, data: bytes) -> list[dict]:
    suffix = Path(name).suffix.lower()
    source = name.replace("\\", "/").rsplit("/", 1)[-1]
    records = []

    def add(text, **metadata):
        if text.strip():
            records.append({"text": text.strip(), "metadata": {"source": source, **metadata}})

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"{source}: 암호를 해제한 PDF를 업로드하세요.")
        for page_number, page in enumerate(reader.pages, 1):
            add(page.extract_text() or "", page=page_number)
    elif suffix == ".docx":
        from docx import Document
        document = Document(BytesIO(data))
        # 문단과 표를 문서에 등장하는 순서로 읽습니다.
        from docx.table import Table
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                add("\n".join(" | ".join(cell.text for cell in row.cells) for row in block.rows))
            else:
                add(block.text)
    elif suffix == ".pptx":
        from pptx import Presentation
        for slide_number, slide in enumerate(Presentation(BytesIO(data)).slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text)
                if shape.has_table:
                    texts.extend(" | ".join(cell.text for cell in row.cells) for row in shape.table.rows)
            add("\n".join(texts), slide=slide_number)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
    if not records:
        raise ValueError(f"{source}: 추출할 텍스트가 없습니다. 이미지/스캔 문서는 OCR이 필요합니다.")
    return records


def source_label(metadata: dict) -> str:
    label = metadata.get("source", "문서")
    if "page" in metadata:
        label += f" · {metadata['page']}페이지"
    if "slide" in metadata:
        label += f" · 슬라이드 {metadata['slide']}"
    return label
