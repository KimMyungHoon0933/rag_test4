import ast
from io import BytesIO
from pathlib import Path
import unittest

from document_utils import extract_file, source_label


class DocumentTests(unittest.TestCase):
    def test_python_syntax(self):
        for path in Path(__file__).resolve().parents[1].rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_pdf_text_and_page(self):
        # 텍스트가 있는 최소 PDF를 별도 생성 패키지 없이 만듭니다.
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 20 250 Td (Hello RAG) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(stream)
        data = BytesIO()
        writer.write(data)
        records = extract_file('../sample.PDF', data.getvalue())
        self.assertIn('Hello RAG', records[0]['text'])
        self.assertEqual(records[0]['metadata'], {'source': 'sample.PDF', 'page': 1})

    def test_docx_paragraph_and_table(self):
        from docx import Document
        doc = Document()
        doc.add_paragraph('안녕하세요')
        doc.add_table(rows=1, cols=1).cell(0, 0).text = '등록금 안내'
        doc.add_paragraph('마지막 문단')
        data = BytesIO()
        doc.save(data)
        records = extract_file('sample.docx', data.getvalue())
        self.assertEqual([r['text'] for r in records], ['안녕하세요', '등록금 안내', '마지막 문단'])

    def test_pptx_slide(self):
        from pptx import Presentation
        doc = Presentation()
        slide = doc.slides.add_slide(doc.slide_layouts[0])
        slide.shapes.title.text = '인공지능 실습'
        data = BytesIO()
        doc.save(data)
        records = extract_file('sample.pptx', data.getvalue())
        self.assertIn('인공지능 실습', records[0]['text'])
        self.assertEqual(records[0]['metadata']['slide'], 1)

    def test_empty_pdf_and_unsupported(self):
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        data = BytesIO()
        writer.write(data)
        with self.assertRaisesRegex(ValueError, 'OCR'):
            extract_file('empty.pdf', data.getvalue())
        with self.assertRaises(ValueError):
            extract_file('bad.exe', b'')

    def test_source_label(self):
        self.assertEqual(source_label({'source': 'a.pdf', 'page': 3}), 'a.pdf · 3페이지')


if __name__ == '__main__':
    unittest.main()