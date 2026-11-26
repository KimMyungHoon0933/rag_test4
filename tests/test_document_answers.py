import json
import unittest

from langchain_core.documents import Document

from app.document_answers import (
    clean_reply, document_reply_abstained, parse_grounded_reply, resolve_document_question,
    select_document_scope,
)



class GroundedReplyTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(page_content="연구실은 오후 7시에 닫습니다.", metadata={"page": 1}),
            Document(page_content="가족관계\n부모님과 형", metadata={"page": 2}),
        ]

    def parse(self, answer, citations):
        return parse_grounded_reply(json.dumps({
            "answer": answer,
            "citations": citations,
        }), self.documents)

    def test_natural_answer_and_verbatim_source(self):
        answer = self.parse(
            "가족으로 부모님과 형이 있습니다.",
            [{"source": 2, "quote": "가족관계 부모님과 형"}],
        )
        self.assertEqual(answer.answer, "가족으로 부모님과 형이 있습니다.")
        self.assertEqual(answer.citations[0].document.metadata["page"], 2)

    def test_no_evidence_falls_through_to_general_chat(self):
        self.assertIsNone(self.parse("서울입니다.", []))
        self.assertIsNone(self.parse(
            "서울입니다.", [{"source": 1, "quote": "서울"}],
        ))

    def test_wrong_source_and_missing_source_are_rejected(self):
        for source in (True, 0, 1, 3):
            with self.subTest(source=source):
                self.assertIsNone(self.parse(
                    "가족으로 부모님과 형이 있습니다.",
                    [{"source": source, "quote": "부모님과 형"}],
                ))

    def test_all_citations_must_be_valid(self):
        self.assertIsNone(self.parse("답변입니다.", [
            {"source": 2, "quote": "부모님과 형"},
            {"source": 1, "quote": "서울"},
        ]))

    def test_template_artifact_is_removed(self):
        self.assertEqual(clean_reply("서울입니다.\n### Response is:"), "서울입니다.")
        self.assertEqual(clean_reply("### Response is:"), "")



    def test_citation_tolerates_pdf_spacing_and_punctuation(self):
        result = self.parse("가족으로 부모님과 형이 있습니다.", [
            {"source": 2, "quote": "가족 관계 : 부모님과 형"},
        ])
        self.assertEqual(result.citations[0].quote, "가족관계 부모님과 형")

    def test_follow_up_keeps_the_person_from_the_previous_question(self):
        history = [{"role": "user", "content": "이하나는 누구야?"}]
        self.assertEqual(
            resolve_document_question("가족은 누가 있어?", history),
            "이하나의 가족은 누가 있어?",
        )
        self.assertEqual(
            resolve_document_question("대한민국의 수도는?", history),
            "대한민국의 수도는?",
        )


    def test_family_name_typo_resolves_from_previous_introduction(self):
        history = [{"role": "user", "content": "박성훈에 대해서 알려줘"}]
        for entered in ("박성훈으", "박성후으"):
            with self.subTest(entered=entered):
                self.assertEqual(
                    resolve_document_question(f"{entered} 가족관계는 어떻게 돼?", history),
                    "박성훈의 가족관계는 어떻게 돼?",
                )

    def test_family_name_typo_resolves_from_active_file_name(self):
        source_names = ("학생_박성훈_기록.pdf",)
        question = resolve_document_question("박성후으 가족관계는?", [], source_names)
        self.assertEqual(question, "박성훈의 가족관계는?")

    def test_ambiguous_name_typo_is_left_for_retrieval(self):
        question = "박성후으 가족관계는?"
        names = ("기록_박성훈.pdf", "기록_박성훌.pdf")
        self.assertEqual(resolve_document_question(question, [], names), question)




    def test_strips_out_prefix(self):
        self.assertEqual(clean_reply("[OUT] 서울입니다."), "서울입니다.")


    def test_scope_uses_all_chunks_of_the_named_source(self):
        profile = Document(page_content="프로필 첫 페이지", metadata={"source": "학생_박성훈_프로필.pdf"})
        family = Document(page_content="프로필 두 번째 페이지", metadata=profile.metadata)
        school = Document(page_content="학교 안내", metadata={"source": "가상대학교 정보.pdf"})
        documents = [school, profile, family]
        self.assertEqual(select_document_scope("박성훈에 대해서 알려줘", documents), [profile, family])
        self.assertEqual(select_document_scope("가상대학교가 어디야?", documents), [school])
        self.assertEqual(select_document_scope("일본의 수도는?", documents), [])

    def test_empty_reply_is_distinct_from_invalid_or_uncited_reply(self):
        self.assertTrue(document_reply_abstained('{"answer":"","citations":[]}'))
        self.assertFalse(document_reply_abstained('{"answer":"서울입니다.","citations":[]}'))
        self.assertFalse(document_reply_abstained("unfinished JSON"))


    def test_answer_date_must_match_the_source_regardless_of_format(self):
        docs = [Document(page_content="행사 날짜: 2028/06/03")]
        for answer, valid in (
            ("행사는 2028년 6월 3일입니다.", True),
            ("행사는 2028-06-03입니다.", True),
            ("행사는 1988년 6월 3일입니다.", False),
            ("행사는 2028년 6월 13일입니다.", False),
        ):
            with self.subTest(answer=answer):
                response = json.dumps({
                    "answer": answer,
                    "citations": [{"source": 1, "quote": "2028/06/03"}],
                })
                self.assertEqual(parse_grounded_reply(response, docs) is not None, valid)
