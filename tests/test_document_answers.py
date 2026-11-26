import json
import unittest

from langchain_core.documents import Document

from app.document_answers import (
    clean_reply, concise_profile_reply, parse_grounded_reply, recover_document_reply,
    requires_document_family_answer, resolve_document_question,
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


    def test_broad_introduction_can_use_a_verified_role(self):
        document = Document(page_content="이하나는 예시대학교 컴퓨터공학과에 재학 중입니다.")
        raw = "이하나는 예시대학교 컴퓨터공학과에 재학 중입니다."
        current = parse_grounded_reply(json.dumps({
            "answer": raw, "citations": [{"source": 1, "quote": raw}],
        }), [document])
        reply = concise_profile_reply("이하나는 누구야?", current, [document])
        self.assertEqual(reply.answer, "이하나는 컴퓨터공학과에 재학 중입니다.")
        self.assertEqual(reply.citations[0].quote, "컴퓨터공학과")

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

    def test_private_family_recovery_requires_the_named_source(self):
        documents = [
            Document(page_content="이하나", metadata={"source": "profile.pdf", "page": 1}),
            Document(page_content="가족관계\n부모님과 형\n테스트 이메일", metadata={"source": "profile.pdf", "page": 2}),
        ]
        self.assertEqual(
            recover_document_reply("이하나의 가족은 누구야?", documents).answer,
            "문서에 따르면 가족은 부모님과 형입니다.",
        )
        self.assertIsNone(recover_document_reply("김하늘의 가족은 누구야?", documents))


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
        self.assertTrue(requires_document_family_answer(question, [], source_names))

    def test_unresolved_family_typo_stays_in_document_path(self):
        question = "박성후으 가족관계는?"
        names = ("기록_박성훈.pdf", "기록_박성훌.pdf")
        self.assertEqual(resolve_document_question(question, [], names), question)
        self.assertTrue(requires_document_family_answer(question, [], names))
        self.assertFalse(requires_document_family_answer("일본의 수도는?", [], names))
