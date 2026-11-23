"""評分器的必要檢查：格式成功不能冒充分類成功，且不偷偷修復 JSON。"""
import json
import unittest

from .evaluation import score_output
from .prompt import ROOT, make_system_prompt
from .schemas import RouterOutput


class EvaluationTests(unittest.TestCase):
    """只使用手寫合成輸出，不拿正式 test 答案調整程式。"""

    def setUp(self):
        self.output = {"route": "order_tracking", "intent": "check_shipping_status",
            "confidence": 0.9, "status": "ready", "extracted_slots": {
                "product_name": None, "quantity": None, "order_id": "123",
                "delivery_address": None, "customer_name": None},
            "missing_slots": [], "next_action": "generate_response"}

    def score(self, value, expected="order_tracking"):
        """將合成物件轉為原始文字，再交給同一個正式評分函式。"""
        return score_output(json.dumps(value), expected)

    def test_correct_format_and_route(self):
        self.assertTrue(self.score(self.output)["end_to_end_success"])

    def test_wrong_route_can_pass_schema(self):
        result = self.score(self.output, "returns")
        self.assertTrue(result["schema_valid"])
        self.assertFalse(result["route_correct"])
        self.assertFalse(result["end_to_end_success"])

    def test_correct_route_can_fail_schema(self):
        result = self.score({**self.output, "unexpected": 1})
        self.assertTrue(result["route_correct"])
        self.assertFalse(result["schema_valid"])
        self.assertFalse(result["end_to_end_success"])

    def test_no_markdown_repair(self):
        raw = "```json\n" + json.dumps(self.output) + "\n```"
        self.assertFalse(score_output(raw, "order_tracking")["json_valid"])

    def test_strict_numbers(self):
        for confidence in ["0.9", 1.1, -0.1, float("nan")]:
            self.assertFalse(self.score({**self.output, "confidence": confidence})["schema_valid"])
        slots = {**self.output["extracted_slots"], "quantity": True}
        self.assertFalse(self.score({**self.output, "extracted_slots": slots})["schema_valid"])

    def test_duplicate_keys(self):
        self.assertFalse(score_output('{"route":"returns","route":"order_tracking"}', "order_tracking")["json_valid"])

    def test_schema_agrees_with_project(self):
        source = json.loads((ROOT / "hw1/scheme.json").read_text(encoding="utf-8"))["router_output_spec"]
        generated = RouterOutput.model_json_schema()
        self.assertEqual(set(source["required"]), set(generated["required"]))
        self.assertEqual(set(source["properties"]), set(generated["properties"]))
        for field in ["route", "status", "next_action"]:
            self.assertEqual(set(source["properties"][field]["enum"]), set(generated["properties"][field]["enum"]))

    def test_prompt_contains_spec_not_dataset_answers(self):
        prompt = make_system_prompt()
        self.assertIn("Required JSON Schema", prompt)
        self.assertNotIn("expected_route", prompt)
        self.assertNotIn("source_intent", prompt)


if __name__ == "__main__":
    unittest.main()
