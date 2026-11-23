"""檢查 B 的標準答案，避免把錯誤業務流程教給模型。"""
import json
import unittest

from .prepare_data import SLOTS, label_question


class TargetTests(unittest.TestCase):
    """人工合成問題，只測資料標註規則，不依正式 test 調整答案。"""

    def row(self,text,route,intent,replacements=None):
        return {"text":text,"route":route,"intent":intent,"mock_replacements":replacements or {}}

    def test_cancel_waits_for_confirmation(self):
        answer = label_question(self.row('Please cancel order 123.','place_order','cancel_order'))
        self.assertEqual(answer['status'],'awaiting_confirmation')
        self.assertEqual(answer['next_action'],'request_cancellation_confirmation')
        self.assertEqual(answer['extracted_slots']['order_id'],'123')
        self.assertEqual(set(answer['extracted_slots']),set(SLOTS))

    def test_tracking_missing_id_keeps_route(self):
        answer = label_question(self.row('Where is my order?','order_tracking','check_shipping_status'))
        self.assertEqual(answer['route'],'order_tracking')
        self.assertEqual(answer['missing_slots'],['order_id'])
        self.assertEqual(answer['status'],'missing_information')

    def test_purchase_extracts_only_explicit_quantity(self):
        answer = label_question(self.row('I want to buy two items.','place_order','purchase_product'))
        self.assertEqual(answer['extracted_slots']['quantity'],2)
        self.assertEqual(answer['missing_slots'],['product_name','delivery_address'])

    def test_refund_amount_is_not_order_number(self):
        answer = label_question(self.row('Please refund TWD 500.','returns','request_return'))
        self.assertIsNone(answer['extracted_slots']['order_id'])
        self.assertEqual(answer['missing_slots'],['order_id'])

    def test_policy_has_no_required_order_id(self):
        answer = label_question(self.row('What is your refund policy?','returns','ask_return_policy'))
        self.assertEqual(answer['missing_slots'],[])
        self.assertEqual(answer['next_action'],'generate_response')

    def test_out_of_scope_has_complete_null_slots(self):
        answer = label_question(self.row('How is the weather?','out_of_scope','unrelated_request'))
        self.assertTrue(all(value is None for value in answer['extracted_slots'].values()))
        self.assertEqual(answer['next_action'],'reject_request')

    def test_selected_logits_loss_matches_full_causal_loss(self):
        """小型數值測試：先選答案 hidden 再算 head，與完整 logits 的標準 causal loss 相同。"""
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest('此數值測試請在 .venv-b 執行')
        torch.manual_seed(7)
        hidden = torch.randn(1,10,6)
        weight = torch.randn(7,6)
        labels = torch.tensor([[-100,-100,-100,2,3,-100,4,1,5,6]])
        full_logits = F.linear(hidden,weight)
        full = F.cross_entropy(full_logits[:,:-1].reshape(-1,7),labels[:,1:].reshape(-1),ignore_index=-100)
        positions = torch.where(labels[0]!=-100)[0]
        selected = F.cross_entropy(F.linear(hidden[0,positions-1],weight),labels[0,positions])
        self.assertTrue(torch.allclose(full,selected,atol=1e-6))


if __name__ == '__main__':
    unittest.main()
