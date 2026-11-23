"""檢查 C 的請求真的包含完整 Schema，且 A 的預設請求沒有受到改動。"""
import io
import json
import unittest
from unittest.mock import patch

from hw2.experiment_a.prompt import ROOT
from hw2.experiment_a.run import call_model,schema_response_format


class ConstraintTests(unittest.TestCase):
    """只模擬 HTTP 傳輸，不使用正式測試資料或模型答案調參。"""

    def setUp(self):
        self.schema=json.loads((ROOT/'hw1/scheme.json').read_text(encoding='utf-8'))['router_output_spec']
        self.messages=[{'role':'user','content':'Synthetic request.'}]

    def payload(self,schema=None):
        """截取送往本機模型 API 的實際 Request body。"""
        body=json.dumps({'choices':[{'message':{'content':'{}'}}]}).encode()
        with patch('urllib.request.urlopen',return_value=io.BytesIO(body)) as opened:
            call_model('http://127.0.0.1:8081/v1','test-model',self.messages,512,schema=schema)
            request=opened.call_args.args[0]
            return json.loads(request.data)

    def test_a_has_no_constraint(self):
        self.assertNotIn('response_format',self.payload())

    def test_c_passes_full_schema_without_changing_messages(self):
        payload=self.payload(self.schema)
        self.assertEqual(payload['messages'],self.messages)
        self.assertEqual(payload['response_format']['json_schema']['schema'],self.schema)
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertNotIn('tools',payload)

    def test_all_five_slots_still_required(self):
        schema=schema_response_format(self.schema)['json_schema']['schema']
        slots=schema['properties']['extracted_slots']
        self.assertEqual(set(slots['required']),{'product_name','quantity','order_id','delivery_address','customer_name'})
        self.assertFalse(slots['additionalProperties'])

    def test_schema_does_not_reveal_expected_route(self):
        schema=schema_response_format(self.schema)['json_schema']['schema']
        self.assertEqual(len(schema['properties']['route']['enum']),7)
        self.assertNotIn('const',schema['properties']['route'])
        self.assertNotIn('expected_route',json.dumps(schema))


if __name__=='__main__':
    unittest.main()
