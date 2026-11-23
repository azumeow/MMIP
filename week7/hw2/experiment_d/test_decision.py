"""以人工合成分數測試介接規則，不讀正式測試集調參。"""
import copy
import json
import unittest
from classifier.common import ROOT
from .decision import questions,validate_choices,assemble,extract_slots


class AdapterTests(unittest.TestCase):
    """保護機率驗證、嚴格拒判邊界及五欄 slots 契約。"""
    def setUp(self):
        self.scheme=json.loads((ROOT/'hw1/scheme.json').read_text(encoding='utf-8'))
        self.questions=questions(self.scheme)

    def response(self,maximum=.8):
        """建立合法七路由、十三意圖機率；不是模型推論結果。"""
        answers={}
        for key,question in self.questions.items():
            winner='place_order' if key=='route' else 'cancel_order'
            p={name:(maximum if name==winner else (1-maximum)/(len(question['criteria'])-1)) for name in question['criteria']}
            answers[key]={'type':'choice','choice':winner,'probabilities':p}
        return {'answers':answers}

    def test_strict_boundary(self):
        for probability,route in [(.6,'clarification'),(.60001,'place_order')]:
            answers=validate_choices(self.response(probability),self.questions)
            router,_=assemble('Cancel order #1234',answers,self.scheme,.6)
            self.assertEqual(router['route'],route)
            self.assertEqual(len(router['extracted_slots']),5)
            if route=='place_order':
                self.assertEqual(router['next_action'],'request_cancellation_confirmation')
                self.assertEqual(router['extracted_slots']['order_id'],'1234')
            else:self.assertTrue(all(v is None for v in router['extracted_slots'].values()))

    def test_reject_invalid_probabilities(self):
        response=self.response()
        variants=[]
        bad=copy.deepcopy(response);bad['answers']['route']['probabilities'].pop('returns');variants.append(bad)
        bad=copy.deepcopy(response);bad['answers']['route']['probabilities']['returns']=float('nan');variants.append(bad)
        bad=copy.deepcopy(response);bad['answers']['route']['choice']='returns';variants.append(bad)
        bad=copy.deepcopy(response);bad['answers']['route']['probabilities']['place_order']=.9;variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):validate_choices(bad,self.questions)

    def test_no_invented_slots(self):
        slots=extract_slots('Please refund my 50 dollars.','request_return')
        self.assertTrue(all(v is None for v in slots.values()))
        self.assertEqual(set(self.questions['route']['criteria']),set(self.scheme['router']))
        self.assertNotIn('expected_route',json.dumps(self.questions))


if __name__=='__main__':
    unittest.main()
