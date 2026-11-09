"""Meaningful algorithm and data-isolation checks, without model downloads."""
import unittest
import torch
from hw3.LLM.common import ROOT,read_json
from hw3.LLM.ppo_math import gae,clipped_loss


class PPOTests(unittest.TestCase):
    def test_gae_terminal_and_returns(self):
        adv,returns=gae(torch.tensor([0.,0.,1.]),torch.zeros(3),gamma=1,lam=1)
        torch.testing.assert_close(adv,torch.ones(3));torch.testing.assert_close(returns,torch.ones(3))
        adv,ret=gae(torch.tensor([1.]),torch.tensor([.4]))
        torch.testing.assert_close(adv,torch.tensor([.6]));torch.testing.assert_close(ret,torch.tensor([1.]))

    def test_clipped_policy_both_advantage_signs(self):
        # Positive advantage cannot improve objective beyond ratio 1.2.
        lp=torch.log(torch.tensor([2.],requires_grad=True));zero=torch.zeros(1)
        loss,stats=clipped_loss(lp,zero,zero,zero,torch.ones(1),zero,value_coef=0)
        self.assertAlmostEqual(loss.item(),-1.2,places=5);self.assertEqual(stats['clip_fraction'],1)
        # Negative advantage cannot improve objective beyond ratio 0.8.
        loss,_=clipped_loss(torch.log(torch.tensor([.1])),zero,zero,zero,-torch.ones(1),zero,value_coef=0)
        self.assertAlmostEqual(loss.item(),.8,places=5)

    def test_policy_gradient_is_not_sft(self):
        for advantage,expected_sign in [(1.,-1.),(-1.,1.)]:
            lp=torch.zeros(1,requires_grad=True);z=torch.zeros(1)
            loss,_=clipped_loss(lp,z,z,z,torch.tensor([advantage]),z,value_coef=0)
            loss.backward();self.assertEqual(torch.sign(lp.grad).item(),expected_sign)

    def test_value_clipping(self):
        z=torch.zeros(1)
        _,stats=clipped_loss(z,torch.ones(1),z,z,z,torch.ones(1))
        self.assertAlmostEqual(stats['value_loss'],.32,places=5)

    def test_data_sizes_and_no_train_test_leakage(self):
        seen_posts=set();seen_prompts=set();seen_answers=set()
        for filename,count in [('train_preferences_100.json',100),('reward_validation_20.json',20),('target_test_50.json',50)]:
            rows=read_json(ROOT/'data'/filename)['examples'];self.assertEqual(len(rows),count)
            for r in rows:
                self.assertNotIn(r['source']['post_id'],seen_posts);seen_posts.add(r['source']['post_id'])
                self.assertNotIn(r['prompt'],seen_prompts);seen_prompts.add(r['prompt'])
                for key in ['chosen','rejected']:
                    self.assertNotIn(r[key],seen_answers);seen_answers.add(r[key])
                expected_a=r['source']['score_A']>r['source']['score_B']
                self.assertEqual(bool(r['source']['original_label']),expected_a)
                self.assertGreaterEqual(r['source']['score_ratio'],2)
        rows=read_json(ROOT/'data/general_benchmark_100.json')['examples']
        self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        for r in rows:self.assertNotIn(r['prompt'],seen_prompts)
        self.assertEqual({c:sum(r['category']==c for r in rows) for c in {r['category'] for r in rows}},
                         {'ARC-Easy':25,'ARC-Challenge':25,'BoolQ':25,'WinoGrande':25})

    def test_unambiguous_content_choice(self):
        from hw3.LLM.report import content_choice
        self.assertEqual(content_choice('B. natural selection'),'B')
        self.assertEqual(content_choice('C.'),'C')
        self.assertEqual(content_choice('A'),'A')
        self.assertIsNone(content_choice('A dog is running'))
        self.assertIsNone(content_choice('A or B'))
        self.assertIsNone(content_choice(''))


if __name__=='__main__':unittest.main()
