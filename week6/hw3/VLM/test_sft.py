"""Check causal assistant-only SFT against Transformers' masked-label loss."""
import unittest
from types import SimpleNamespace

import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for the FP16 training path')
class SFTLossTests(unittest.TestCase):
    def test_selected_logits_equal_masked_labels_and_receive_gradients(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from hw3.VLM.experiment import autocast, embeddings, example_loss
        torch.manual_seed(42)
        lm = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=32,
            intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=16, attention_dropout=0.,
            attn_implementation='eager')).cuda().eval()
        model = SimpleNamespace(language_model=lm, img_context_token_id=3)
        item = {'ids': [1, 3, 3, 5, 7, 8, 2], 'prefix': 4,
                'targets': [7, 8, 2], 'visual': torch.randn(2, 32)}
        actual = example_loss(model, item, training=True)
        embedded, mask = embeddings(model, item['ids'], item['visual'], training=True)
        labels = torch.tensor([item['ids']], device='cuda')
        labels[:, :item['prefix']] = -100
        with autocast():
            expected = lm(inputs_embeds=embedded, attention_mask=mask,
                          labels=labels, use_cache=False).loss
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)
        actual.backward()
        self.assertTrue(torch.isfinite(lm.model.layers[0].self_attn.q_proj.weight.grad).all())
        self.assertGreater(lm.model.layers[0].self_attn.q_proj.weight.grad.norm().item(), 0.)
        altered = dict(item, visual=-item['visual'])
        self.assertGreater(abs(example_loss(model, altered).item() - actual.item()), 1e-5)


if __name__ == '__main__':
    unittest.main()
