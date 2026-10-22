import unittest

import torch

from data import build_vocab, encode, tokenize
from models import PositionalEncoding, SeqClassifier


class DataSmokeTests(unittest.TestCase):
    def test_tokenization_vocabulary_and_encoding(self):
        self.assertEqual(tokenize("Great film, isn't it?"), ["great", "film", "isn't", "it"])

        stoi, itos = build_vocab(["great film", "great story"], vocab_size=5)
        self.assertEqual(itos[:2], ["<pad>", "<unk>"])
        self.assertEqual(encode("great unknown story", stoi, max_len=4), [2, 1, 4, 0])
        self.assertEqual(encode("great story film", stoi, max_len=2), [2, 4])


class ModelSmokeTests(unittest.TestCase):
    def test_all_mixers_run_on_cpu_and_backpropagate(self):
        ids = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 8],
        ])

        for mixer_type in ("glu", "linear_attn", "softmax_attn", "gated_fusion"):
            with self.subTest(mixer_type=mixer_type):
                model = SeqClassifier(
                    vocab_size=16,
                    mixer_type=mixer_type,
                    d_model=16,
                    n_heads=2,
                    d_ff=24,
                    n_layers=1,
                    max_len=4,
                )
                logits = model(ids)

                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all().item())
                logits.sum().backward()
                self.assertIsNotNone(model.head.weight.grad)

    def test_positional_encoding_extends_past_initial_length(self):
        encoding = PositionalEncoding(d_model=16, max_len=4)
        output = encoding(torch.zeros(2, 7, 16))

        self.assertEqual(tuple(output.shape), (2, 7, 16))
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
