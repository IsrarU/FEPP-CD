"""A small local WordPiece tokenizer for offline smoke tests only."""

from __future__ import annotations

import string

from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece
from transformers import PreTrainedTokenizerFast

_WORDS = """i you he she it we they the a an and or but not no is are was were be
been am do does did have has had this that those these all any some to of in on
for with at by from as about stupid idiot hate kill ugly dumb loser fat shut up
go away love great good nice thanks friend school girl boy women men gay muslim
christian jew black white people bully bullying harassment rape joke funny lol
my day game today job everyone see meet what likes nobody yourself dear""".split()


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    chars = list(string.ascii_lowercase + string.digits)
    vocab_list = specials + sorted(set(_WORDS)) + chars + ["##" + c for c in chars]
    vocab = {w: i for i, w in enumerate(vocab_list)}
    tk = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
