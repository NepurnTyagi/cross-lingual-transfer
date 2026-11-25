import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--read", type=str)
parser.add_argument("--write", type=str)
args=parser.parse_args()

tokenizer=Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
trainer=BpeTrainer(special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"], vocab_size=8000)
tokenizer.train_from_iterator(pd.read_csv(args.read)["translated"].astype(str).tolist(), trainer)
fast_tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", unk_token="<unk>", bos_token="<sos>", eos_token="<eos>")
print("BOS id:", fast_tokenizer.bos_token_id)
fast_tokenizer.save_pretrained(args.write)
