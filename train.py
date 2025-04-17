import torch 
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevel 
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path



def get_or_build_tokenizer(config,dataset,lang):
    # config['tokenizer_file] = ' ../tokenizers/toknizer_{0}.json'
    tokenizer=Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer):
        tokenizer=Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer=Whitespace()
        trainer= WordLevelTrainer(special_tokens=['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency=2)




