import torch 
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # Source-side special tokens
        self.sos_src = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_src = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_src = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

        # Target-side special tokens
        self.sos_tgt = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_tgt = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_tgt = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
         
        src_target_pair = self.ds[index] 
        src_txt = src_target_pair['translation'][self.src_lang]
        tgt_txt = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_txt).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_txt).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('sentence is too long')
        
        # Encoder input (SRC specials)
        encoder_input = torch.cat([
            self.sos_src,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_src,
            torch.tensor([self.pad_src.item()] * enc_num_padding_tokens, dtype=torch.int64)
        ])

        # Decoder input (TGT specials)
        decoder_input = torch.cat([
            self.sos_tgt,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_tgt.item()] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        # Label (TGT specials)
        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_tgt,
            torch.tensor([self.pad_tgt.item()] * dec_num_padding_tokens, dtype=torch.int64)
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # Masks (use the right PAD for each side)
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_src).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_tgt).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label': label,
            'src_txt': src_txt,
            'tgt_txt': tgt_txt
        }  
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0