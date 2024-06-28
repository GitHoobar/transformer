import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(nn.Module):
    def __init__(self, dataset, src_tokenizer, target_tokenizer, src_lang, target_lang, seq_length):
        super().__init__()

        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.target_tokenizer = target_tokenizer
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_length = seq_length

        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype = torch.long)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype = torch.long)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype = torch.long)
        
    def __len__(self):
        return len(self.dataset)
       
    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        enc_input_tokens = self.src_tokenizer.encode(src_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        enc_num_padding_tokens = self.seq_length - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_length - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sequence length is too long")
        
        # Add SOS and EOS to source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.long)
            ]
        )
        
        # add SOS to decoder
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.long),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.long)
            ]
        )
        
        # add EOS to the label ( what we expect as output from the decoder )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.long),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.long)
            ]
        )

        assert encoder_input.size(0) == self.seq_length
        assert decoder_input.size(0) == self.seq_length
        assert label.size(0) == self.seq_length

        return {
            'encoder_input': encoder_input, # (seq_len)
            'decoder_input': decoder_input, # (seq_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) &  (1, seq_len, seq_len)
            'label': label, # (seq_len)
            'src_text': src_text,
            'target_text': target_text
        }


def causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1)
    return mask == 0
