import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens  = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer        

def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split = 'train')

    # Build Tokenizer
    src_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    target_tokenizer = get_or_build_tokenizer(config, dataset_raw, config['lang_target'])

    # Keep 90% for training and 10% for validation
    train_dataset_size = int(len(dataset_raw) * 0.9)
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, src_tokenizer, target_tokenizer, config['lang_src'], config['lang_target'], config['seq_length'])
    val_dataset = BilingualDataset(val_dataset_raw, src_tokenizer, target_tokenizer, config['lang_src'], config['lang_target'], config['seq_length'])

    max_len_src = 0
    max_len_target = 0

    for item in dataset_raw:
        src_ids = src_tokenizer.encode(item['translation'][config['lang_src']]).ids
        target_ids = target_tokenizer.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f"Max length of source text: {max_len_src}")
    print(f"Max length of target text: {max_len_target}")


    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False)

    return train_dataloader, val_dataloader, src_tokenizer, target_tokenizer


def get_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_length'], config['seq_length'], d_model = config['d_model'])
    return model


def train_model(config):
    #define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, src_tokenizer, target_tokenizer = get_dataset(config)

    model = get_model(config, src_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

    #Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloaing model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index= src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing Epoch {epoch}", total = len(train_dataloader))
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Run the transformer through thr transformer

            encoder_output = model.encode(encoder_input, encoder_mask) # (Batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (Batch, seq_len, d_model)
            proj_output = model.project(decoder_output) # (Batch, seq_len, target_vocab_size)

            label = batch['label'].to(device) # (Batch, seq_len)

            # (Batch, seq_len, target_vocab_size) ---> (Batch * seq_len, target_vocab_size)
            loss = loss_fn(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})

            #Log the loss
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.flush()

            #Backpropagation
            loss.backward()

            #update the optimizer
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save the model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_filename)


if __name__ == '__main__':
    config = get_config()
    train_model(config)










