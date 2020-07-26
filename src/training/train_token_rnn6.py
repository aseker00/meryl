from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import trange
from src.modeling import model_ft
from src.modeling.model_token_rnn6 import *
# from src.processing import ma
from src.processing.morph_eval import *
from src.modeling.model_token_dataset import *
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.processing.spmrl.lexicon import Lexicon
from src.processing.spmrl.treebank import Treebank


device = (0 if torch.cuda.is_available() else None)
print(device)
torch.manual_seed(1)
home_path = Path('.')
src_tokens_idx = '01'
src_lattice_idx = '10'
tb_files = {'train-hebtb.tokens': f'{home_path}/data/clean/spmrl/hebtb/train-hebtb-{src_tokens_idx}-tokens.txt',
            'train-hebtb-gold.lattices': f'{home_path}/data/clean/spmrl/hebtb/train-hebtb-{src_lattice_idx}-gold.lattices',
            'dev-hebtb.tokens': f'{home_path}/data/clean/spmrl/hebtb/dev-hebtb-{src_tokens_idx}-tokens.txt',
            'dev-hebtb-gold.lattices': f'{home_path}/data/clean/spmrl/hebtb/dev-hebtb-{src_lattice_idx}-gold.lattices',
            'test-hebtb.tokens': f'{home_path}/data/clean/spmrl/hebtb/test-hebtb-{src_tokens_idx}-tokens.txt',
            'test-hebtb-gold.lattices': f'{home_path}/data/clean/spmrl/hebtb/test-hebtb-{src_lattice_idx}-gold.lattices'}
lex_files = {'pref-lex': 'data/raw/spmrl/bgulex/bgupreflex_withdef.utf8.hr',
             'lex': 'data/clean/spmrl/bgulex/bgulex-03.hr'}
bgulex_file_path = Path(f'{home_path}/data/processed/spmrl/bgulex.pickle')
hebtb_file_path = Path(f'{home_path}/data/processed/spmrl/hebtb.pickle')
vocab_file_path = Path(f'{home_path}/data/processed/spmrl/hebtb-token-vocab/vocab.pickle')
bgulex = Lexicon.load(bgulex_file_path)
hebtb = Treebank.load(hebtb_file_path)
hebtb_partition = {'train-inf': hebtb.infused_train_sentences,
                   'dev-inf': hebtb.infused_dev_sentences,
                   'test-inf': hebtb.infused_test_sentences,
                   'dev-uninf': hebtb.uninfused_dev_sentences,
                   'test-uninf': hebtb.uninfused_test_sentences}
tb_vocab = TokenVocab.load(vocab_file_path)

# Data
train_set = get_token_dataset_partition('train-inf', home_path, tb_vocab, hebtb)
dev_inf_set = get_token_dataset_partition('dev-inf', home_path, tb_vocab, hebtb)
test_inf_set = get_token_dataset_partition('test-inf', home_path, tb_vocab, hebtb)
train_set = get_model_token_dataset_partition(home_path, train_set)
dev_inf_set = get_model_token_dataset_partition(home_path, dev_inf_set)
test_inf_set = get_model_token_dataset_partition(home_path, test_inf_set)
train_sampler = RandomSampler(train_set)
dev_inf_sampler = SequentialSampler(dev_inf_set)
test_inf_sampler = SequentialSampler(test_inf_set)
train_dataloader = DataLoader(train_set, batch_size=1, sampler=train_sampler)
dev_inf_dataloader = DataLoader(dev_inf_set, batch_size=1, sampler=dev_inf_sampler)
test_inf_dataloader = DataLoader(test_inf_set, batch_size=1, sampler=test_inf_sampler)

# Embedding
ft_token_vec_file_path = Path('data/processed/spmrl/hebtb-token-vocab/word-token.vec')


def get_model(hidden_size: int, num_layers: int, emb_dropout: float, hidden_dropout: float, rnn_dropout: float, class_dropout: float) -> TagRNN6:
    token_emb = model_ft.load_embedding_weight_matrix(home_path, ft_token_vec_file_path, tb_vocab.tokens, device)
    model = TagRNN6(token_emb, emb_dropout, hidden_size, num_layers, hidden_dropout, rnn_dropout, class_dropout, tb_vocab)
    if torch.cuda.is_available():
        model.cuda(device)
    return model


# Optimization
class ModelOptimizer:
    def __init__(self, step_every: int, optimizer: optim.Optimizer, parameters: list, max_grad_norm: float):
        self.optimizer = optimizer
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm
        self.step_every = step_every
        self.steps = 0

    def step(self, losses):
        self.steps += 1
        for loss in losses[:-1]:
            loss = loss/self.step_every
            loss.backward(retain_graph=True)
        loss = losses[-1]
        loss = loss/self.step_every
        loss.backward()
        if self.steps % self.step_every == 0:
            if self.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(parameters=self.parameters, max_norm=self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()


def get_optimizer(parameters: list, max_grad_norm: float, batch_size: int, lr: float) -> ModelOptimizer:
    adam = optim.Adam(parameters, lr=lr)
    return ModelOptimizer(batch_size, adam, parameters, max_grad_norm)


# def _ma(tokens: list, lex: Lexicon, model: TagRNN) -> nlp.Sentence:
#     lattice = ma.lattice(tokens, lex)
#     gold_lattice = morph.Lattice()
#     for token_id in lattice:
#         gold_lattice[token_id] = [morph.Analysis([], [], [])]
#     sentence = nlp.Sentence(tokens, lattice, gold_lattice)
#     new_tokens, new_pref_forms, new_host_forms, new_suff_forms, new_pref_lemmas, new_host_lemmas, new_suff_lemmas = model.vocab.update(sentence)
#     if new_tokens:
#         new_token_matrix = model_ft.get_word_vectors(home_path, sorted(new_tokens))
#         new_token_matrix = torch.tensor(new_token_matrix, dtype=torch.float, device=device)
#         model.emb.update_token_emb_(new_token_matrix)
#     return sentence
#
#
# def _md(tokens: list, lex: Lexicon, model: TagRNN) -> nlp.Sentence:
#     sent = _ma(tokens, lex, model)
#     lattice = sentence_to_tensor(sent, model.vocab)
#     pred_indices = model.decode(lattice)
#     pred_lattice = _to_pred_lattice(lattice, pred_indices, model)
#     return tensor_to_sentence(pred_lattice, model.vocab)


def print_eval(gold_sentences: list, pred_sentences: list):
    print(eval_msr(gold_sentences, pred_sentences))
    print(eval_seg_tag(gold_sentences, pred_sentences))
    print(eval_tag_feats(gold_sentences, pred_sentences))
    print(eval_seg(gold_sentences, pred_sentences))
    print(eval_tag(gold_sentences, pred_sentences))
    print(eval_feats(gold_sentences, pred_sentences))


def tensors_to_sentences(token_seq: torch.Tensor, morph_seq: torch.Tensor, model: TagRNN6) -> list:
    morph_host_lemmas_seq = torch.ones(token_seq.shape, dtype=torch.long) * model.vocab.host_lemma2id[('_', )]
    morph_suff_lemmas_seq = torch.ones(token_seq.shape, dtype=torch.long) * model.vocab.suff_lemma2id[('הוא', )]
    output_seq = torch.stack([morph_seq[0], morph_seq[0], morph_seq[1], morph_seq[2],
                              morph_seq[3], morph_host_lemmas_seq, morph_seq[4], morph_seq[5],
                              morph_seq[6], morph_suff_lemmas_seq, morph_seq[7], morph_seq[8]], dim=2)
    return to_sentences(token_seq, output_seq, model.vocab)


def process_step(batch: tuple, model: TagRNN6, optimizer: ModelOptimizer):
    batch = tuple(t.to(device) for t in batch)
    token_seq = batch[0]
    seq_lengths = batch[1]
    gold_morph_seq = batch[2][:, :, [0, 2, 3, 4, 6, 7, 8, 10, 11]]
    sorted_seq_lengths, sorted_seq_idx, sorted_token_seq, sorted_preds, sorted_golds, losses = model(token_seq, seq_lengths, gold_morph_seq)
    if optimizer is not None:
        optimizer.step(losses)
    return sorted_seq_lengths, sorted_seq_idx, sorted_token_seq, sorted_preds, sorted_golds, losses


def process(epoch: int, print_every: int, phase: str, data: DataLoader, model: TagRNN6, optimizer=None):
    total_losses, print_losses = [0] * 3, [0] * 3
    total_pred_sentences, total_gold_sentences, print_pred_sentences, print_gold_sentences = [], [], [], []
    for j, batch in enumerate(data):
        step = j + 1
        sorted_seq_lengths, sorted_seq_idx, sorted_token_seq, sorted_preds, sorted_golds, losses = process_step(batch, model, optimizer)
        for l in range(len(losses)):
            total_losses[l] += losses[l].item()
            print_losses[l] += losses[l].item()
        gold_sentences = tensors_to_sentences(sorted_token_seq.cpu(), sorted_golds.cpu(), model)
        pred_sentences = tensors_to_sentences(sorted_token_seq.cpu(), sorted_preds.cpu(), model)
        print_gold_sentences.extend(gold_sentences)
        print_pred_sentences.extend(pred_sentences)
        total_gold_sentences.extend(gold_sentences)
        total_pred_sentences.extend(pred_sentences)
        if step % print_every == 0:
            # print(f'epoch {epoch} {phase} step {step} pref form loss {print_losses[0]/print_every}')
            # print(f'epoch {epoch} {phase} step {step} pref tag loss {print_losses[0] / print_every}')
            print(f'epoch {epoch} {phase} step {step} pref feats loss {print_losses[0] / print_every}')
            # print(f'epoch {epoch} {phase} step {step} host form loss {print_losses[3] / print_every}')
            # print(f'epoch {epoch} {phase} step {step} host tag loss {print_losses[1] / print_every}')
            print(f'epoch {epoch} {phase} step {step} host feats loss {print_losses[1] / print_every}')
            # print(f'epoch {epoch} {phase} step {step} suff form loss {print_losses[5] / print_every}')
            # print(f'epoch {epoch} {phase} step {step} suff tag loss {print_losses[2] / print_every}')
            print(f'epoch {epoch} {phase} step {step} suff feats loss {print_losses[2] / print_every}')
            print_eval(print_gold_sentences, print_pred_sentences)
            for token_id in total_gold_sentences[-1].gold_lattice:
                print(f'gold: {print_gold_sentences[-1].gold_lattice[token_id]}')
                print(f'pred: {print_pred_sentences[-1].gold_lattice[token_id]}')
            print_losses = [0] * 3
            print_pred_sentences, print_gold_sentences = [], []
    # print(f'epoch {epoch} {phase} total pref form loss {total_losses[0] / print_every}')
    # print(f'epoch {epoch} {phase} total pref tag loss {total_losses[0] / print_every}')
    print(f'epoch {epoch} {phase} total pref feats loss {total_losses[0] / print_every}')
    # print(f'epoch {epoch} {phase} total host form loss {total_losses[3] / print_every}')
    # print(f'epoch {epoch} {phase} total host tag loss {total_losses[1] / print_every}')
    print(f'epoch {epoch} {phase} total host feats loss {total_losses[1] / print_every}')
    # print(f'epoch {epoch} {phase} total suff form loss {total_losses[5] / print_every}')
    # print(f'epoch {epoch} {phase} total suff tag loss {total_losses[2] / print_every}')
    print(f'epoch {epoch} {phase} total suff feats loss {total_losses[2] / print_every}')
    print_eval(total_gold_sentences, total_pred_sentences)
    for token_id in total_gold_sentences[-1].gold_lattice:
        print(f'gold: {total_gold_sentences[-1].gold_lattice[token_id]}')
        print(f'pred: {total_pred_sentences[-1].gold_lattice[token_id]}')


rnn = get_model(hidden_size=32, num_layers=1, emb_dropout=0.1, hidden_dropout=0.1, rnn_dropout=0.0, class_dropout=0.0)
print(rnn)
optimizer_params = list(rnn.parameters())
lr = 1e-2
epochs = 3
print_every = 100
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    rnn.train()
    rnn_optimizer = get_optimizer(parameters=optimizer_params, max_grad_norm=5.0, batch_size=1, lr=lr)
    lr /= 10
    process(epoch, print_every, "train_infused", train_dataloader, rnn, rnn_optimizer)
    rnn.eval()
    with torch.no_grad():
        process(epoch, print_every, "dev_infused", dev_inf_dataloader, rnn)
        process(epoch, print_every, "test_infused", test_inf_dataloader, rnn)
