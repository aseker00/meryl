import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import trange
from src.modeling import model_ft
from src.modeling.model_ptrnet import *
from src.modeling.model_morph_dataset import *
from src.processing import ma, morph
from src.processing.morph_dataset import get_morph_dataset_partition
from src.processing.morph_eval import *
from src.processing.morph_vocab import MorphVocab
from src.processing.spmrl.lexicon import Lexicon
from src.processing.spmrl.treebank import Treebank


device = (0 if torch.cuda.is_available() else None)
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
vocab_file_path = Path(f'{home_path}/data/processed/spmrl/hebtb-morph-vocab/vocab.pickle')
bgulex = Lexicon.load(bgulex_file_path)
hebtb = Treebank.load(hebtb_file_path)
hebtb_partition = {'train-inf': hebtb.infused_train_sentences,
                   'dev-inf': hebtb.infused_dev_sentences,
                   'test-inf': hebtb.infused_test_sentences,
                   'dev-uninf': hebtb.uninfused_dev_sentences,
                   'test-uninf': hebtb.uninfused_test_sentences}
tb_vocab = MorphVocab.load(vocab_file_path)

# Data
train_set = get_morph_dataset_partition('train-inf', home_path, tb_vocab, hebtb)
dev_inf_set = get_morph_dataset_partition('dev-inf', home_path, tb_vocab, hebtb)
test_inf_set = get_morph_dataset_partition('test-inf', home_path, tb_vocab, hebtb)
dev_uninf_set = get_morph_dataset_partition('dev-uninf', home_path, tb_vocab, hebtb)
test_uninf_set = get_morph_dataset_partition('test-uninf', home_path, tb_vocab, hebtb)
train_set = get_model_morpheme_dataset_partition(home_path, train_set)
dev_inf_set = get_model_morpheme_dataset_partition(home_path, dev_inf_set)
test_inf_set = get_model_morpheme_dataset_partition(home_path, test_inf_set)
dev_uninf_set = get_model_morpheme_dataset_partition(home_path, dev_uninf_set)
test_uninf_set = get_model_morpheme_dataset_partition(home_path, test_uninf_set)
train_sampler = RandomSampler(train_set)
dev_inf_sampler = SequentialSampler(dev_inf_set)
test_inf_sampler = SequentialSampler(test_inf_set)
dev_uninf_sampler = SequentialSampler(dev_uninf_set)
test_uninf_sampler = SequentialSampler(test_uninf_set)
train_dataloader = DataLoader(train_set, sampler=train_sampler)
dev_inf_dataloader = DataLoader(dev_inf_set, sampler=dev_inf_sampler)
test_inf_dataloader = DataLoader(test_inf_set, sampler=test_inf_sampler)
dev_uninf_dataloader = DataLoader(dev_uninf_set, sampler=dev_uninf_sampler)
test_uninf_dataloader = DataLoader(test_uninf_set, sampler=test_uninf_sampler)

# Embedding
ft_form_vec_file_path = Path('data/processed/spmrl/hebtb-morph-vocab/word-form.vec')
ft_lemma_vec_file_path = Path('data/processed/spmrl/hebtb-morph-vocab/word-lemma.vec')
tag_vec_file_path = Path('data/processed/spmrl/hebtb-morph-vocab/tag.vec')
feat_vec_file_path = Path('data/processed/spmrl/hebtb-morph-vocab/feat.vec')


def get_model(hidden_size: int, num_layers: int, enc_dropout: float, enc_emb_dropout: float, dec_dropout: float,
              dec_emb_dropout: float, seq_enc_dropout, seq_dec_dropout: float) -> PtrNetModel:
    form_emb = model_ft.load_embedding_weight_matrix(home_path, ft_form_vec_file_path, tb_vocab.forms, device)
    lemma_emb = model_ft.load_embedding_weight_matrix(home_path, ft_lemma_vec_file_path, tb_vocab.lemmas, device)
    tags_num = len(tb_vocab.tags)
    feats_num = len(tb_vocab.feats)
    tag_emb = nn.Embedding(num_embeddings=tags_num, embedding_dim=50, padding_idx=0)
    feats_emb = nn.Embedding(num_embeddings=feats_num, embedding_dim=50, padding_idx=0)
    analysis_emb = AnalysisEmbedding(form_emb, lemma_emb, tag_emb, feats_emb)
    encoder = AnalysisEncoder(enc_dropout, analysis_emb.embedding_dim, hidden_size, num_layers, enc_emb_dropout)
    decoder = AnalysisDecoder(dec_dropout, analysis_emb.embedding_dim, hidden_size, num_layers, dec_emb_dropout)
    attention = Attention()
    model = PtrNetModel(tb_vocab, analysis_emb, encoder, seq_enc_dropout, decoder, seq_dec_dropout, attention)
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

    def step(self, loss):
        self.steps += 1
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


def ptrnet_ma(tokens: list, lex: Lexicon, model: PtrNetModel) -> nlp.Sentence:
    lattice = ma.lattice(tokens, lex)
    gold_lattice = morph.Lattice()
    for token_id in lattice:
        gold_lattice[token_id] = [morph.Analysis([], [], [])]
    sentence = nlp.Sentence(tokens, lattice, gold_lattice)
    new_tokens, new_forms, new_lemmas = model.vocab.update(sentence)
    if new_forms:
        new_form_matrix = model_ft.get_word_vectors(home_path, sorted(new_forms))
        new_form_matrix = torch.tensor(new_form_matrix, dtype=torch.float, device=device)
        model.emb.update_form_emb_(new_form_matrix)
    if new_lemmas:
        new_lemma_matrix = model_ft.get_word_vectors(home_path, sorted(new_lemmas))
        new_lemma_matrix = torch.tensor(new_lemma_matrix, dtype=torch.float, device=device)
        model.emb.update_lemma_emb_(new_lemma_matrix)
    return sentence


def to_pred_lattice(lattice: torch.Tensor, pred_indices: torch.Tensor) -> torch.Tensor:
    pred_lattice = lattice.clone()
    pred_lattice[:, :, lattice_is_gold_column_pos] = 0
    pred_indices_sparse = torch.zeros(pred_lattice.shape[0], dtype=torch.bool, device=device).scatter(0, pred_indices, 1).unsqueeze(1).repeat(1, pred_lattice.shape[1])
    valid_morphemes = pred_lattice[:, :, 0] != 0
    valid_pred_indices = pred_indices_sparse & valid_morphemes
    pred_lattice[:, :, lattice_is_gold_column_pos] = valid_pred_indices
    return pred_lattice


def ptrnet_md(tokens: list, lex: Lexicon, model: PtrNetModel) -> nlp.Sentence:
    sent = ptrnet_ma(tokens, lex, model)
    lattice = to_tensor(sent, model.vocab)
    lattice = lattice.to(device)
    pred_indices = model.decode(lattice)
    pred_lattice = to_pred_lattice(lattice, pred_indices)
    return to_sentence(pred_lattice.cpu(), model.vocab)


def process_step(batch: tuple, model: PtrNetModel, optimizer: ModelOptimizer) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    batch = tuple(t.to(device) for t in batch)
    use_teacher_forcing = optimizer is not None
    lattice = batch[0]
    pred_indices, loss = model(lattice, use_teacher_forcing=use_teacher_forcing)
    if optimizer is not None:
        optimizer.step(loss)
    pred_lattice = to_pred_lattice(lattice, pred_indices)
    return lattice, pred_lattice, loss


def print_eval(gold_sentences: list, pred_sentences: list):
    print(eval_msr(gold_sentences, pred_sentences))
    print(eval_seg_tag(gold_sentences, pred_sentences))
    print(eval_tag_feats(gold_sentences, pred_sentences))
    print(eval_seg(gold_sentences, pred_sentences))
    print(eval_tag(gold_sentences, pred_sentences))
    print(eval_feats(gold_sentences, pred_sentences))


def process(epoch: int, print_every: int, phase: str, data: DataLoader, model: PtrNetModel, optimizer=None):
    total_loss, print_loss = 0, 0
    total_gold_sentences, total_pred_sentences, print_gold_sentences, print_pred_sentences = [], [], [], []
    for j, batch in enumerate(data):
        step = j + 1
        gold_lattice, pred_lattice, loss = process_step(batch, model, optimizer)
        total_loss += loss.item()
        print_loss += loss.item()
        gold_sentence = to_sentence(gold_lattice.cpu(), model.vocab)
        pred_sentence = to_sentence(pred_lattice.cpu(), model.vocab)
        print_gold_sentences.append(gold_sentence)
        print_pred_sentences.append(pred_sentence)
        total_gold_sentences.append(gold_sentence)
        total_pred_sentences.append(pred_sentence)
        if step % print_every == 0:
            print(f'epoch {epoch} {phase} step {step} loss {print_loss/print_every}')
            print_eval(print_gold_sentences, print_pred_sentences)
            print_loss = 0
            print_gold_sentences, print_pred_sentences = [], []
    print(f'epoch {epoch} {phase} total {len(data)} loss {total_loss / len(data)}')
    print_eval(total_gold_sentences, total_pred_sentences)


ptrnet = get_model(hidden_size=64, num_layers=1, enc_dropout=0.1, enc_emb_dropout=0.1, dec_dropout=0.1,
                   dec_emb_dropout=0.1, seq_enc_dropout=0.1, seq_dec_dropout=0.1)
print(ptrnet)
optimizer_params = list(ptrnet.parameters())
epochs = 3
lr = 1e-3
print_every = 32
for i in trange(epochs, desc="Epoch"):
    epoch = i + 1
    ptrnet.train()
    ptrnet_optimizer = get_optimizer(parameters=optimizer_params, max_grad_norm=0.0, batch_size=1, lr=lr)
    lr /= 10
    process(epoch, print_every, "train_infused", train_dataloader, ptrnet, ptrnet_optimizer)
    ptrnet.eval()
    with torch.no_grad():
        dec_sentence = ptrnet_md('מצאנו חמש בלוטות לימפה שפירות'.split(), bgulex, ptrnet)
        for token_id in dec_sentence.gold_lattice:
            print(dec_sentence.gold_lattice[token_id])
        dec_sentence = ptrnet_md('אני שוטף ידיים בסבון ומים'.split(), bgulex, ptrnet)
        for token_id in dec_sentence.gold_lattice:
            print(dec_sentence.gold_lattice[token_id])
        process(epoch, print_every, "dev_infused", dev_inf_dataloader, ptrnet)
        process(epoch, print_every, "test_infused", test_inf_dataloader, ptrnet)
        process(epoch, print_every, "dev_uninfused", dev_uninf_dataloader, ptrnet)
        process(epoch, print_every, "test_uninfused", test_uninf_dataloader, ptrnet)
