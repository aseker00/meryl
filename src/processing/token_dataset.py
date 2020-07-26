import random
from collections import defaultdict
from pathlib import Path
from src.processing.spmrl.lexicon import Lexicon
from src.processing.spmrl.treebank import Treebank
from src.processing.token_vocab import TokenVocab
from src.processing import morph, nlp
import pandas as pd
import numpy as np


_columns = ['sent_idx',
            'token_idx',
            'analysis_idx',
            'is_gold',
            'pref_forms',
            'pref_forms_id',
            'pref_lemmas',
            'pref_lemmas_id',
            'pref_tags',
            'pref_tags_id',
            'pref_feats',
            'pref_feats_id',
            'host_forms',
            'host_forms_id',
            'host_lemmas',
            'host_lemmas_id',
            'host_tags',
            'host_tags_id',
            'host_feats',
            'host_feats_id',
            'suff_forms',
            'suff_forms_id',
            'suff_lemmas',
            'suff_lemmas_id',
            'suff_tags',
            'suff_tags_id',
            'suff_feats',
            'suff_feats_id',
            'token',
            'token_id']
lattice_columns = _columns[5::2] + _columns[:4]
lattice_pref_form_column_pos = 0
lattice_pref_lemma_column_pos = 1
lattice_pref_tag_column_pos = 2
lattice_pref_feats_column_pos = 3
lattice_host_form_column_pos = 4
lattice_host_lemma_column_pos = 5
lattice_host_tag_column_pos = 6
lattice_host_feats_column_pos = 7
lattice_suff_form_column_pos = 8
lattice_suff_lemma_column_pos = 9
lattice_suff_tag_column_pos = 10
lattice_suff_feats_column_pos = 11
lattice_token_column_pos = 12
lattice_sent_idx_column_pos = 13
lattice_token_idx_column_pos = 14
lattice_analysis_idx_column_pos = 15
lattice_is_gold_column_pos = 16


def _gold_idx(gold_analysis: morph.Analysis, analyses: list, indices: list) -> int:
    gold_index = -1
    features = [morph.Gender, morph.Number, morph.Person, morph.Tense, morph.Binyan, morph.Polarity]
    while gold_index < 0:
        for i, index in enumerate(indices):
            if morph.analysis_equals_no_lemma(gold_analysis, analyses[index], features):
                gold_index = i
        if not features:
            break
        features = features[1:]
    return gold_index


def _dataframe(sentences: list, vocab: TokenVocab, shuffle: bool, validate: bool) -> pd.DataFrame:
    
    def append_analysis_prefixes(analysis: morph.Analysis):
        pref_form = tuple(analysis.pref_forms)
        pref_form_id = vocab.pref_form2id[pref_form]
        pref_lemma = tuple(analysis.pref_lemmas)
        pref_lemma_id = vocab.pref_lemma2id[pref_lemma]
        pref_tag = tuple(analysis.pref_tags)
        pref_tag_id = vocab.pref_tag2id[pref_tag]
        pref_feat = tuple(analysis.pref_feats)
        pref_feat_id = vocab.pref_feat2id[pref_feat]
        pref_forms.append(pref_form)
        pref_lemmas.append(pref_lemma)
        pref_tags.append(pref_tag)
        pref_feats.append(pref_feat)
        pref_form_ids.append(pref_form_id)
        pref_lemma_ids.append(pref_lemma_id)
        pref_tag_ids.append(pref_tag_id)
        pref_feats_ids.append(pref_feat_id)

    def append_analysis_hosts(analysis: morph.Analysis):
        host_form = tuple(analysis.host_forms)
        host_form_id = vocab.host_form2id[host_form]
        host_lemma = tuple(analysis.host_lemmas)
        host_lemma_id = vocab.host_lemma2id[host_lemma]
        host_tag = tuple(analysis.host_tags)
        host_tag_id = vocab.host_tag2id[host_tag]
        host_feat = tuple(analysis.host_feats)
        host_feat_id = vocab.host_feat2id[host_feat]
        host_forms.append(host_form)
        host_lemmas.append(host_lemma)
        host_tags.append(host_tag)
        host_feats.append(host_feat)
        host_form_ids.append(host_form_id)
        host_lemma_ids.append(host_lemma_id)
        host_tag_ids.append(host_tag_id)
        host_feats_ids.append(host_feat_id)
        
    def append_analysis_suffixes(analysis: morph.Analysis):
        suff_form = tuple(analysis.suff_forms)
        suff_form_id = vocab.suff_form2id[suff_form]
        suff_lemma = tuple(analysis.suff_lemmas)
        suff_lemma_id = vocab.suff_lemma2id[suff_lemma]
        suff_tag = tuple(analysis.suff_tags)
        suff_tag_id = vocab.suff_tag2id[suff_tag]
        suff_feat = tuple(analysis.suff_feats)
        suff_feat_id = vocab.suff_feat2id[suff_feat]
        suff_forms.append(suff_form)
        suff_lemmas.append(suff_lemma)
        suff_tags.append(suff_tag)
        suff_feats.append(suff_feat)
        suff_form_ids.append(suff_form_id)
        suff_lemma_ids.append(suff_lemma_id)
        suff_tag_ids.append(suff_tag_id)
        suff_feats_ids.append(suff_feat_id)
        
    def append_analysis(sent_idx: int, token_idx: int, analysis_idx: int, analysis: morph.Analysis, is_gold: int, token: str):
        sentence_indices.append(sent_idx)
        token_indices.append(token_idx)
        analysis_indices.append(analysis_idx)
        gold_indices.append(is_gold)
        append_analysis_prefixes(analysis)
        append_analysis_hosts(analysis)
        append_analysis_suffixes(analysis)
        tokens.append(token)
        token_ids.append(vocab.token2id[token])

    def append_sos_analysis(sent_idx: int):
        t = '<SOS>'
        m = morph.Morpheme(t, t, t, morph.EMPTY_FEATURES)
        a = morph.Analysis([], [m], [])
        append_analysis(sent_idx, 0, 0, a, 1, t)

    def append_sentence(sent: nlp.Sentence, sent_idx: int, shuffle: bool):
        append_sos_analysis(sent_idx)
        for i, token in enumerate(sent.tokens):
            token_idx = i + 1
            token_analyses = sent.lattice[token_idx]
            token_gold_analysis = sent.analysis(token_idx)
            token_analyses_indices = list(range(len(token_analyses)))
            if shuffle:
                random.shuffle(token_analyses_indices)
            gold_index = _gold_idx(token_gold_analysis, token_analyses, token_analyses_indices)
            if validate:
                assert gold_index > -1
            if gold_index < 0:
                append_analysis(sent_idx, token_idx, gold_index, token_gold_analysis, gold_index, token)
            for j, analysis_idx in enumerate(token_analyses_indices):
                analysis = token_analyses[analysis_idx]
                is_gold = int(j == gold_index)
                append_analysis(sent_idx, token_idx, analysis_idx, analysis, is_gold, token)

    sentence_indices, token_indices, analysis_indices, gold_indices = [], [], [], []
    pref_forms, pref_lemmas, pref_tags, pref_feats = [], [], [], []
    pref_form_ids, pref_lemma_ids, pref_tag_ids, pref_feats_ids = [], [], [], []
    host_forms, host_lemmas, host_tags, host_feats = [], [], [], []
    host_form_ids, host_lemma_ids, host_tag_ids, host_feats_ids = [], [], [], []
    suff_forms, suff_lemmas, suff_tags, suff_feats = [], [], [], []
    suff_form_ids, suff_lemma_ids, suff_tag_ids, suff_feats_ids = [], [], [], []
    tokens = []
    token_ids = []
    for i, sentence in enumerate(sentences):
        sentence_index = i + 1
        append_sentence(sentence, sentence_index, shuffle)
    d = {_columns[0]: sentence_indices, _columns[1]: token_indices,
         _columns[2]: analysis_indices, _columns[3]: gold_indices,
         _columns[4]: pref_forms, _columns[5]: pref_form_ids,
         _columns[6]: pref_lemmas, _columns[7]: pref_lemma_ids,
         _columns[8]: pref_tags, _columns[9]: pref_tag_ids,
         _columns[10]: pref_feats, _columns[11]: pref_feats_ids,
         _columns[12]: host_forms, _columns[13]: host_form_ids,
         _columns[14]: host_lemmas, _columns[15]: host_lemma_ids,
         _columns[16]: host_tags, _columns[17]: host_tag_ids,
         _columns[18]: host_feats, _columns[19]: host_feats_ids,
         _columns[20]: suff_forms, _columns[21]: suff_form_ids,
         _columns[22]: suff_lemmas, _columns[23]: suff_lemma_ids,
         _columns[24]: suff_tags, _columns[25]: suff_tag_ids,
         _columns[26]: suff_feats, _columns[27]: suff_feats_ids,
         _columns[28]: tokens, _columns[29]: token_ids}
    return pd.DataFrame(d)


def _arr_to_dataframe(arr: np.ndarray, vocab: TokenVocab) -> pd.DataFrame:
    tokens = []
    pref_forms, pref_lemmas, pref_tags, pref_feats = [], [], [], []
    host_forms, host_lemmas, host_tags, host_feats = [], [], [], []
    suff_forms, suff_lemmas, suff_tags, suff_feats = [], [], [], []
    sent_indices, token_indices, analysis_indices, gold_indices = [], [], [], []
    for analysis in arr:
        analysis_pref_forms = vocab.pref_forms[analysis[0]]
        analysis_pref_lemmas = vocab.pref_lemmas[analysis[1]]
        analysis_pref_tags = vocab.pref_tags[analysis[2]]
        analysis_pref_feats = vocab.pref_tags[analysis[3]]
        analysis_host_forms = vocab.pref_forms[analysis[4]]
        analysis_host_lemmas = vocab.pref_lemmas[analysis[5]]
        analysis_host_tags = vocab.pref_tags[analysis[6]]
        analysis_host_feats = vocab.pref_tags[analysis[7]]
        analysis_suff_forms = vocab.pref_forms[analysis[8]]
        analysis_suff_lemmas = vocab.pref_lemmas[analysis[9]]
        analysis_suff_tags = vocab.pref_tags[analysis[10]]
        analysis_suff_feats = vocab.pref_tags[analysis[11]]
        pref_forms.append(analysis_pref_forms)
        pref_lemmas.append(analysis_pref_lemmas)
        pref_tags.append(analysis_pref_tags)
        pref_feats.append(analysis_pref_feats)
        host_forms.append(analysis_host_forms)
        host_lemmas.append(analysis_host_lemmas)
        host_tags.append(analysis_host_tags)
        host_feats.append(analysis_host_feats)
        suff_forms.append(analysis_suff_forms)
        suff_lemmas.append(analysis_suff_lemmas)
        suff_tags.append(analysis_suff_tags)
        suff_feats.append(analysis_suff_feats)
        token = vocab.tokens[analysis[12]]
        tokens.append(token)
        sent_idx = analysis[13]
        sent_indices.append(sent_idx)
        token_idx = analysis[14]
        token_indices.append(token_idx)
        analysis_idx = analysis[15]
        analysis_indices.append(analysis_idx)
        is_gold = analysis[16]
        gold_indices.append(is_gold)
    d = {_columns[0]: sent_indices,
         _columns[1]: token_indices,
         _columns[2]: analysis_indices,
         _columns[3]: gold_indices,
         _columns[4]: pref_forms,
         _columns[6]: pref_lemmas,
         _columns[8]: pref_tags,
         _columns[10]: pref_feats,
         _columns[12]: host_forms,
         _columns[14]: host_lemmas,
         _columns[16]: host_tags,
         _columns[18]: host_feats,
         _columns[20]: suff_forms,
         _columns[22]: suff_lemmas,
         _columns[24]: suff_tags,
         _columns[26]: suff_feats,
         _columns[28]: tokens}
    return pd.DataFrame(d)


def arr_to_sentence(arr: np.ndarray, vocab: TokenVocab) -> nlp.Sentence:
    tokens = {}
    token_analyses = defaultdict(list)
    token_gold_analyses = defaultdict(list)
    for analysis_arr in arr:
        pref_forms = vocab.pref_forms[analysis_arr[0]]
        pref_lemmas = vocab.pref_lemmas[analysis_arr[1]]
        pref_tags = vocab.pref_tags[analysis_arr[2]]
        pref_feats = vocab.pref_feats[analysis_arr[3]]
        host_forms = vocab.host_forms[analysis_arr[4]]
        host_lemmas = vocab.host_lemmas[analysis_arr[5]]
        host_tags = vocab.host_tags[analysis_arr[6]]
        host_feats = vocab.host_feats[analysis_arr[7]]
        suff_forms = vocab.suff_forms[analysis_arr[8]]
        suff_lemmas = vocab.suff_lemmas[analysis_arr[9]]
        suff_tags = vocab.suff_tags[analysis_arr[10]]
        suff_feats = vocab.suff_feats[analysis_arr[11]]
        token = vocab.tokens[analysis_arr[12]]
        # sent_idx = analysis_arr[13]
        token_idx = analysis_arr[14]
        # analysis_idx = analysis_arr[15]
        is_gold = analysis_arr[16]
        # if analysis_arr[6] == vocab.host_tag2id[tuple('<PAD>')]:
        if analysis_arr[6] == 0:
            break
        tokens[token_idx] = token
        prefixes, hosts, suffixes = [], [], []
        for form, lemma, tag, feats in zip(pref_forms, pref_lemmas, pref_tags, pref_feats):
            # feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
            m = morph.Morpheme(form, lemma, tag, feats)
            prefixes.append(m)
        for form, lemma, tag, feats in zip(host_forms, host_lemmas, host_tags, host_feats):
            # feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
            m = morph.Morpheme(form, lemma, tag, feats)
            hosts.append(m)
        for form, lemma, tag, feats in zip(suff_forms, suff_lemmas, suff_tags, suff_feats):
            # feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
            m = morph.Morpheme(form, lemma, tag, feats)
            suffixes.append(m)
        analysis = morph.Analysis(prefixes, hosts, suffixes)
        token_analyses[token_idx].append(analysis)
        if is_gold:
            token_gold_analyses[token_idx].append(analysis)
    tokens = [tokens[token_id] for token_id in sorted(tokens)]
    lattice = morph.Lattice()
    for token_id in token_analyses:
        lattice[token_id] = token_analyses[token_id]
    gold_lattice = morph.Lattice()
    for token_id in token_gold_analyses:
        gold_lattice[token_id] = token_gold_analyses[token_id]
    return nlp.Sentence(tokens, lattice, gold_lattice)


def sentence_to_arr(sent: nlp.Sentence, vocab: TokenVocab) -> np.ndarray:
    sent_df = _dataframe([sent], vocab, False, False)
    ds = TokenDataset(sent_df)
    return ds[0]
    # return sent_df.loc[:, _lattice_columns].to_numpy()


def _dataframe_to_sentence(sent_df: pd.DataFrame) -> nlp.Sentence:
    token_gb = sent_df.iloc[1:].groupby(sent_df.token_idx)
    tokens = [tg[1].iloc[0].token for tg in sorted(token_gb)]
    token_analyses = defaultdict(list)
    token_gold_analyses = defaultdict(list)
    for tg in sorted(token_gb):
        token_idx = tg[0]
        token_analyses = []
        gold_token_analyses = []
        analysis_gb = tg[1].groupby(sent_df.analysis_idx)
        for ag in analysis_gb:
            pref_forms = ag[1]['pref_forms'].iloc[0]
            pref_lemmas = ag[1]['pref_lemmas'].iloc[0]
            pref_tags = ag[1]['pref_tags'].iloc[0]
            pref_feats = ag[1]['pref_feats'].iloc[0]
            host_forms = ag[1]['host_forms'].iloc[0]
            host_lemmas = ag[1]['host_lemmas'].iloc[0]
            host_tags = ag[1]['host_tags'].iloc[0]
            host_feats = ag[1]['host_feats'].iloc[0]
            suff_forms = ag[1]['suff_forms'].iloc[0]
            suff_lemmas = ag[1]['suff_lemmas'].iloc[0]
            suff_tags = ag[1]['suff_tags'].iloc[0]
            suff_feats = ag[1]['suff_feats'].iloc[0]
            prefixes, hosts, suffixes = [], [], []
            for form, lemma, tag, fstr in zip(pref_forms, pref_lemmas, pref_tags, pref_feats):
                feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
                m = morph.Morpheme(form, lemma, tag, feats)
                prefixes.append(m)
            for form, lemma, tag, fstr in zip(host_forms, host_lemmas, host_tags, host_feats):
                feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
                m = morph.Morpheme(form, lemma, tag, feats)
                hosts.append(m)
            for form, lemma, tag, fstr in zip(suff_forms, suff_lemmas, suff_tags, suff_feats):
                feats = morph.Features.create([f for f in fstr.split("|") if f != '_'])
                m = morph.Morpheme(form, lemma, tag, feats)
                suffixes.append(m)
            analysis = morph.Analysis(prefixes, hosts, suffixes)
            token_analyses[token_idx].append(analysis)
            is_gold = ag[1]['is_gold'].iloc[0]
            if is_gold:
                gold_token_analyses[token_idx].append(analysis)
    lattice = morph.Lattice(token_analyses)
    gold_lattice = morph.Lattice(token_gold_analyses)
    return nlp.Sentence(tokens, lattice, gold_lattice)


class TokenDataset:

    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.df = df
        self.max_token_num = df.token_idx.max() + 1

    def __getitem__(self, idx: int) -> np.ndarray:

        def resize_lattice(a: np.ndarray, size: int) -> np.ndarray:
            pad_size = size - a.shape[0]
            if pad_size == 0:
                return a
            npad = ((0, pad_size), (0, 0))
            return np.pad(a, pad_width=npad, mode='constant', constant_values=0)

        lattice = []
        sent_df = self.df.loc[self.df.sent_idx == idx + 1]
        gb = sent_df.groupby(_columns[1:3])
        for g in sorted(gb):
            garr = g[1].loc[:, lattice_columns].to_numpy()
            lattice.append(garr.reshape(-1))
        lattice = np.stack(lattice)
        gold_indices = lattice[:, lattice_is_gold_column_pos]
        gold_indices = np.nonzero(gold_indices)
        return resize_lattice(lattice[gold_indices], self.max_token_num)

    def __len__(self) -> int:
        return self.df.sent_idx.nunique()

    def save(self, path: Path):
        print(f"Saving token dataset to {path}")
        self.df.to_csv(str(path))

    @classmethod
    def load(cls, name: str, path: Path):
        print(f"Loading token dataset from {path}")
        df = pd.read_csv(str(path))
        return cls(name, df)


def get_token_dataset_partition(partition_name: str, home_path: Path, vocab: TokenVocab, hebtb: Treebank) -> TokenDataset:
    path = home_path / f'data/processed/spmrl/hebtb-token-dataset/{partition_name}.csv'
    if path.exists():
        return TokenDataset.load(partition_name, path)
    if partition_name == 'dev-inf':
        df = _dataframe(hebtb.infused_dev_sentences, vocab, False, True)
    elif partition_name == 'test-inf':
        df = _dataframe(hebtb.infused_test_sentences, vocab, False, True)
    elif partition_name == 'dev-uninf':
        df = _dataframe(hebtb.dev_sentences, vocab, False, False)
    elif partition_name == 'test-uninf':
        df = _dataframe(hebtb.test_sentences, vocab, False, False)
    else:
        df = _dataframe(hebtb.infused_train_sentences, vocab, False, True)
    ds = TokenDataset(partition_name, df)
    ds.save(path)
    return ds


def main():
    home_path = Path('.')
    src_tokens_idx = '01'
    src_suffix = '10'
    tb_files = {'train-hebtb.tokens': f'data/clean/spmrl/hebtb/train-hebtb-{src_tokens_idx}-tokens.txt',
                'train-hebtb-gold.lattices': f'data/clean/spmrl/hebtb/train-hebtb-{src_suffix}-gold.lattices',
                'dev-hebtb.tokens': f'data/clean/spmrl/hebtb/dev-hebtb-{src_tokens_idx}-tokens.txt',
                'dev-hebtb-gold.lattices': f'data/clean/spmrl/hebtb/dev-hebtb-{src_suffix}-gold.lattices',
                'test-hebtb.tokens': f'data/clean/spmrl/hebtb/test-hebtb-{src_tokens_idx}-tokens.txt',
                'test-hebtb-gold.lattices': f'data/clean/spmrl/hebtb/test-hebtb-{src_suffix}-gold.lattices'}
    lex_files = {'pref-lex': 'data/raw/spmrl/bgulex/bgupreflex_withdef.utf8.hr',
                 'lex': 'data/clean/spmrl/bgulex/bgulex-03.hr'}
    bgulex_file_path = Path('data/processed/spmrl/bgulex.pickle')
    hebtb_file_path = Path('data/processed/spmrl/hebtb.pickle')
    vocab_file_path = Path('data/processed/spmrl/hebtb-token-vocab/vocab.pickle')
    if bgulex_file_path.exists():
        bgulex = Lexicon.load(bgulex_file_path)
    else:
        bgulex = Lexicon(lex_files)
        bgulex.save(bgulex_file_path)
    if hebtb_file_path.exists():
        hebtb = Treebank.load(hebtb_file_path)
    else:
        hebtb = Treebank(bgulex, tb_files)
        hebtb.save(hebtb_file_path)
    tb_train_size = len(hebtb.infused_train_sentences)
    tb_dev_size = len(hebtb.infused_dev_sentences)
    tb_test_size = len(hebtb.infused_test_sentences)
    print(f"Train sentences: {tb_train_size}")
    print(f"Dev sentences: {tb_dev_size}")
    print(f"Test sentences: {tb_test_size}")
    tb_sentences = (hebtb.infused_train_sentences + hebtb.infused_dev_sentences + hebtb.infused_test_sentences)
    if vocab_file_path.exists():
        tb_vocab = TokenVocab.load(vocab_file_path)
    else:
        tb_vocab = TokenVocab(tb_sentences)
        tb_vocab.save(vocab_file_path)
    print("Vocab tokens: {}".format(len(tb_vocab.tokens)))
    print("Vocab pref forms: {}".format(len(tb_vocab.pref_forms)))
    print("Vocab host forms: {}".format(len(tb_vocab.host_forms)))
    print("Vocab suff forms: {}".format(len(tb_vocab.suff_forms)))
    print("Vocab pref lemmas: {}".format(len(tb_vocab.pref_lemmas)))
    print("Vocab host lemmas: {}".format(len(tb_vocab.host_lemmas)))
    print("Vocab suff lemmas: {}".format(len(tb_vocab.suff_lemmas)))
    print("Vocab pref tags: {}".format(len(tb_vocab.pref_tags)))
    print("Vocab host tags: {}".format(len(tb_vocab.host_tags)))
    print("Vocab suff tags: {}".format(len(tb_vocab.suff_tags)))
    print("Vocab pref feats: {}".format(len(tb_vocab.pref_feats)))
    print("Vocab host feats: {}".format(len(tb_vocab.host_feats)))
    print("Vocab suff feats: {}".format(len(tb_vocab.suff_feats)))

    train_ds = get_token_dataset_partition('train-inf', home_path, tb_vocab, hebtb)
    dev_inf_ds = get_token_dataset_partition('dev-inf', home_path, tb_vocab, hebtb)
    test_inf_ds = get_token_dataset_partition('test-inf', home_path, tb_vocab, hebtb)
    dev_uninf_ds = get_token_dataset_partition('dev-uninf', home_path, tb_vocab, hebtb)
    test_uninf_ds = get_token_dataset_partition('test-uninf', home_path, tb_vocab, hebtb)
    print("Train infused dataset: {}".format(len(train_ds)))
    print("Dev infused dataset: {}".format(len(dev_inf_ds)))
    print("Test infused dataset: {}".format(len(test_inf_ds)))
    print("Dev uninfused dataset: {}".format(len(dev_uninf_ds)))
    print("Test uninfused dataset: {}".format(len(test_uninf_ds)))


if __name__ == "__main__":
    main()
