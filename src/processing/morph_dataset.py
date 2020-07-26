import random
from collections import defaultdict
from pathlib import Path
from src.processing.spmrl.lexicon import Lexicon
from src.processing.spmrl.treebank import Treebank
from src.processing.morph_vocab import MorphVocab
from src.processing import morph, nlp
import pandas as pd
import numpy as np


_columns = ['sent_idx',
            'token_idx',
            'analysis_idx',
            'is_gold',
            'morpheme_idx',
            'mtype',
            'mtype_id',
            'form',
            'form_id',
            'lemma',
            'lemma_id',
            'tag',
            'tag_id',
            'gender',
            'gender_id',
            'number',
            'number_id',
            'person',
            'person_id',
            'tense',
            'tense_id',
            'binyan',
            'binyan_id',
            'polarity',
            'polarity_id',
            'token',
            'token_id']
_lattice_columns = _columns[8::2] + _columns[:5] + [_columns[6]]
lattice_form_column_pos = 0
lattice_lemma_column_pos = 1
lattice_tag_column_pos = 2
lattice_gender_column_pos = 3
lattice_number_column_pos = 4
lattice_person_column_pos = 5
lattice_tense_column_pos = 6
lattice_binyan_column_pos = 7
lattice_polarity_column_pos = 8
lattice_token_column_pos = 9
lattice_sent_idx_column_pos = 10
lattice_token_idx_column_pos = 11
lattice_analysis_idx_column_pos = 12
lattice_is_gold_column_pos = 13
lattice_morpheme_idx_column_pos = 14
lattice_type_column_pos = 15

_mtypes = ['<PAD>', '<SOS>', '<ET>', 'pref', 'host', 'suff']
_mtype2id = {v: i for i, v in enumerate(_mtypes)}


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


def _dataframe(sentences: list, vocab: MorphVocab, shuffle: bool, validate: bool) -> pd.DataFrame:

    def append_morpheme(m: morph.Morpheme, mtype: str):
        mtype_id = _mtype2id[mtype]
        form_id = vocab.form2id[m.form]
        lemma_id = vocab.lemma2id[m.lemma]
        tag_id = vocab.tag2id[m.tag]
        feats = {f: str(m.feats[f]) if m.feats[f] else '_' for f in m.feats}
        feat_ids = {name: vocab.feat2id[value] for name, value in feats.items()}
        mtypes.append(mtype)
        forms.append(m.form)
        lemmas.append(m.lemma)
        tags.append(m.tag)
        for name in feats:
            value = feats[name]
            features[name].append(value)
        mtype_ids.append(mtype_id)
        form_ids.append(form_id)
        lemma_ids.append(lemma_id)
        tag_ids.append(tag_id)
        for name in feat_ids:
            value = feat_ids[name]
            feature_ids[name].append(value)

    def append_analysis(sent_idx: int, token_idx: int, analysis_idx: int, morpheme_idx: int,
                        mtype: str, m: morph.Morpheme, is_gold: int, token: str):
        sentence_indices.append(sent_idx)
        token_indices.append(token_idx)
        analysis_indices.append(analysis_idx)
        morpheme_indices.append(morpheme_idx)
        gold_indices.append(is_gold)
        append_morpheme(m, mtype)
        tokens.append(token)
        token_ids.append(vocab.token2id[token])

    def append_sos_analysis(sent_idx: int):
        t = '<SOS>'
        m = morph.Morpheme(t, t, t, morph.EMPTY_FEATURES)
        append_analysis(sent_idx, 0, 0, 0, t, m, 1, t)

    # def append_et_analysis(sent_idx: int, token_idx: int, analysis_idx: int, is_gold: int, morpheme_idx: int, token: str):
    #     t = '<ET>'
    #     m = morph.Morpheme(t, t, t, morph.EMPTY_FEATURES)
    #     append_analysis(sent_idx, token_idx, analysis_idx, morpheme_idx, t, m, is_gold, token)

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
                morpheme_idx = 0
                for morpheme in token_gold_analysis.prefixes:
                    append_analysis(sent_idx, token_idx, gold_index, morpheme_idx, 'pref', morpheme, gold_index, token)
                    morpheme_idx += 1
                for morpheme in token_gold_analysis.hosts:
                    append_analysis(sent_idx, token_idx, gold_index, morpheme_idx, 'host', morpheme, gold_index, token)
                    morpheme_idx += 1
                for morpheme in token_gold_analysis.suffixes:
                    append_analysis(sent_idx, token_idx, gold_index, morpheme_idx, 'suff', morpheme, gold_index, token)
                    morpheme_idx += 1
            for j, analysis_idx in enumerate(token_analyses_indices):
                analysis = token_analyses[analysis_idx]
                is_gold = int(j == gold_index)
                morpheme_idx = 0
                for morpheme in analysis.prefixes:
                    append_analysis(sent_idx, token_idx, analysis_idx, morpheme_idx, 'pref', morpheme, is_gold, token)
                    morpheme_idx += 1
                for morpheme in analysis.hosts:
                    append_analysis(sent_idx, token_idx, analysis_idx, morpheme_idx, 'host', morpheme, is_gold, token)
                    morpheme_idx += 1
                for morpheme in analysis.suffixes:
                    append_analysis(sent_idx, token_idx, analysis_idx, morpheme_idx, 'suff', morpheme, is_gold, token)
                    morpheme_idx += 1
            #     if end_token:
            #         append_et_analysis(sent_idx, token_idx, analysis_idx, is_gold, morpheme_idx, token)
            # if end_token:
            #     append_et_analysis(sent_idx, token_idx, analysis_idx, is_gold, morpheme_idx, token)


    sentence_indices, token_indices, analysis_indices, morpheme_indices, gold_indices = [], [], [], [], []
    mtypes, forms, lemmas, tags = [], [], [], []
    mtype_ids, form_ids, lemma_ids, tag_ids = [], [], [], []
    features = {'gen': [], 'num': [], 'per': [], 'tense': [], 'binyan': [], 'pol': []}
    feature_ids = {'gen': [], 'num': [], 'per': [], 'tense': [], 'binyan': [], 'pol': []}
    tokens = []
    token_ids = []
    for i, sentence in enumerate(sentences):
        sentence_index = i + 1
        append_sentence(sentence, sentence_index, shuffle)
    d = {_columns[0]: sentence_indices, _columns[1]: token_indices,
         _columns[2]: analysis_indices, _columns[3]: gold_indices, _columns[4]: morpheme_indices,
         _columns[5]: mtypes, _columns[6]: mtype_ids,
         _columns[7]: forms, _columns[8]: form_ids,
         _columns[9]: lemmas, _columns[10]: lemma_ids,
         _columns[11]: tags, _columns[12]: tag_ids,
         _columns[13]: features['gen'], _columns[14]: feature_ids['gen'],
         _columns[15]: features['num'], _columns[16]: feature_ids['num'],
         _columns[17]: features['per'], _columns[18]: feature_ids['per'],
         _columns[19]: features['tense'], _columns[20]: feature_ids['tense'],
         _columns[21]: features['binyan'], _columns[22]: feature_ids['binyan'],
         _columns[23]: features['pol'], _columns[24]: feature_ids['pol'],
         _columns[25]: tokens, _columns[26]: token_ids}
    return pd.DataFrame(d)


def _arr_to_dataframe(arr: np.ndarray, vocab: MorphVocab) -> pd.DataFrame:
    tokens = []
    mtypes, forms, lemmas, tags = [], [], [], []
    features = {'gen': [], 'num': [], 'per': [], 'tense': [], 'binyan': [], 'pol': []}
    sent_indices, token_indices, analysis_indices, morpheme_indices, gold_indices = [], [], [], [], []
    morpheme_arr = np.resize(arr, (-1, arr.shape[2]))
    for morpheme in morpheme_arr:
        if morpheme[0] == 0:
            continue
        form = vocab.forms[morpheme[0]]
        forms.append(form)
        lemma = vocab.lemmas[morpheme[1]]
        lemmas.append(lemma)
        tag = vocab.tags[morpheme[2]]
        tags.append(tag)
        gender = vocab.feats[morpheme[3]]
        features['gen'].append(gender)
        number = vocab.feats[morpheme[4]]
        features['num'].append(number)
        person = vocab.feats[morpheme[5]]
        features['per'].append(person)
        tense = vocab.feats[morpheme[6]]
        features['tense'].append(tense)
        binyan = vocab.feats[morpheme[7]]
        features['binyan'].append(binyan)
        polarity = vocab.feats[morpheme[8]]
        features['pol'].append(polarity)
        token = vocab.tokens[morpheme[9]]
        tokens.append(token)
        sent_idx = morpheme[10]
        sent_indices.append(sent_idx)
        token_idx = morpheme[11]
        token_indices.append(token_idx)
        analysis_idx = morpheme[12]
        analysis_indices.append(analysis_idx)
        is_gold = morpheme[13]
        gold_indices.append(is_gold)
        morpheme_idx = morpheme[14]
        morpheme_indices.append(morpheme_idx)
        mtype = _mtypes[morpheme[15]]
        mtypes.append(mtype)
    d = {_columns[0]: sent_indices,
         _columns[1]: token_indices,
         _columns[2]: analysis_indices,
         _columns[3]: gold_indices,
         _columns[4]: morpheme_indices,
         _columns[5]: mtypes,
         _columns[7]: forms,
         _columns[9]: lemmas,
         _columns[11]: tags,
         _columns[13]: features['gen'],
         _columns[15]: features['num'],
         _columns[17]: features['per'],
         _columns[19]: features['tense'],
         _columns[21]: features['binyan'],
         _columns[23]: features['pol'],
         _columns[25]: tokens}
    return pd.DataFrame(d)


def arr_to_sentence(arr: np.ndarray, vocab: MorphVocab) -> nlp.Sentence:
    morphemes = {}
    a2m = defaultdict(list)
    t2a = defaultdict(set)
    t2t = {}
    analysis_arr = np.resize(arr, (-1, arr.shape[2]))
    for morpheme_arr in analysis_arr:
        if morpheme_arr[0] == 0:
            continue
        form = vocab.forms[morpheme_arr[0]]
        lemma = vocab.lemmas[morpheme_arr[1]]
        tag = vocab.tags[morpheme_arr[2]]
        gender = vocab.feats[morpheme_arr[3]]
        number = vocab.feats[morpheme_arr[4]]
        person = vocab.feats[morpheme_arr[5]]
        tense = vocab.feats[morpheme_arr[6]]
        binyan = vocab.feats[morpheme_arr[7]]
        polarity = vocab.feats[morpheme_arr[8]]
        token = vocab.tokens[morpheme_arr[9]]
        sent_idx = morpheme_arr[10]
        token_idx = morpheme_arr[11]
        analysis_idx = morpheme_arr[12]
        is_gold = morpheme_arr[13]
        morpheme_idx = morpheme_arr[14]
        mtype = _mtypes[morpheme_arr[15]]
        feats = morph.Features.create([ff for f in [gender, number, person, tense, binyan, polarity] for ff in f.split("|") if ff != '_'])
        m = morph.Morpheme(form, lemma, tag, feats)
        morphemes[(token_idx, analysis_idx, morpheme_idx)] = (mtype, m, is_gold)
        a2m[(token_idx, analysis_idx)].append(morpheme_idx)
        t2a[token_idx].add(analysis_idx)
        t2t[token_idx] = token
    lattice = morph.Lattice()
    gold_lattice = morph.Lattice()
    tokens = []
    for token_idx in t2a:
        if token_idx == 0:
            continue
        token_analyses = []
        gold_token_analyses = []
        for analysis_idx in sorted(t2a[token_idx]):
            prefixes, hosts, suffixes = [], [], []
            is_gold_analysis = False
            for morpheme_idx in a2m[(token_idx, analysis_idx)]:
                mtype, m, is_gold = morphemes[(token_idx, analysis_idx, morpheme_idx)]
                if mtype == 'pref':
                    prefixes.append(m)
                elif mtype == 'suff':
                    suffixes.append(m)
                else:
                    hosts.append(m)
                is_gold_analysis = is_gold != 0
            a = morph.Analysis(prefixes, hosts, suffixes)
            token_analyses.append(a)
            if is_gold_analysis:
                gold_token_analyses.append(a)
        lattice[token_idx] = token_analyses
        gold_lattice[token_idx] = gold_token_analyses
        tokens.append(t2t[token_idx])
    return nlp.Sentence(tokens, lattice, gold_lattice)


def sentence_to_arr(sent: nlp.Sentence, vocab: MorphVocab) -> np.ndarray:
    sent_df = _dataframe([sent], vocab, False, False)
    ds = MorphDataset('', sent_df)
    return ds[0]
    # return sent_df.loc[:, _lattice_columns].to_numpy()


def _dataframe_to_sentence(sent_df: pd.DataFrame) -> nlp.Sentence:
    token_gb = sent_df.iloc[1:].groupby(sent_df.token_idx)
    tokens = [tg[1].iloc[0].token for tg in sorted(token_gb)]
    lattice = morph.Lattice()
    gold_lattice = morph.Lattice()
    for tg in sorted(token_gb):
        token_idx = tg[0]
        token_analyses = []
        gold_token_analyses = []
        analysis_gb = tg[1].groupby(sent_df.analysis_idx)
        for ag in analysis_gb:
            prefixes, hosts, suffixes = [], [], []
            morpheme_gb = ag[1].groupby(sent_df.morpheme_idx)
            for mg in morpheme_gb:
                mtype = mg[1]['mtype'].iloc[0]
                form = mg[1]['form'].iloc[0]
                lemma = mg[1]['lemma'].iloc[0]
                tag = mg[1]['tag'].iloc[0]
                gender = mg[1]['gender'].iloc[0]
                number = mg[1]['number'].iloc[0]
                person = mg[1]['person'].iloc[0]
                tense = mg[1]['tense'].iloc[0]
                binyan = mg[1]['binyan'].iloc[0]
                polarity = mg[1]['polarity'].iloc[0]
                feats = morph.Features.create([ff for f in [gender, number, person, tense, binyan, polarity] for ff in f.split("|") if ff != '_'])
                m = morph.Morpheme(form, lemma, tag, feats)
                if mtype == 'pref':
                    prefixes.append(m)
                elif mtype == 'suff':
                    suffixes.append(m)
                else:
                    hosts.append(m)
            a = morph.Analysis(prefixes, hosts, suffixes)
            token_analyses.append(a)
            is_gold = ag[1]['is_gold'].iloc[0]
            if is_gold:
                gold_token_analyses.append(a)
        lattice[token_idx] = token_analyses
        gold_lattice[token_idx] = gold_token_analyses
    return nlp.Sentence(tokens, lattice, gold_lattice)


class MorphDataset:

    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.df = df
        self.max_morpheme_num = df.morpheme_idx.max() + 1
        self.max_token_num = df.token_idx.max() + 1

    def __getitem__(self, idx: int) -> np.ndarray:

        def resize_token(a: np.ndarray, size: int) -> np.ndarray:
            pad_size = size - a.shape[0]
            if pad_size == 0:
                return a
            npad = ((0, pad_size), (0, 0))
            return np.pad(a, pad_width=npad, mode='constant', constant_values=0)

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
            garr = resize_token(g[1].loc[:, _lattice_columns].to_numpy(), self.max_morpheme_num)
            lattice.append(garr)
        # return resize_lattice(np.stack(lattice), self.max_token_num)
        return np.stack(lattice)

    def __len__(self) -> int:
        return self.df.sent_idx.nunique()

    def save(self, path: Path):
        print(f"Saving morpheme dataset to {path}")
        self.df.to_csv(str(path))

    @classmethod
    def load(cls, name: str, path: Path):
        print(f"Loading morpheme dataset from {path}")
        df = pd.read_csv(str(path))
        return cls(name, df)


def get_morph_dataset_partition(partition_name: str, home_path: Path, vocab: MorphVocab, hebtb: Treebank) -> MorphDataset:
    path = home_path / f'data/processed/spmrl/hebtb-morph-dataset/{partition_name}.csv'
    if path.exists():
        return MorphDataset.load(partition_name, path)
    if partition_name == 'dev-inf':
        df = _dataframe(hebtb.infused_dev_sentences, vocab, False, True)
    elif partition_name == 'test-inf':
        df = _dataframe(hebtb.infused_test_sentences, vocab, False, True)
    elif partition_name == 'dev-uninf':
        df = _dataframe(hebtb.dev_sentences, vocab, False, False)
    elif partition_name == 'test-uninf':
        df = _dataframe(hebtb.test_sentences, vocab, False, False)
    else:
        df = _dataframe(hebtb.infused_train_sentences, vocab, True, True)
    ds = MorphDataset(partition_name, df)
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
    vocab_file_path = Path('data/processed/spmrl/hebtb-morph-vocab/vocab.pickle')
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
        tb_vocab = MorphVocab.load(vocab_file_path)
    else:
        tb_vocab = MorphVocab(tb_sentences)
        tb_vocab.save(vocab_file_path)
    print("Vocab tokens: {}".format(len(tb_vocab.tokens)))
    print("Vocab forms: {}".format(len(tb_vocab.forms)))
    print("Vocab lemmas: {}".format(len(tb_vocab.lemmas)))
    print("Vocab tags: {}".format(len(tb_vocab.tags)))
    print("Vocab feats: {}".format(len(tb_vocab.feats)))

    train_ds = get_morph_dataset_partition('train-inf', home_path, tb_vocab, hebtb)
    dev_inf_ds = get_morph_dataset_partition('dev-inf', home_path, tb_vocab, hebtb)
    test_inf_ds = get_morph_dataset_partition('test-inf', home_path, tb_vocab, hebtb)
    dev_uninf_ds = get_morph_dataset_partition('dev-uninf', home_path, tb_vocab, hebtb)
    test_uninf_ds = get_morph_dataset_partition('test-uninf', home_path, tb_vocab, hebtb)
    print("Train infused dataset: {}".format(len(train_ds)))
    print("Dev infused dataset: {}".format(len(dev_inf_ds)))
    print("Test infused dataset: {}".format(len(test_inf_ds)))
    print("Dev uninfused dataset: {}".format(len(dev_uninf_ds)))
    print("Test uninfused dataset: {}".format(len(test_uninf_ds)))


if __name__ == "__main__":
    main()
