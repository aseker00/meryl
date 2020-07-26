import pickle
from collections import defaultdict
from pathlib import Path

from src.processing import nlp, morph


def _analysis_morphemes(analysis: morph.Analysis) -> list:
    return analysis.prefixes + analysis.hosts + analysis.suffixes


def _sentence_morphemes(sent: nlp.Sentence) -> list:
    return [m for token_id in sent.lattice for analysis in sent.lattice[token_id] for m in _analysis_morphemes(analysis)]


def _sentence_gold_morphemes(sent: nlp.Sentence) -> list:
    return [m for token_id in sent.lattice for m in _analysis_morphemes(sent.analysis(token_id))]


# def _sentence_forms(sent: nlp.Sentence) -> set:
#     lattice_forms = {m.form for m in _sentence_morphemes(sent)}
#     gold_lattice_forms = {m.form for m in _sentence_gold_morphemes(sent)}
#     return lattice_forms.union(gold_lattice_forms)
def _sentence_forms(sent: nlp.Sentence) -> set:
    return {m.form for m in _sentence_morphemes(sent) + _sentence_gold_morphemes(sent)}


# def _sentence_lemmas(sent: nlp.Sentence) -> set:
#     lattice_lemmas = {m.lemma for m in _sentence_morphemes(sent)}
#     gold_lattice_lemmas = {m.lemma for m in _sentence_gold_morphemes(sent)}
#     return lattice_lemmas.union(gold_lattice_lemmas)
def _sentence_lemmas(sent: nlp.Sentence) -> set:
    return {m.lemma for m in _sentence_morphemes(sent) + _sentence_gold_morphemes(sent)}


# def _sentence_tags(sent: nlp.Sentence) -> set:
#     lattice_tags = {m.tag for m in _sentence_morphemes(sent)}
#     gold_lattice_tags = {m.tag for m in _sentence_gold_morphemes(sent)}
#     return lattice_tags.union(gold_lattice_tags)
def _sentence_tags(sent: nlp.Sentence) -> set:
    return {m.tag for m in _sentence_morphemes(sent) + _sentence_gold_morphemes(sent)}


def _sentence_feats(sent: nlp.Sentence) -> dict:
    feats = defaultdict(set)
    for m in _sentence_morphemes(sent) + _sentence_gold_morphemes(sent):
        for f in m.feats:
            fstr = str(m.feats[f]) if m.feats[f] else '_'
            feats[f].add(fstr)
    return feats


def _sentence_tokens(sent: nlp.Sentence) -> set:
    return set(sent.tokens)


class MorphVocab:

    def __init__(self, sentences: list = None):
        self.tokens, self.forms, self.lemmas, self.tags, self.feats = [], [], [], [], []
        self.token2id, self.form2id, self.lemma2id, self.tag2id, self.feat2id = {}, {}, {}, {}, {}
        if sentences:
            self._init_sentences(sentences)

    def save(self, path: Path):
        print(f"Saving vocab to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading vocab from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)

    def _init_sentences(self, sentences: list):
        tokens = {token for sent in sentences for token in _sentence_tokens(sent)}
        forms = {form for sent in sentences for form in _sentence_forms(sent)}
        lemmas = {lemma for sent in sentences for lemma in _sentence_lemmas(sent)}
        tags = {tag for sent in sentences for tag in _sentence_tags(sent)}
        feats = set()
        for sent in sentences:
            sent_feats = _sentence_feats(sent)
            for f in sent_feats:
                feats.update(sent_feats[f])
        self.tokens = ['<PAD>', '<SOS>', '<ET>'] + [token for token in sorted(tokens)]
        self.forms = ['<PAD>', '<SOS>', '<ET>'] + [form for form in sorted(forms)]
        self.lemmas = ['<PAD>', '<SOS>', '<ET>'] + [lemma for lemma in sorted(lemmas)]
        self.tags = ['<PAD>', '<SOS>', '<ET>'] + [tag for tag in sorted(tags)]
        self.feats = ['<PAD>', '<SOS>', '<ET>'] + [feat for feat in sorted(feats)]
        self.token2id = {v: k for k, v in enumerate(self.tokens)}
        self.form2id = {v: k for k, v in enumerate(self.forms)}
        self.lemma2id = {v: k for k, v in enumerate(self.lemmas)}
        self.tag2id = {v: k for k, v in enumerate(self.tags)}
        self.feat2id = {v: k for k, v in enumerate(self.feats)}

    def update(self, sent: nlp.Sentence) -> (set, set, set):
        sent_tokens = _sentence_tokens(sent)
        sent_forms = _sentence_forms(sent)
        sent_lemmas = _sentence_lemmas(sent)
        new_tokens = sent_tokens - self.token2id.keys()
        new_forms = sent_forms - self.form2id.keys()
        new_lemmas = sent_lemmas - self.lemma2id.keys()
        for token in new_tokens:
            self.token2id[token] = len(self.tokens)
            self.tokens.append(token)
        for form in new_forms:
            self.form2id[form] = len(self.forms)
            self.forms.append(form)
        for lemma in new_lemmas:
            self.lemma2id[lemma] = len(self.lemmas)
            self.lemmas.append(lemma)
        return new_tokens, new_forms, new_lemmas
