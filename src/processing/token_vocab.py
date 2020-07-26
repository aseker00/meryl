import pickle
from pathlib import Path
from src.processing import nlp


def _sentence_analyses_pref_forms(sent: nlp.Sentence) -> list:
    return [a.pref_forms for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_host_forms(sent: nlp.Sentence) -> list:
    return [a.host_forms for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_suff_forms(sent: nlp.Sentence) -> list:
    return [a.suff_forms for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_pref_lemmas(sent: nlp.Sentence) -> list:
    return [a.pref_lemmas for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_host_lemmas(sent: nlp.Sentence) -> list:
    return [a.host_lemmas for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_suff_lemmas(sent: nlp.Sentence) -> list:
    return [a.suff_lemmas for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_pref_tags(sent: nlp.Sentence) -> list:
    return [a.pref_tags for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_host_tags(sent: nlp.Sentence) -> list:
    return [a.host_tags for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_suff_tags(sent: nlp.Sentence) -> list:
    return [a.suff_tags for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_pref_feats(sent: nlp.Sentence) -> list:
    return [a.pref_feats for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_host_feats(sent: nlp.Sentence) -> list:
    return [a.host_feats for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_analyses_suff_feats(sent: nlp.Sentence) -> list:
    return [a.suff_feats for token_id in sent.lattice for a in sent.lattice[token_id]]


def _sentence_gold_analyses_pref_forms(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).pref_forms for token_id in sent.lattice]


def _sentence_gold_analyses_host_forms(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).host_forms for token_id in sent.lattice]


def _sentence_gold_analyses_suff_forms(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).suff_forms for token_id in sent.lattice]


def _sentence_gold_analyses_pref_lemmas(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).pref_lemmas for token_id in sent.lattice]


def _sentence_gold_analyses_host_lemmas(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).host_lemmas for token_id in sent.lattice]


def _sentence_gold_analyses_suff_lemmas(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).suff_lemmas for token_id in sent.lattice]


def _sentence_gold_analyses_pref_tags(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).pref_tags for token_id in sent.lattice]


def _sentence_gold_analyses_host_tags(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).host_tags for token_id in sent.lattice]


def _sentence_gold_analyses_suff_tags(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).suff_tags for token_id in sent.lattice]


def _sentence_gold_analyses_pref_feats(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).pref_feats for token_id in sent.lattice]


def _sentence_gold_analyses_host_feats(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).host_feats for token_id in sent.lattice]


def _sentence_gold_analyses_suff_feats(sent: nlp.Sentence) -> list:
    return [sent.analysis(token_id).suff_feats for token_id in sent.lattice]


def _sentence_pref_forms(sent: nlp.Sentence) -> set:
    return {tuple(forms) for forms in _sentence_analyses_pref_forms(sent) + _sentence_gold_analyses_pref_forms(sent)}


def _sentence_host_forms(sent: nlp.Sentence) -> set:
    return {tuple(forms) for forms in _sentence_analyses_host_forms(sent) + _sentence_gold_analyses_host_forms(sent)}


def _sentence_suff_forms(sent: nlp.Sentence) -> set:
    return {tuple(forms) for forms in _sentence_analyses_suff_forms(sent) + _sentence_gold_analyses_suff_forms(sent)}


def _sentence_pref_lemmas(sent: nlp.Sentence) -> set:
    return {tuple(lemmas) for lemmas in _sentence_analyses_pref_lemmas(sent) + _sentence_gold_analyses_pref_lemmas(sent)}


def _sentence_host_lemmas(sent: nlp.Sentence) -> set:
    return {tuple(lemmas) for lemmas in _sentence_analyses_host_lemmas(sent) + _sentence_gold_analyses_host_lemmas(sent)}


def _sentence_suff_lemmas(sent: nlp.Sentence) -> set:
    return {tuple(lemmas) for lemmas in _sentence_analyses_suff_lemmas(sent) + _sentence_gold_analyses_suff_lemmas(sent)}


def _sentence_pref_tags(sent: nlp.Sentence) -> set:
    return {tuple(tags) for tags in _sentence_analyses_pref_tags(sent) + _sentence_gold_analyses_pref_tags(sent)}


def _sentence_host_tags(sent: nlp.Sentence) -> set:
    return {tuple(tags) for tags in _sentence_analyses_host_tags(sent) + _sentence_gold_analyses_host_tags(sent)}


def _sentence_suff_tags(sent: nlp.Sentence) -> set:
    return {tuple(tags) for tags in _sentence_analyses_suff_tags(sent) + _sentence_gold_analyses_suff_tags(sent)}


# def _sentence_pref_feats(sent: nlp.Sentence) -> set:
#     pref_feats = set()
#     analyses_pref_feats = {tuple(feats) for feats in _sentence_analyses_pref_feats(sent) + _sentence_gold_analyses_pref_feats(sent)}
#     for analysis_feats in analyses_pref_feats:
#         analysis_fstr = tuple(str(feats) for feats in analysis_feats)
#         pref_feats.add(analysis_fstr)
#     return pref_feats
def _sentence_pref_feats(sent: nlp.Sentence) -> set:
    return {tuple(feats) for feats in _sentence_analyses_pref_feats(sent) + _sentence_gold_analyses_pref_feats(sent)}


# def _sentence_host_feats(sent: nlp.Sentence) -> set:
#     host_feats = set()
#     analyses_host_feats = {tuple(feats) for feats in _sentence_analyses_host_feats(sent) + _sentence_gold_analyses_host_feats(sent)}
#     for analysis_feats in analyses_host_feats:
#         analysis_fstr = tuple(str(feats) for feats in analysis_feats)
#         host_feats.add(analysis_fstr)
#     return host_feats
def _sentence_host_feats(sent: nlp.Sentence) -> set:
    return {tuple(feats) for feats in _sentence_analyses_host_feats(sent) + _sentence_gold_analyses_host_feats(sent)}


# def _sentence_suff_feats(sent: nlp.Sentence) -> set:
#     suff_feats = set()
#     analyses_suff_feats = {tuple(feats) for feats in _sentence_analyses_suff_feats(sent) + _sentence_gold_analyses_suff_feats(sent)}
#     for analysis_feats in analyses_suff_feats:
#         analysis_fstr = tuple(str(feats) for feats in analysis_feats)
#         suff_feats.add(analysis_fstr)
#     return suff_feats
def _sentence_suff_feats(sent: nlp.Sentence) -> set:
    return {tuple(feats) for feats in _sentence_analyses_suff_feats(sent) + _sentence_gold_analyses_suff_feats(sent)}


def _sentence_tokens(sent: nlp.Sentence) -> set:
    return set(sent.tokens)


class TokenVocab:

    def __init__(self, sentences: list = None):
        self.tokens = []
        self.pref_forms, self.pref_lemmas, self.pref_tags, self.pref_feats = [], [], [], []
        self.host_forms, self.host_lemmas, self.host_tags, self.host_feats = [], [], [], []
        self.suff_forms, self.suff_lemmas, self.suff_tags, self.suff_feats = [], [], [], []
        self.token2id = {}
        self.pref_form2id, self.pref_lemma2id, self.pref_tag2id, self.pref_feat2id = {}, {}, {}, {}
        self.host_form2id, self.host_lemma2id, self.host_tag2id, self.host_feat2id = {}, {}, {}, {}
        self.suff_form2id, self.suff_lemma2id, self.suff_tag2id, self.suff_feat2id = {}, {}, {}, {}
        if sentences:
            tokens = {token for sent in sentences for token in _sentence_tokens(sent)}
            self.tokens = ['<PAD>', '<SOS>', '<ET>'] + [token for token in sorted(tokens)]
            self.token2id = {v: k for k, v in enumerate(self.tokens)}
            self._init_pref_sentences(sentences)
            self._init_host_sentences(sentences)
            self._init_suff_sentences(sentences)

    def save(self, path: Path):
        print(f"Saving vocab to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading vocab from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)

    def _init_pref_sentences(self, sentences: list):
        pref_forms = {form for sent in sentences for form in _sentence_pref_forms(sent)}
        pref_lemmas = {lemma for sent in sentences for lemma in _sentence_pref_lemmas(sent)}
        pref_tags = {tag for sent in sentences for tag in _sentence_pref_tags(sent)}
        pref_feats = set()
        for sent in sentences:
            sent_feats = _sentence_pref_feats(sent)
            pref_feats.update(sent_feats)
        self.pref_forms = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [form for form in sorted(pref_forms)]
        self.pref_lemmas = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [lemma for lemma in sorted(pref_lemmas)]
        self.pref_tags = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [tag for tag in sorted(pref_tags)]
        self.pref_feats = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [feat for feat in pref_feats]
        self.pref_form2id = {v: k for k, v in enumerate(self.pref_forms)}
        self.pref_lemma2id = {v: k for k, v in enumerate(self.pref_lemmas)}
        self.pref_tag2id = {v: k for k, v in enumerate(self.pref_tags)}
        self.pref_feat2id = {v: k for k, v in enumerate(self.pref_feats)}
        
    def _init_host_sentences(self, sentences: list):
        host_forms = {form for sent in sentences for form in _sentence_host_forms(sent)}
        # for i, sent in enumerate(sentences):
        #     sent_id = i + 1
        #     if sent_id == 591:
        #         lemmas = _sentence_host_lemmas(sent)
        #         print(lemmas)
        host_lemmas = {lemma for sent in sentences for lemma in _sentence_host_lemmas(sent)}
        host_tags = {tag for sent in sentences for tag in _sentence_host_tags(sent)}
        host_feats = set()
        for sent in sentences:
            sent_feats = _sentence_host_feats(sent)
            host_feats.update(sent_feats)
        self.host_forms = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [form for form in sorted(host_forms)]
        self.host_lemmas = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [lemma for lemma in sorted(host_lemmas)]
        self.host_tags = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [tag for tag in sorted(host_tags)]
        self.host_feats = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [feat for feat in host_feats]
        self.host_form2id = {v: k for k, v in enumerate(self.host_forms)}
        self.host_lemma2id = {v: k for k, v in enumerate(self.host_lemmas)}
        self.host_tag2id = {v: k for k, v in enumerate(self.host_tags)}
        self.host_feat2id = {v: k for k, v in enumerate(self.host_feats)}
        
    def _init_suff_sentences(self, sentences: list):
        suff_forms = {form for sent in sentences for form in _sentence_suff_forms(sent)}
        suff_lemmas = {lemma for sent in sentences for lemma in _sentence_suff_lemmas(sent)}
        suff_tags = {tag for sent in sentences for tag in _sentence_suff_tags(sent)}
        suff_feats = set()
        for sent in sentences:
            sent_feats = _sentence_suff_feats(sent)
            suff_feats.update(sent_feats)
        self.suff_forms = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [form for form in sorted(suff_forms)]
        self.suff_lemmas = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [lemma for lemma in sorted(suff_lemmas)]
        self.suff_tags = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [tag for tag in sorted(suff_tags)]
        self.suff_feats = [('<PAD>', ), ('<SOS>', ), ('<ET>', )] + [feat for feat in suff_feats]
        self.suff_form2id = {v: k for k, v in enumerate(self.suff_forms)}
        self.suff_lemma2id = {v: k for k, v in enumerate(self.suff_lemmas)}
        self.suff_tag2id = {v: k for k, v in enumerate(self.suff_tags)}
        self.suff_feat2id = {v: k for k, v in enumerate(self.suff_feats)}

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
