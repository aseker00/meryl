import pickle
from collections import defaultdict
from pathlib import Path
from .. import morph
from dateutil.parser import parse as date_parse


_punct = {
    '"': ("\"", ['yyQUOT'], []),
    ',': (",", ['yyCM'], []),
    '.': (".", ['yyDOT'], []),
    '(': ("(", ['yyLRB'], []),
    ')': (")", ['yyRRB'], []),
    ':': (":", ['yyCLN'], []),
    '-': ("-", ['yyDASH'], []),
    '?': ("?", ['yyQM'], []),
    '...': ("...", ['yyELPS'], []),
    '!': ("!", ['yyEXCL'], []),
    ';': (";", ['yySCLN'], [])
}


def is_punct(analysis: morph.Analysis) -> bool:
    return analysis.hosts and analysis.hosts[0].form in _punct


def is_number(token) -> bool:
    try:
        float(token.replace(',', ''))
        return True
    except ValueError:
        return False


def is_date(token) -> bool:
    try:
        date_parse(token)
        return True
    except ValueError:
        return False


def _load_preflex(bgupreflex_file: str) -> dict:
    bgupreflex = defaultdict(list)
    with open(bgupreflex_file, "r+", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            token = parts[0]
            token_morph_parts = parts[1:]
            values = []
            it = iter(token_morph_parts)
            morph_parts = list(zip(it, it))
            for morph_part in morph_parts:
                lemmas = morph_part[0].split('^')
                morph_analysis = morph_part[1].split(':')
                pref_analysis = morph_analysis[0].split('+')
                values.append((lemmas, pref_analysis))
            bgupreflex[token].extend(values)
    return bgupreflex


def _load_lex(bgulex_file: str) -> dict:
    bgulex = defaultdict(list)
    with open(bgulex_file, "r+", encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            token = parts[0]
            # token_morph_parts = parts[1:]
            token_morph_parts = parts[-1:0:-1]
            values = []
            it = iter(token_morph_parts)
            morph_parts = list(zip(it, it))
            for morph_part in morph_parts:
                # lemmas = morph_part[0].split('^')
                lemma = morph_part[0]
                morph_analysis = morph_part[1].split(':')
                if morph_analysis[0] and lemma != token and token[0] == 'ה':
                    continue
                # pref_analysis = morph_analysis[0].split('-')
                host_analysis = morph_analysis[1].split('-')
                suff_analysis = morph_analysis[2].split('-')
                suff_analysis = list(filter(None, suff_analysis))
                values.append((lemma, host_analysis, suff_analysis))
            if values:
                bgulex[token].extend(values)
    return bgulex


def _load_sufflex() -> dict:
    bgusufflex = defaultdict(list)
    bgusufflex['י'] = [('S_PP', morph.Features(morph.Gender.MALE_FEMALE, morph.Number.SINGULAR, morph.Person.FIRST))]
    bgusufflex['יי'] = [('S_PP', morph.Features(morph.Gender.MALE_FEMALE, morph.Number.SINGULAR, morph.Person.FIRST))]
    bgusufflex['ינו'] = [('S_PP', morph.Features(morph.Gender.MALE_FEMALE, morph.Number.PLURAL, morph.Person.FIRST))]

    bgusufflex['ך'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR, morph.Person.SECOND)),
                       ('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.SINGULAR, morph.Person.SECOND))]
    # bgusufflex['ך'] = [('S_PP', morph.Features(morph.Gender.MALE_FEMALE, morph.Number.SINGULAR, morph.Person.SECOND))]
    bgusufflex['יך'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR, morph.Person.SECOND)),
                        ('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.SINGULAR, morph.Person.SECOND))]
    # bgusufflex['יך'] = [('S_PP', morph.Features(morph.Gender.MALE_FEMALE, morph.Number.SINGULAR, morph.Person.SECOND))]
    bgusufflex['ייך'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.SINGULAR, morph.Person.SECOND))]
    bgusufflex['יכם'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.PLURAL, morph.Person.SECOND))]
    bgusufflex['יכן'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.PLURAL, morph.Person.SECOND))]
    bgusufflex['כם'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.PLURAL, morph.Person.SECOND))]
    bgusufflex['כן'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.PLURAL, morph.Person.SECOND))]

    bgusufflex['ה'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.SINGULAR, morph.Person.THIRD))]
    bgusufflex['ו'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR, morph.Person.THIRD))]
    bgusufflex['יה'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.SINGULAR, morph.Person.THIRD))]
    bgusufflex['יו'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.SINGULAR, morph.Person.THIRD))]
    bgusufflex['ם'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.PLURAL, morph.Person.THIRD))]
    bgusufflex['ן'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.PLURAL, morph.Person.THIRD))]
    bgusufflex['הם'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.PLURAL, morph.Person.THIRD))]
    bgusufflex['הן'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.PLURAL, morph.Person.THIRD))]
    bgusufflex['יהם'] = [('S_PP', morph.Features(morph.Gender.MALE, morph.Number.PLURAL, morph.Person.THIRD))]
    bgusufflex['יהן'] = [('S_PP', morph.Features(morph.Gender.FEMALE, morph.Number.PLURAL, morph.Person.THIRD))]
    return bgusufflex


class LexicalEntry:

    def __init__(self, form: str, analyses: list):
        self.form = form
        self.analyses = analyses

    def __getitem__(self, key) -> morph.Analysis:
        return self.analyses[key]

    def __iter__(self):
        return iter(self.analyses)

    def __len__(self):
        return len(self.analyses)

    def __str__(self):
        # return self.word + ': ' + str(self._analyses)
        return str(self.analyses)

    def __repr__(self):
        return self.__str__()


class Lexicon:

    def __init__(self, file_paths):
        bgupreflex_file = file_paths['pref-lex']
        bgulex_file = file_paths['lex']
        print("Lexicon: loading preflex ...")
        self.bgupreflex = _load_preflex(bgupreflex_file)
        print("Lexicon: {} entries: {}".format('preflex', len(self.bgupreflex)))
        print("Lexicon: loading lex ...")
        self.bgulex = _load_lex(bgulex_file)
        print("Lexicon: {} entries: {}".format('lex', len(self.bgulex)))
        print("Lexicon: loading sufflex ...")
        self.bgusufflex = _load_sufflex()
        print("Lexicon: {} entries: {}".format('sufflex', len(self.bgusufflex)))

        self.preflex_analyses = {}
        for k in self.bgupreflex:
            self.preflex_analyses[k] = [self._parse_pref_morph_analysis(*a) for a in self.bgupreflex[k]]
        self.sufflex_analyses = {}
        for k in self.bgusufflex:
            self.sufflex_analyses[k] = [self._parse_suff_morph_analysis(*a) for a in self.bgusufflex[k]]

        self.lex_entries = {}
        for k in _punct:
            a = self._parse_morph_analysis(k, *_punct[k])
            self.lex_entries[k] = LexicalEntry(k, [a])

    def entry(self, token: str) -> LexicalEntry:
        if token not in self.lex_entries:
            self.lex_entries[token] = LexicalEntry(token, self._get_token_analyses(token))
        return self.lex_entries[token]

    def save(self, path: Path):
        print(f"Saving lexicon to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading lexicon from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)

    def _get_token_analyses(self, token: str) -> list:
        res = [morph.Analysis(a.prefixes, a.hosts, a.suffixes) for a in self._get_analyses(token, with_defaults=True)]
        prefixed_analyses = self._get_prefixed_token_analyses(token)
        res.extend(prefixed_analyses)
        return res

    def _get_suffixed_token_analyses(self, token: str) -> list:
        res = []
        for i in range(len(token) - 1, len(token) - 4, -1):
            suffix = token[i:]
            suff_analyses = self._get_suffix_analyses(suffix)
            if not suff_analyses:
                continue
            sub_token = token[:i]
            if not sub_token:
                continue
            sub_token_analyses = self._get_token_analyses(sub_token)
            if not sub_token_analyses:
                continue
            for token_analysis in sub_token_analyses:
                if token_analysis.prefixes:
                    continue
                if token_analysis.suffixes:
                    continue
                for host in token_analysis.hosts:
                    # TODO: add AT+PRP suffixes for VB, e.g. יעבירם
                    if host.tag == 'NN' or host.tag == 'NNT':
                        for suff_analysis in suff_analyses:
                            a = morph.Analysis(token_analysis.prefixes, token_analysis.hosts, suff_analysis.suffixes)
                            res.append(a)
        return res

    def _get_prefixed_token_analyses(self, token: str) -> list:
        res = []
        for i in range(1, len(token)):
            prefix = token[:i]
            pref_analyses = self._get_prefix_analyses(prefix)
            if not pref_analyses:
                continue
            form = token[i:]
            form_analyses = self._get_analyses(form, with_defaults=True)
            if not form_analyses:
                continue
            for form_analysis in form_analyses:
                for pref_analysis in pref_analyses:
                    a = morph.Analysis(pref_analysis.prefixes, form_analysis.hosts, form_analysis.suffixes)
                    res.append(a)
        return res

    def _get_prefix_analyses(self, prefix) -> list:
        return self.preflex_analyses.get(prefix, [])

    def _get_suffix_analyses(self, suffix) -> list:
        return self.sufflex_analyses.get(suffix, [])

    def _get_analyses(self, form, with_defaults=False) -> list:
        analyses = []
        if form in self.bgulex:
            analyses = [self._parse_morph_analysis(form, *a) for a in self.bgulex[form]]
        elif form[0] in _punct:
            punct_hosts = self.lex_entries[form[0]].analyses[0].hosts
            if len(form) > 1:
                if form[1:] in self.bgulex:
                    analyses = [self._parse_morph_analysis(form[1:], *a) for a in self.bgulex[form[1:]]]
                elif is_number(form[1:]):
                    analyses = morph.default_cd_analyses(form[1:])
                elif is_date(form[1:]):
                    analyses = morph.default_ncd_analyses(form[1:])
                else:
                    analyses = morph.default_nn_analyses(form[1:])
                    analyses.extend(morph.default_nnp_analyses(form[1:]))
                    suffixed_analyeses = self._get_suffixed_token_analyses(form[1:])
                    analyses.extend(suffixed_analyeses)
                for a in analyses:
                    a.hosts = punct_hosts + a.hosts
        elif is_number(form):
            analyses = morph.default_cd_analyses(form)
        elif is_date(form):
            analyses = morph.default_ncd_analyses(form)
        elif with_defaults:
            analyses = morph.default_nn_analyses(form)
            analyses.extend(morph.default_nnp_analyses(form))
            suffixed_analyeses = self._get_suffixed_token_analyses(form)
            analyses.extend(suffixed_analyeses)
        analyses_set = list(set(analyses))
        return analyses_set

    def _parse_pref_morph_analysis(self, lemmas: list, tags: list) -> morph.Analysis:
        pref_morphemes = []
        for lemma, tag in iter(zip(lemmas, tags)):
            p = morph.prefix(lemma, lemma, tag)
            pref_morphemes.append(p)
        return morph.Analysis(pref_morphemes, [], [])

    def _parse_suff_morph_analysis(self, tag: str, feats: morph.Features) -> morph.Analysis:
        s = morph.suffix(tag, feats)
        return morph.Analysis([], [], [s])

    def _parse_morph_analysis(self, form: str, lemma: str, tag_feats: list, suff_tag_feats: list) -> morph.Analysis:
        pref_morphemes = []
        host_morphemes = []
        suff_morphemes = []
        tag = tag_feats[0]
        features = morph.features(tag_feats[1:])
        h = morph.host(form, lemma, tag, features)
        host_morphemes.append(h)
        if suff_tag_feats:
            suff_tag = suff_tag_feats[0]
            suff_features = morph.features(suff_tag_feats[1:])
            s = morph.suffix(suff_tag, suff_features)
            if s is not None:
                h.form = lemma
                suff_morphemes.append(s)
        return morph.Analysis(pref_morphemes, host_morphemes, suff_morphemes)
