from enum import Enum
from copy import deepcopy


class Polarity(Enum):
    POSITIVE = 1
    NEGATIVE = 2

    def __str__(self) -> str:
        return "pol={}".format(self.name)


class Binyan(Enum):
    HIFIL = 1
    HITPAEL = 2
    HUFAL = 3
    NIFAL = 4
    PAAL = 5
    PIEL = 6
    PUAL = 7

    def __str__(self) -> str:
        return "binyan={}".format(self.name)


class Tense(Enum):
    BEINONI = 1
    FUTURE = 2
    IMPERATIVE = 3
    PAST = 4
    TOINFINITIVE = 5

    def __str__(self) -> str:
        return "tense={}".format(self.name)


class Person(Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3
    ALL = 4

    def __str__(self) -> str:
        return "per={}".format('A' if self.value == 4 else self.value)


class Number(Enum):
    SINGULAR = 1
    DOUBLE = 2
    PLURAL = 3
    SINGULAR_PLURAL = 4
    DOUBLE_PLURAL = 5

    def __str__(self) -> str:
        return "|".join(["num={}".format(part[0]) for part in self.name.split("_")])


class Gender(Enum):
    MALE = 1
    FEMALE = 2
    MALE_FEMALE = 3

    def __str__(self) -> str:
        return "|".join(["gen={}".format(part[0]) for part in self.name.split("_")])


gender_features = {
    'M': Gender.MALE,
    'F': Gender.FEMALE, 
    'FM': Gender.MALE_FEMALE, 
    'MF': Gender.MALE_FEMALE}
number_features = {
    'S': Number.SINGULAR, 
    'D': Number.DOUBLE, 
    'P': Number.PLURAL, 
    'SP': Number.SINGULAR_PLURAL,
    'PS': Number.SINGULAR_PLURAL,
    'DP': Number.DOUBLE_PLURAL, # e.g. שוליים
    'PD': Number.DOUBLE_PLURAL}
person_features = {
    '1': Person.FIRST, 
    '2': Person.SECOND, 
    '3': Person.THIRD, 
    'A': Person.ALL}
tense_features = {name: member for name, member in Tense.__members__.items()}
binyan_features = {name: member for name, member in Binyan.__members__.items()}
polarity_features = {name: member for name, member in Polarity.__members__.items()}


class Features:

    def __init__(self, gender: Gender = None, number: Number = None, person: Person = None,
                 tense: Tense = None, binyan: Binyan = None, polarity: Polarity = None):
        self.gender = gender
        self.number = number
        self.person = person
        self.tense = tense
        self.binyan = binyan
        self.polarity = polarity
        self._items = {'gen': self.gender, 'num': self.number, 'per': self.person,
                       'tense': self.tense, 'binyan': self.binyan, 'pol': self.polarity}

    def __len__(self):
        count = 0
        if self.gender:
            count += 1
        if self.number:
            count += 1
        if self.person:
            count += 1
        if self.tense:
            count += 1
        if self.binyan:
            count += 1
        if self.polarity:
            count += 1
        return count

    def __eq__(self, other) -> bool:
        if not isinstance(other, Features):
            return NotImplemented
        return self.gender == other.gender \
            and self.number == other.number \
            and self.person == other.person \
            and self.tense == other.tense \
            and self.binyan == other.binyan \
            and self.polarity == other.polarity

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, key):
        return self._items[key]

    def __contains__(self, key):
        return key in self._items

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __bool__(self):
        return bool(self.gender or self.number or self.person or self.tense or self.binyan or self.polarity)

    def __deepcopy__(self):
        return Features(self.gender, self.number, self.person, self.tense, self.binyan, self.polarity)

    def __str__(self) -> str:
        gender_str = str(self.gender) if self.gender else None
        number_str = str(self.number) if self.number else None
        person_str = str(self.person) if self.person else None
        tense_str = str(self.tense) if self.tense else None
        binyan_str = str(self.binyan) if self.binyan else None
        pol_str = str(self.polarity) if self.polarity else None
        result = "|".join(filter(None, [gender_str, number_str, person_str, tense_str, binyan_str, pol_str]))
        if not result:
            return "_"
        return result

    @classmethod
    def create(cls, feats: list):
        if not feats:
            return EMPTY_FEATURES
        gender = None
        number = None
        person = None
        tense = None
        binyan = None
        polarity = None
        for feat in feats:
            name, value = feat.split("=")
            if name == 'gen':
                if gender:
                    gender = Gender.MALE_FEMALE
                else:
                    gender = gender_features[value]
            elif name == 'num':
                if number:
                    add_number = number_features[value]
                    if add_number == Number.SINGULAR:
                        number = Number.SINGULAR_PLURAL
                    elif add_number == Number.PLURAL:
                        number = Number.DOUBLE_PLURAL
                    elif add_number == Number.DOUBLE:
                        number = Number.DOUBLE_PLURAL
                else:
                    number = number_features[value]
            elif name == 'per':
                person = person_features[value]
            elif name == 'tense':
                tense = tense_features[value]
            elif name == 'binyan':
                binyan = binyan_features[value]
            elif name == 'pol':
                polarity = polarity_features[value]
        return Features(gender, number, person, tense, binyan, polarity)


EMPTY_FEATURES = Features(None, None, None, None, None, None)


def features(feats: list) -> Features:
    if not feats:
        return EMPTY_FEATURES
    gender = None
    number = None
    person = None
    tense = None
    binyan = None
    polarity = None
    for feat in feats:
        if feat in gender_features:
            gender = gender_features[feat]
        elif feat in number_features:
            number = number_features[feat]
        elif feat in person_features:
            person = person_features[feat]
        elif feat in tense_features:
            tense = tense_features[feat]
        elif feat in binyan_features:
            binyan = binyan_features[feat]
        elif feat in polarity_features:
            polarity = polarity_features[feat]
    return Features(gender, number, person, tense, binyan, polarity)


def _get_subset_features(f: Features, feature_enums: list) -> Features:
    gender = None
    number = None
    person = None
    tense = None
    binyan = None
    polarity = None
    for feature_enum in feature_enums:
        if issubclass(feature_enum, Gender):
            gender = f.gender
        elif issubclass(feature_enum, Number):
            number = f.number
        elif issubclass(feature_enum, Person):
            person = f.person
        elif issubclass(feature_enum, Tense):
            tense = f.tense
        elif issubclass(feature_enum, Binyan):
            binyan = f.binyan
        elif issubclass(feature_enum, Polarity):
            polarity = f.polarity
    return Features(gender, number, person, tense, binyan, polarity)


class Morpheme:

    def __init__(self, form: str, lemma: str, tag: str, feats: Features):
        self.form = form
        self.lemma = lemma
        self.tag = tag
        self.feats = feats

    def __eq__(self, other) -> bool:
        if not isinstance(other, Morpheme):
            return NotImplemented
        return (self.form == other.form and
                self.lemma == other.lemma and
                self.tag == other.tag and
                self.feats == other.feats)

    def __str__(self):
        # return "{}, {}, {}, {}".format(self.form, self.lemma, self.tag, self.feats)
        return str((self.form, self.lemma, self.tag, self.feats))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__key())

    def __key(self):
        return self.__str__()

    def __deepcopy__(self):
        return Morpheme(self.form, self.lemma, self.tag, deepcopy(self.feats))

    @property
    def msr(self) -> tuple:
        return self.form, self.tag, self.feats


def _get_subset_features_morpheme(m: Morpheme, feature_enums: list) -> Morpheme:
    return Morpheme(m.form, m.lemma, m.tag, _get_subset_features(m.feats, feature_enums))


open_tags = {'NN', 'NNT', 'VB', 'JJ', 'JJT', 'RB', 'NNP', 'BN', 'BNT', 'INTJ', 'NN_S_PP', 'CD', 'TTL', 'CDT', 'NCD', 'ZVL'}

_MA_PREFIX_MORPHEMES = {
    ("ב", 'PREPOSITION'): Morpheme("ב", "ב", 'PREPOSITION', EMPTY_FEATURES),
    # ("ב", 'IN'): Morpheme("ב", "ב", 'PREPOSITION', EMPTY_FEATURES),
    ("כ", 'PREPOSITION'): Morpheme("כ", "כ", 'PREPOSITION', EMPTY_FEATURES),
    # ("כ", 'IN'): Morpheme("כ", "כ", 'PREPOSITION', EMPTY_FEATURES),
    ("ל", 'PREPOSITION'): Morpheme("ל", "ל", 'PREPOSITION', EMPTY_FEATURES),
    # ("ל", 'IN'): Morpheme("ל", "ל", 'PREPOSITION', EMPTY_FEATURES),
    ("מ", 'PREPOSITION'): Morpheme("מ", "מ", 'PREPOSITION', EMPTY_FEATURES),
    # ("מ", 'IN'): Morpheme("מ", "מ", 'PREPOSITION', EMPTY_FEATURES),
    ("כ", 'ADVERB'): Morpheme("כ", "כ", 'ADVERB', EMPTY_FEATURES),
    ("ו", 'CONJ'): Morpheme("ו", "ו", 'CONJ', EMPTY_FEATURES),
    ("ה", 'DEF'): Morpheme("ה", "ה", 'DEF', EMPTY_FEATURES),
    ("ה", 'REL'): Morpheme("ה", "ה", 'REL', EMPTY_FEATURES),
    ("ש", 'REL'): Morpheme("ש", "ש", 'REL', EMPTY_FEATURES),
    ("ש", 'REL-SUBCONJ'): Morpheme("ש", "ש", 'REL', EMPTY_FEATURES),
    ("כש", 'TEMP'): Morpheme("כש", "כש", 'TEMP', EMPTY_FEATURES),
    ("כש", 'TEMP-SUBCONJ'): Morpheme("כש", "כש", 'TEMP', EMPTY_FEATURES),
    ("לכש", 'TEMP-SUBCONJ'): Morpheme("לכש", "לכש", 'TEMP', EMPTY_FEATURES),
    ("מש", 'TEMP'): Morpheme("מש", "מש", 'TEMP', EMPTY_FEATURES),
    ("מש", 'TEMP-SUBCONJ'): Morpheme("מש", "מש", 'TEMP', EMPTY_FEATURES)
}


_MA_SUFFIX_PP_MORPHEMES = {
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST):
        Morpheme("אני", "הוא", 'S_PP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("את", "הוא", 'S_PP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE, Number.SINGULAR, Person.SECOND):
        Morpheme("אתה", "הוא", 'S_PP', Features(Gender.MALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("_", "הוא", 'S_PP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD):
        Morpheme("היא", "הוא", 'S_PP', Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE, Number.SINGULAR, Person.THIRD):
        Morpheme("הוא", "הוא", 'S_PP', Features(Gender.MALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST):
        Morpheme("אנחנו", "הוא", 'S_PP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST)),
    Features(Gender.FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתן", "הוא", 'S_PP', Features(Gender.FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתם", "הוא", 'S_PP', Features(Gender.MALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("_", "הוא", 'S_PP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.FEMALE, Number.PLURAL, Person.THIRD):
        Morpheme("הן", "הוא", 'S_PP', Features(Gender.FEMALE, Number.PLURAL, Person.THIRD)),
    Features(Gender.MALE, Number.PLURAL, Person.THIRD):
        Morpheme("הם", "הוא", 'S_PP', Features(Gender.MALE, Number.PLURAL, Person.THIRD))
}


_MA_SUFFIX_ANP_MORPHEMES = {
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST):
        Morpheme("אני", "הוא", 'S_ANP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("את", "הוא", 'S_ANP', Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE, Number.SINGULAR, Person.SECOND):
        Morpheme("אתה", "הוא", 'S_ANP', Features(Gender.MALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("_", "הוא", 'S_ANP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD):
        Morpheme("היא", "הוא", 'S_ANP', Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE, Number.SINGULAR, Person.THIRD):
        Morpheme("הוא", "הוא", 'S_ANP', Features(Gender.MALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST):
        Morpheme("אנחנו", "הוא", 'S_ANP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST)),
    Features(Gender.FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתן", "הוא", 'S_ANP', Features(Gender.FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתם", "הוא", 'S_ANP', Features(Gender.MALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("_", "הוא", 'S_ANP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.FEMALE, Number.PLURAL, Person.THIRD):
        Morpheme("הן", "הוא", 'S_ANP', Features(Gender.FEMALE, Number.PLURAL, Person.THIRD)),
    Features(Gender.MALE, Number.PLURAL, Person.THIRD):
        Morpheme("הם", "הוא", 'S_ANP', Features(Gender.MALE, Number.PLURAL, Person.THIRD))
}


_MA_SUFFIX_PRN_MORPHEMES = {
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST):
        Morpheme("אני", "הוא", 'S_PRN', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("את", "הוא", 'S_PRN', Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE, Number.SINGULAR, Person.SECOND):
        Morpheme("אתה", "הוא", 'S_PRN', Features(Gender.MALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND):
        Morpheme("_", "הוא", 'S_PRN', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND)),
    Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD):
        Morpheme("היא", "הוא", 'S_PRN', Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE, Number.SINGULAR, Person.THIRD):
        Morpheme("הוא", "הוא", 'S_PRN', Features(Gender.MALE, Number.SINGULAR, Person.THIRD)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST):
        Morpheme("אנחנו", "הוא", 'S_PRN', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST)),
    Features(Gender.FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתן", "הוא", 'S_PRN', Features(Gender.FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE, Number.PLURAL, Person.SECOND):
        Morpheme("אתם", "הוא", 'S_PRN', Features(Gender.MALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND):
        Morpheme("_", "הוא", 'S_PRN', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND)),
    Features(Gender.FEMALE, Number.PLURAL, Person.THIRD):
        Morpheme("הן", "הוא", 'S_PRN', Features(Gender.FEMALE, Number.PLURAL, Person.THIRD)),
    Features(Gender.MALE, Number.PLURAL, Person.THIRD):
        Morpheme("הם", "הוא", 'S_PRN', Features(Gender.MALE, Number.PLURAL, Person.THIRD))
}


# _MA_SUFFIX_PRP_MORPHEMES = {
#     Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST):
#         Morpheme("אני", "הוא", 'S_PRP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.FIRST)),
#     Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND):
#         Morpheme("את", "הוא", 'S_PRP', Features(Gender.FEMALE, Number.SINGULAR, Person.SECOND)),
#     Features(Gender.MALE, Number.SINGULAR, Person.SECOND):
#         Morpheme("אתה", "הוא", 'S_PRP', Features(Gender.MALE, Number.SINGULAR, Person.SECOND)),
#     Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND):
#         Morpheme("_", "הוא", 'S_PRP', Features(Gender.MALE_FEMALE, Number.SINGULAR, Person.SECOND)),
#     Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD):
#         Morpheme("היא", "הוא", 'S_PRP', Features(Gender.FEMALE, Number.SINGULAR, Person.THIRD)),
#     Features(Gender.MALE, Number.SINGULAR, Person.THIRD):
#         Morpheme("הוא", "הוא", 'S_PRP', Features(Gender.MALE, Number.SINGULAR, Person.THIRD)),
#     Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST):
#         Morpheme("אנחנו", "הוא", 'S_PRP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.FIRST)),
#     Features(Gender.FEMALE, Number.PLURAL, Person.SECOND):
#         Morpheme("אתן", "הוא", 'S_PRP', Features(Gender.FEMALE, Number.PLURAL, Person.SECOND)),
#     Features(Gender.MALE, Number.PLURAL, Person.SECOND):
#         Morpheme("אתם", "הוא", 'S_PRP', Features(Gender.MALE, Number.PLURAL, Person.SECOND)),
#     Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND):
#         Morpheme("_", "הוא", 'S_PRP', Features(Gender.MALE_FEMALE, Number.PLURAL, Person.SECOND)),
#     Features(Gender.FEMALE, Number.PLURAL, Person.THIRD):
#         Morpheme("הן", "הוא", 'S_PRP', Features(Gender.FEMALE, Number.PLURAL, Person.THIRD)),
#     Features(Gender.MALE, Number.PLURAL, Person.THIRD):
#         Morpheme("הם", "הוא", 'S_PRP', Features(Gender.MALE, Number.PLURAL, Person.THIRD))
# }


_MA_MAIN_DEFAULT_SINGULAR_NN_FEATURES = [
    Features(Gender.FEMALE, Number.SINGULAR),
    Features(Gender.MALE, Number.SINGULAR)
]


_MA_MAIN_DEFAULT_PLURAL_NN_FEATURES = [
    Features(Gender.FEMALE, Number.PLURAL),
    Features(Gender.MALE, Number.PLURAL)
]

_MA_MAIN_DEFAULT_SINGULAR_NNP_FEATURES = [
    # EMPTY_FEATURES,
    Features(Gender.FEMALE, Number.SINGULAR),
    Features(Gender.MALE, Number.SINGULAR),
    Features(Gender.MALE_FEMALE, Number.SINGULAR),
]


def prefix(form: str, lemma: str, tag: str) -> Morpheme:
    if (lemma, tag) in _MA_PREFIX_MORPHEMES:
        return _MA_PREFIX_MORPHEMES[(lemma, tag)]
    elif (form, tag) in _MA_PREFIX_MORPHEMES:
        return _MA_PREFIX_MORPHEMES[(form, tag)]
    return None


def host(form: str, lemma: str, tag: str, features: Features) -> Morpheme:
    return Morpheme(form, lemma, tag, features)


def suffix(tag: str, features: Features) -> Morpheme:
    if tag == 'S_PP' and features in _MA_SUFFIX_PP_MORPHEMES:
        return _MA_SUFFIX_PP_MORPHEMES[features]
    elif tag == 'S_ANP' and features in _MA_SUFFIX_ANP_MORPHEMES:
        return _MA_SUFFIX_ANP_MORPHEMES[features]
    elif tag == 'S_PRN' and features in _MA_SUFFIX_PRN_MORPHEMES:
        return _MA_SUFFIX_PRN_MORPHEMES[features]
    # elif tag == 'S_PRP' and features in _MA_SUFFIX_PRP_MORPHEMES:
    #     return _MA_SUFFIX_PRP_MORPHEMES[features]
    return None


class Analysis:

    def __init__(self, prefs: list, hosts: list, suffs: list):
        self.prefixes = prefs
        self.hosts = hosts
        self.suffixes = suffs

    def __eq__(self, other) -> bool:
        if not isinstance(other, Analysis):
            return NotImplemented
        return self.prefixes == other.prefixes \
            and self.hosts == other.hosts \
            and self.suffixes == other.suffixes

    def __str__(self):
        # return "({}, {}, {})".format(self.lemmas_str, self.tags_str, self.feats_str)
        return f"({self.forms_str}, {self.lemmas_str}, {self.tags_str}, {self.feats_str})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __deepcopy__(self):
        return Analysis([deepcopy(pref) for pref in self.prefixes],
                        [deepcopy(host) for host in self.hosts],
                        [deepcopy(suff) for suff in self.suffixes])
    
    @property
    def pref_feats(self) -> list:
        return [m.feats for m in self.prefixes]

    @property
    def host_feats(self) -> list:
        return [m.feats for m in self.hosts]

    @property
    def suff_feats(self) -> list:
        return [m.feats for m in self.suffixes]

    @property
    def pref_tags(self) -> list:
        return [m.tag for m in self.prefixes]

    @property
    def host_tags(self) -> list:
        return [m.tag for m in self.hosts]

    @property
    def suff_tags(self) -> list:
        return [m.tag for m in self.suffixes]

    @property
    def pref_lemmas(self) -> list:
        return [m.lemma for m in self.prefixes]

    @property
    def host_lemmas(self) -> list:
        return [m.lemma for m in self.hosts]

    @property
    def suff_lemmas(self) -> list:
        return [m.lemma for m in self.suffixes]

    @property
    def pref_forms(self) -> list:
        return [m.form for m in self.prefixes]

    @property
    def host_forms(self) -> list:
        return [m.form for m in self.hosts]

    @property
    def suff_forms(self) -> list:
        return [m.form for m in self.suffixes]

    @property
    def pref_feats_str(self) -> str:
        features = [str(m.feats) for m in self.prefixes if str(m.feats) != '_']
        return "|".join(features) if features else "_"

    @property
    def host_feats_str(self) -> str:
        features = [str(m.feats) for m in self.hosts if str(m.feats) != '_']
        return "|".join(features) if features else "_"

    @property
    def suff_feats_str(self) -> str:
        features = ["{0}_{1}".format("suf", s) for m in self.suffixes for s in str(m.feats).split("|") if str(m.feats) != '_']
        return "|".join(features) if features else "_"

    @property
    def feats_str(self) -> str:
        return "||".join('{0}'.format(l) for l in [self.pref_feats_str, self.host_feats_str, self.suff_feats_str])

    @property
    def pref_tags_str(self) -> str:
        tags = [m.tag for m in self.prefixes if m.tag]
        return "+".join(tags)

    @property
    def host_tags_str(self) -> str:
        tags = [m.tag for m in self.hosts if m.tag]
        return "+".join(tags)

    @property
    def suff_tags_str(self) -> str:
        tags = [m.tag for m in self.suffixes if m.tag]
        return "+".join(tags)

    @property
    def tags_str(self) -> str:
        return "++".join('{0}'.format(l) for l in [self.pref_tags_str, self.host_tags_str, self.suff_tags_str])

    @property
    def pref_lemmas_str(self) -> str:
        lemmas = [m.lemma for m in self.prefixes if m.lemma]
        return "^".join(lemmas)

    @property
    def host_lemmas_str(self) -> str:
        lemmas = [m.lemma for m in self.hosts if m.lemma]
        return "^".join(lemmas)

    @property
    def suff_lemmas_str(self) -> str:
        lemmas = [m.lemma for m in self.suffixes if m.lemma]
        return "^".join(lemmas)

    @property
    def lemmas_str(self) -> str:
        return "^^".join('{0}'.format(l) for l in [self.pref_lemmas_str, self.host_lemmas_str, self.suff_lemmas_str])

    @property
    def pref_forms_str(self) -> str:
        forms = [m.form for m in self.prefixes if m.form]
        return "-".join(forms)

    @property
    def host_forms_str(self) -> str:
        forms = [m.form for m in self.hosts if m.form]
        return "-".join(forms)

    @property
    def suff_forms_str(self) -> str:
        forms = [m.form for m in self.suffixes if m.form]
        return "-".join(forms)

    @property
    def forms_str(self) -> str:
        return "--".join('{0}'.format(l) for l in [self.pref_forms_str, self.host_forms_str, self.suff_forms_str])


def analysis_no_lemma(analysis: Analysis) -> Analysis:
    hosts_no_lemma = [Morpheme(host.form, '_', host.tag, host.feats) for host in analysis.hosts]
    return Analysis(analysis.prefixes, hosts_no_lemma, analysis.suffixes)


def _get_subset_features_morphemes_analysis(a: Analysis, feature_enums: list) -> Analysis:
    # pref_morphemes = [_get_subset_features_morpheme(m, feature_enums) for m in a.prefixes]
    host_morphemes = [_get_subset_features_morpheme(m, feature_enums) for m in a.hosts]
    # suff_morphemes = [_get_subset_features_morpheme(m, feature_enums) for m in a.suffixes]
    return Analysis(a.prefixes, host_morphemes, a.suffixes)


def analyses_equals(ma1: list, ma2: list, feature_enums: list) -> bool:
    s1 = sorted([_get_subset_features_morphemes_analysis(ma1, feature_enums) for ma1 in ma1], key=hash)
    s2 = sorted([_get_subset_features_morphemes_analysis(ma2, feature_enums) for ma2 in ma2], key=hash)
    for a1, a2 in zip(s1, s2):
        if not a1 == a2:
            return False
    return True


def analysis_equals(ma1: Analysis, ma2: Analysis, feature_enums: list) -> bool:
    a1 = _get_subset_features_morphemes_analysis(ma1, feature_enums)
    a2 = _get_subset_features_morphemes_analysis(ma2, feature_enums)
    return a1 == a2


def analysis_equals_no_lemma(ma1: Analysis, ma2: Analysis, feature_enums: list) -> bool:
    a1 = _get_subset_features_morphemes_analysis(analysis_no_lemma(ma1), feature_enums)
    a2 = _get_subset_features_morphemes_analysis(analysis_no_lemma(ma2), feature_enums)
    return a1 == a2


def default_cd_analyses(form: str) -> list:
    return [Analysis([], [Morpheme(form, '_', 'CD', EMPTY_FEATURES)], []),
            Analysis([], [Morpheme(form, '_', 'CDT', EMPTY_FEATURES)], [])]


def default_ncd_analyses(form: str) -> list:
    return [Analysis([], [Morpheme(form, '_', 'NCD', EMPTY_FEATURES)], [])]


def default_nn_analyses(form: str) -> list:
    analyses = [Analysis([], [Morpheme(form, '_', 'NN', f)], []) for f in _MA_MAIN_DEFAULT_SINGULAR_NN_FEATURES]
    analyses.extend([Analysis([], [Morpheme(form, '_', 'NNT', f)], []) for f in _MA_MAIN_DEFAULT_SINGULAR_NN_FEATURES])
    if len(form) > 2 and (form[-2:] == 'ות' or form[-2:] == 'ים'):
        analyses.extend([Analysis([], [Morpheme(form, '_', 'NN', f)], []) for f in _MA_MAIN_DEFAULT_PLURAL_NN_FEATURES])
        analyses.extend([Analysis([], [Morpheme(form, '_', 'NNT', f)], []) for f in _MA_MAIN_DEFAULT_PLURAL_NN_FEATURES])
    return analyses


def default_nnp_analyses(form: str) -> list:
    return [Analysis([], [Morpheme(form, '_', 'NNP', f)], []) for f in _MA_MAIN_DEFAULT_SINGULAR_NNP_FEATURES]


# dict from token id to list of Analysis
class Lattice(dict):

    class Edge:
        def __init__(self, from_node_id: int, to_node_id: int, morpheme: Morpheme, morpheme_type: str):
            self.from_node_id = from_node_id
            self.to_node_id = to_node_id
            self.morpheme = morpheme
            self.morpheme_type = morpheme_type

        def __key(self):
            return self.from_node_id, self.to_node_id, self.morpheme

        def __str__(self):
            return str((self.from_node_id, self.to_node_id, self.morpheme, self.morpheme_type))

        def __repr__(self):
            return self.__str__()

        def __hash__(self):
            return hash((self.__key()))

        def __eq__(self, other):
            if isinstance(other, Lattice.Edge):
                return self.__key() == other.__key()
            return NotImplemented

        def msr(self) -> tuple:
            return self.from_node_id, self.to_node_id, self.morpheme.msr

    def __init__(self, *args):
        dict.__init__(self, args)


def copy_lattice(other: Lattice) -> Lattice:
    copy_lattice = Lattice()
    for tid in other:
        copy_analyses = []
        other_analyses = other[tid]
        for other_analysis in other_analyses:
            copy_analysis = deepcopy(other_analysis)
            copy_analyses.append(copy_analysis)
        copy_lattice[tid] = copy_analyses
    return copy_lattice
