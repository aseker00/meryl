import pickle
from pathlib import Path
from src.processing.spmrl import conllx, lexicon as lex
from src.processing import nlp, morph


def _create_features(feats: dict) -> morph.Features:
    gender = morph.gender_features[''.join(feats['gen'])] if 'gen' in feats else None
    number = morph.number_features[''.join(feats['num'])] if 'num' in feats else None
    person = morph.person_features[''.join(feats['per'])] if 'per' in feats else None
    tense = morph.tense_features[''.join(feats['tense'])] if 'tense' in feats else None
    binyan = morph.binyan_features[''.join(feats['binyan'])] if 'binyan' in feats else None
    polarity = morph.polarity_features[''.join(feats['pol'])] if 'pol' in feats else None
    return morph.Features(gender, number, person, tense, binyan, polarity)


def _create_suff_features(feats: dict) -> morph.Features:
    gender = morph.gender_features[''.join(feats['suf_gen'])] if 'suf_gen' in feats else None
    number = morph.number_features[''.join(feats['suf_num'])] if 'suf_num' in feats else None
    person = morph.person_features[''.join(feats['suf_per'])] if 'suf_per' in feats else None
    return morph.Features(gender, number, person)


def _find_next_empy_line(lines, start_idx):
    while lines[start_idx]:
        start_idx += 1
    return start_idx


class Treebank:

    def __init__(self, lexicon: lex.Lexicon, file_paths: dict, max_num: int = 0):
        original_sentences = {}
        uninfused_sentences = {}
        infused_sentences = {}
        for data_set_name in ['train', 'dev', 'test']:
            tokens_file = file_paths['{}-hebtb.tokens'.format(data_set_name)]
            md_file = file_paths['{}-hebtb-gold.lattices'.format(data_set_name)]
            print("Treebank: loading {} ...".format(data_set_name))
            original_sentences[data_set_name] = self._load_sentences(lexicon, tokens_file, md_file, max_num)
            print("Treebank: {} sentences: {}".format(data_set_name, len(original_sentences[data_set_name])))
            uninfused_sentences[data_set_name] = _normalize(original_sentences[data_set_name])
            infused_sentences[data_set_name] = _infuse(data_set_name, uninfused_sentences[data_set_name])
        self.train_sentences = original_sentences['train']
        self.dev_sentences = original_sentences['dev']
        self.test_sentences = original_sentences['test']
        self.uninfused_train_sentences = uninfused_sentences['train']
        self.uninfused_dev_sentences = uninfused_sentences['dev']
        self.uninfused_test_sentences = uninfused_sentences['test']
        self.infused_train_sentences = infused_sentences['train']
        self.infused_dev_sentences = infused_sentences['dev']
        self.infused_test_sentences = infused_sentences['test']

    def save(self, path: Path):
        print(f"Saving treebank to {path}")
        with open(str(path), 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        print(f"Loading treebank from {path}")
        with open(str(path), 'rb') as f:
            return pickle.load(f)

    def _load_sentences(self, lexicon: lex.Lexicon, tokens_file: str, md_file: str, max_num) -> list:
        tokens_lines = Path(tokens_file).read_text(encoding='utf-8').splitlines()
        md_lines = Path(md_file).read_text(encoding='utf-8').splitlines()
        sentences = []
        tokens_slice_from_idx = 0
        md_slice_from_idx = 0
        tokens_line_size = len(tokens_lines)
        while tokens_slice_from_idx < tokens_line_size:
            tokens_slice_to_idx = _find_next_empy_line(tokens_lines, tokens_slice_from_idx)
            lines = tokens_lines[tokens_slice_from_idx:tokens_slice_to_idx]
            tokens = conllx.read_tokens(lines)
            tokens_slice_from_idx = tokens_slice_to_idx + 1

            md_slice_to_idx = _find_next_empy_line(md_lines, md_slice_from_idx)
            lines = md_lines[md_slice_from_idx:md_slice_to_idx]
            md_lattice = conllx.read_lattice(lines)
            md_slice_from_idx = md_slice_to_idx + 1

            if len(tokens) != len(md_lattice._token_paths):
                raise RuntimeError("misaligned tokens and md lattice: {}, {}".format(tokens, md_lattice))
            sent = self._create_sentence(lexicon, tokens, md_lattice)
            sentences.append(sent)
            if max_num > 0 and len(sentences) == max_num:
                break
        return sentences

    def _create_sentence(self, lexicon: lex.Lexicon, tokens: list, md_lattice: conllx.LatticeGraph) -> nlp.Sentence:
        lex_entries = [lexicon.entry(token) for token in tokens]
        lattice = morph.Lattice()
        for i, entry in enumerate(lex_entries):
            lattice[i + 1] = entry.analyses
        gold_lattice = morph.Lattice()
        for tid, token in enumerate(tokens):
            token_id = tid + 1
            gold_analysis = []
            for path in md_lattice._token_paths[token_id]:
                gold_analysis.append(self._create_analysis(path))
            if len(gold_analysis) != 1:
                raise ValueError("token gold analysis: {}".format(gold_analysis))
            gold_lattice[token_id] = gold_analysis
        return nlp.Sentence(tokens, lattice, gold_lattice)

    def _create_analysis(self, path: list) -> morph.Analysis:
        prefixes = []
        hosts = []
        suffixes = []
        for edge in path:
            p = self._create_pref_morpheme(edge)
            if p is not None:
                if hosts:
                    hosts.append(p)
                else:
                    prefixes.append(p)
                continue
            h = self._create_host_morpheme(edge)
            if h is not None:
                hosts.append(h)
            s = self._create_suff_morpheme(edge)
            if s is not None:
                suffixes.append(s)
        return morph.Analysis(prefixes, hosts, suffixes)

    def _create_pref_morpheme(self, edge: conllx.LatticeEdge) -> morph.Morpheme:
        return morph.prefix(edge.form, edge.lemma, edge.postag)

    def _create_host_morpheme(self, edge: conllx.LatticeEdge) -> morph.Morpheme:
        # if edge.cpostag not in ['S_PRN', 'S_PP', 'S_PRP', 'S_ANP']:
        if edge.cpostag not in ['S_PRN', 'S_PP', 'S_ANP']:
            features = _create_features(edge.features)
            return morph.host(edge.form, edge.lemma, edge.cpostag, features)
        return None

    def _create_suff_morpheme(self, edge: conllx.LatticeEdge) -> morph.Morpheme:
        # In SPMRL: NN+S_PP is represented as a single edge with host and suffix combined: bclm/NN_S_PP
        # In UD: this is no longer the case and NN_S_PP is split into 3: bcl/NN + fl/POS + hm/S_PP
        # if edge.cpostag in ['S_PRN', 'S_PP', 'S_PRP', 'S_ANP']:
        if edge.cpostag in ['S_PRN', 'S_PP', 'S_ANP']:
            features = _create_features(edge.features)
            return morph.suffix(edge.postag, features)
        elif edge.postag in ['NN_S_PP']:
            features = _create_suff_features(edge.features)
            return morph.suffix(edge.postag[-4:], features)
        # elif edge.postag in ['S_PRN', 'S_PP', 'S_PRP', 'S_ANP']:
        elif edge.postag in ['S_PRN', 'S_PP', 'S_ANP']:
            features = _create_suff_features(edge.features)
            return morph.suffix(edge.postag, features)
        return None


def _norm_analysis_equals(analysis: morph.Analysis, gold: morph.Analysis, morph_features: dict, morph_equals_func) -> bool:
    if analysis.hosts:
        afeats = [f for f in analysis.hosts[-1].feats if analysis.hosts[-1].feats[f] is not None]
    else:
        afeats = []
    if gold.hosts:
        gfeats = [f for f in gold.hosts[-1].feats if gold.hosts[-1].feats[f] is not None]
    else:
        gfeats = []
    mfeats = [morph_features[f] for f in afeats if f in gfeats and f in morph_features]
    if morph_equals_func(analysis, gold, mfeats):
        return True
    if morph.Gender in mfeats:
        if (analysis.hosts[-1].feats.gender == morph.Gender.MALE_FEMALE or
                gold.hosts[-1].feats.gender == morph.Gender.MALE_FEMALE):
            mfeats = [f for f in mfeats if f != morph.Gender]
            return morph_equals_func(analysis, gold, mfeats)
    return False


def _map_normalized_gold_analysis(lattice: morph.Lattice, gold: morph.Analysis) -> list:
    morph_features = {'gen': morph.Gender, 'num': morph.Number, 'per': morph.Person, 'tense': morph.Tense,
                      'binyan': morph.Binyan, 'pol': morph.Polarity}
    normalized_gold = [a for a in lattice if _norm_analysis_equals(a, gold, morph_features, morph.analysis_equals)]
    if len(normalized_gold) == 0:
        normalized_gold = [a for a in lattice if _norm_analysis_equals(a, gold, morph_features, morph.analysis_equals_no_lemma)]
    if len(normalized_gold) == 0:
        normalized_gold.append(gold)
    return normalized_gold


def _get_norm_analysis_features(normalized_gold: list, gold: morph.Analysis) -> morph.Features:
    genders = list({f.gender for a in normalized_gold for f in a.host_feats})
    numbers = list({f.number for a in normalized_gold for f in a.host_feats})
    persons = list({f.person for a in normalized_gold for f in a.host_feats})
    tenses = list({f.tense for a in normalized_gold for f in a.host_feats})
    binyans = list({f.binyan for a in normalized_gold for f in a.host_feats})
    polarities = list({f.polarity for a in normalized_gold for f in a.host_feats})
    gold_gender = list({f.gender for f in gold.host_feats})[0]
    gold_number = list({f.number for f in gold.host_feats})[0]
    gold_person = list({f.person for f in gold.host_feats})[0]
    gold_tense = list({f.tense for f in gold.host_feats})[0]
    gold_binyan = list({f.binyan for f in gold.host_feats})[0]
    gold_polarity = list({f.polarity for f in gold.host_feats})[0]
    if len(genders) == 1:
        gender = genders[0]
    elif gold_gender in genders:
        gender = gold_gender
    else:
        gender = None
    if len(numbers) == 1:
        number = numbers[0]
    elif gold_number in numbers:
        number = gold_number
    else:
        number = None
    if len(persons) == 1:
        person = persons[0]
    elif gold_person in persons:
        person = gold_person
    else:
        person = None
    if len(tenses) == 1:
        tense = tenses[0]
    elif gold_tense in tenses:
        tense = gold_tense
    else:
        tense = None
    if len(binyans) == 1:
        binyan = binyans[0]
    elif gold_binyan in binyans:
        binyan = gold_binyan
    else:
        binyan = None
    if len(polarities) == 1:
        polarity = polarities[0]
    elif gold_polarity in polarities:
        polarity = gold_polarity
    else:
        polarity = None
    return morph.Features(gender, number, person, tense, binyan, polarity)


def _reduce_normalized_gold_analysis(gold: morph.Analysis, normalized_gold: list):
    if len(normalized_gold) == 1:
        return normalized_gold[0]
    f = _get_norm_analysis_features(normalized_gold, gold)
    if len(f) < len(gold.hosts[0].feats):
        return gold
    forms = list({form for a in normalized_gold for form in a.host_forms})
    lemmas = list({lemma for a in normalized_gold for lemma in a.host_lemmas})
    tags = list({tag for a in normalized_gold for tag in a.host_tags})
    form = forms[0]
    if len(lemmas) == 1:
        lemma = lemmas[0]
    elif len(lemmas) == 0:
        lemma = None
    else:
        lemma = gold.host_lemmas[-1]
    tag = tags[0]
    m = morph.Morpheme(form, lemma, tag, f)
    return morph.Analysis(gold.prefixes, [m], gold.suffixes)


def _normalize(sentences: list) -> list:
    normalized_sentences = []
    for sent in sentences:
        normalized_gold_lattice = morph.Lattice()
        for token_index in sent.lattice:
            token_lattice = sent.lattice[token_index]
            gold = sent.gold_lattice[token_index][0]
            normalized_gold = _map_normalized_gold_analysis(token_lattice, gold)
            normalized_gold = _reduce_normalized_gold_analysis(gold, normalized_gold)
            normalized_gold_lattice[token_index] = [normalized_gold]
        normalized_sent = nlp.Sentence(sent.tokens, sent.lattice, normalized_gold_lattice)
        normalized_sentences.append(normalized_sent)
    return normalized_sentences


def _infuse(data_set_name: str, sentences: list) -> list:
    infused_sentences = []
    total_infused_token_lattices = 0
    total_infused_sentence_lattices = 0
    for sent_index, sent in enumerate(sentences):
        sentence_infused = False
        infused_lattice = morph.Lattice()
        for token_index in sent.gold_lattice:
            infused_lattice[token_index] = sent.lattice[token_index].copy()
            gold_analysis = sent.analysis(token_index)
            found = False
            for analysis in sent.lattice[token_index]:
                if morph.analysis_equals_no_lemma(gold_analysis, analysis, []):
                    found = True
                    break
            if not found:
                sentence_infused = True
                total_infused_token_lattices += 1
                print('Infusing {} sent_index {} token_index {}: {}'.format(data_set_name, sent_index, token_index, gold_analysis))
                infused_lattice[token_index].append(gold_analysis)
        if sentence_infused:
            total_infused_sentence_lattices += 1
        infused_sent = nlp.Sentence(sent.tokens, infused_lattice, sent.gold_lattice)
        infused_sentences.append(infused_sent)
    print("Total {} infused token lattices = {}".format(data_set_name, total_infused_token_lattices))
    print("Total {} infused sentence lattices = {}".format(data_set_name, total_infused_sentence_lattices))
    return infused_sentences


def normalize_feats(feats: morph.Features, is_suffix=False) -> str:
    norm_feats = []
    for f in str(feats).split('|'):
        norm_feats.append(f)
    if is_suffix:
        return '|'.join('suf_' + f for f in norm_feats)
    return '|'.join(norm_feats)


lattice_line = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'


def format_gold_lattice(sentence: nlp.Sentence) -> list:
    lines = []
    from_node_id = 0
    to_node_id = 1
    for i in range(len(sentence.tokens)):
        token_id = i + 1
        ga = sentence.analysis(token_id)
        for p in ga.prefixes:
            pref_line = lattice_line.format(from_node_id, to_node_id, p.form, p.lemma, p.tag, p.tag, normalize_feats(p.feats), token_id)
            lines.append(pref_line)
            from_node_id = to_node_id
            to_node_id += 1
        for h in ga.hosts[:-1]:
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        if not ga.suffixes and ga.hosts:
            h = ga.hosts[-1]
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        elif not ga.hosts:
            for s in ga.suffixes:
                suff_line = lattice_line.format(from_node_id, to_node_id, s.form, s.lemma, s.tag, s.tag, normalize_feats(s.feats), token_id)
                lines.append(suff_line)
                from_node_id = to_node_id
                to_node_id += 1
        else:
            h = ga.hosts[-1]
            s = ga.suffixes[-1]
            if (h.tag == 'PRP' or h.tag == 'DT' or h.tag == 'EX' or h.tag == 'RB') and s.tag == 'S_PRN':
                host_suff_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, s.tag, normalize_feats(s.feats, True), token_id)
                lines.append(host_suff_line)
                from_node_id = to_node_id
                to_node_id += 1
            elif h.tag == 'NN' and s.tag == 'S_PP':
                host_suff_tag = '{}_{}'.format(h.tag, s.tag)
                host_feats = normalize_feats(h.feats)
                suff_feats = normalize_feats(s.feats, True)
                host_suff_feats = '{}|{}'.format(host_feats, suff_feats)
                host_suff_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, host_suff_tag, host_suff_feats, token_id)
                lines.append(host_suff_line)
                from_node_id = to_node_id
                to_node_id += 1
            else:
                host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
                lines.append(host_line)
                from_node_id = to_node_id
                to_node_id += 1
                suff_line = lattice_line.format(from_node_id, to_node_id, s.form, s.lemma, s.tag, s.tag, normalize_feats(s.feats), token_id)
                lines.append(suff_line)
                from_node_id = to_node_id
                to_node_id += 1
    return lines


def format_gold_lattice1(sentence: nlp.Sentence) -> list:
    lines = []
    from_node_id = 0
    to_node_id = 1
    for i, token in enumerate(sentence.tokens):
        token_id = i + 1
        ga = sentence.analysis(token_id)
        for p in ga.prefixes:
            pref_line = lattice_line.format(from_node_id, to_node_id, p.form, p.lemma, p.tag, p.tag, normalize_feats(p.feats), token_id)
            lines.append(pref_line)
            from_node_id = to_node_id
            to_node_id += 1
        for h in ga.hosts[:-1]:
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        if not ga.suffixes and ga.hosts:
            h = ga.hosts[-1]
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        elif not ga.hosts:
            for s in ga.suffixes:
                suff_line = lattice_line.format(from_node_id, to_node_id, s.form, s.lemma, s.tag, s.tag, normalize_feats(s.feats), token_id)
                lines.append(suff_line)
                from_node_id = to_node_id
                to_node_id += 1
        else:
            h = ga.hosts[-1]
            s = ga.suffixes[-1]
            host_feats = normalize_feats(h.feats)
            suff_feats = normalize_feats(s.feats, True)
            if host_feats == '_':
                host_suff_feats = '{}'.format(suff_feats)
            else:
                host_suff_feats = '{}|{}'.format(host_feats, suff_feats)
            host_suff_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, s.tag, host_suff_feats, token_id)
            lines.append(host_suff_line)
            from_node_id = to_node_id
            to_node_id += 1
    return lines


def format_gold_lattice2(sentence: nlp.Sentence) -> list:
    lines = []
    from_node_id = 0
    to_node_id = 1
    for i in range(len(sentence.tokens)):
        token_id = i + 1
        ga = sentence.analysis(token_id)
        for p in ga.prefixes:
            pref_line = lattice_line.format(from_node_id, to_node_id, p.form, p.lemma, p.tag, p.tag, normalize_feats(p.feats), token_id)
            lines.append(pref_line)
            from_node_id = to_node_id
            to_node_id += 1
        for h in ga.hosts:
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        for s in ga.suffixes:
            suff_line = lattice_line.format(from_node_id, to_node_id, s.form, s.lemma, s.tag, s.tag, normalize_feats(s.feats), token_id)
            lines.append(suff_line)
            from_node_id = to_node_id
            to_node_id += 1
    return lines


def format_gold_lattice3(sentence: nlp.Sentence) -> list:
    lines = []
    from_node_id = 0
    to_node_id = 1
    for i in range(len(sentence.tokens)):
        token_id = i + 1
        ga = sentence.analysis(token_id)
        for p in ga.prefixes:
            pref_line = lattice_line.format(from_node_id, to_node_id, p.form, p.lemma, p.tag, p.tag, normalize_feats(p.feats), token_id)
            lines.append(pref_line)
            from_node_id = to_node_id
            to_node_id += 1
        for h in ga.hosts[:-1]:
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        if not ga.suffixes and ga.hosts:
            h = ga.hosts[-1]
            host_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, h.tag, normalize_feats(h.feats), token_id)
            lines.append(host_line)
            from_node_id = to_node_id
            to_node_id += 1
        elif not ga.hosts:
            for s in ga.suffixes:
                suff_line = lattice_line.format(from_node_id, to_node_id, s.form, s.lemma, s.tag, s.tag, normalize_feats(s.feats), token_id)
                lines.append(suff_line)
                from_node_id = to_node_id
                to_node_id += 1
        else:
            h = ga.hosts[-1]
            s = ga.suffixes[-1]
            host_feats = normalize_feats(h.feats)
            suff_feats = normalize_feats(s.feats, True)
            if host_feats == '_':
                host_suff_feats = '{}'.format(suff_feats)
            else:
                host_suff_feats = '{}|{}'.format(host_feats, suff_feats)
            host_suff_line = lattice_line.format(from_node_id, to_node_id, h.form, h.lemma, h.tag, s.tag, host_suff_feats, token_id)
            lines.append(host_suff_line)
            from_node_id = to_node_id
            to_node_id += 1
    return lines
