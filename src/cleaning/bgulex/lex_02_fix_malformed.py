from pathlib import Path


def load(lex_file: str):
    lex_file_lines = Path(lex_file).read_text(encoding='utf-8').splitlines()
    entries = []
    for line in lex_file_lines:
        line_parts = line.split()
        analyses = int(len(line_parts[1:]) / 2)
        # :VB-PAST-PIEL:-MF-S-1
        fixed_analyses = [line_parts[0]]
        for i in range(analyses):
            tags = line_parts[1 + 2 * i]
            lemma = line_parts[1 + 2 * i + 1]
            tag_parts = tags.split(':')
            if tag_parts[2] and tag_parts[2][0] == '-':
                tag1_parts = tag_parts[1].split('-')
                tag1 = tag1_parts[0]
                feats1 = '-'.join(tag1_parts[1:])
                fixed_tag = '{}:{}{}-{}: {}'.format(tag_parts[0], tag1, tag_parts[2], feats1, lemma)
            else:
                fixed_tag = '{} {}'.format(tags, lemma)
            fixed_analyses.append(fixed_tag)
        entries.append(' '.join(fixed_analyses))
    return entries


lex_lines = load('data/clean/spmrl/bgulex/bgulex-01.hr')
with open('data/clean/spmrl/bgulex/bgulex-02.hr', 'w') as f:
    f.write('\n'.join(lex_lines))
    f.write('\n')
