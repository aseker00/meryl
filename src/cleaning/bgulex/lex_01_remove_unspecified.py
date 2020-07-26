from pathlib import Path


def load(lex_file: str):
    lex_file_lines = Path(lex_file).read_text(encoding='utf-8').splitlines()
    entries = []
    for line in lex_file_lines:
        parts = line.split()
        analyses = int(len(parts[1:]) / 2)
        if analyses == 1:
            lemma = parts[2]
            if lemma != 'unspecified':
                entries.append(' '.join(parts))
        else:
            removed_unk_parts = [parts[0]]
            analyses = int(len(parts[1:])/2)
            for i in range(analyses):
                lemma = parts[1 + 2*i + 1]
                if lemma != 'unspecified':
                    removed_unk_parts.append(parts[1 + 2*i])
                    removed_unk_parts.append(parts[1 + 2 * i + 1])
            entries.append(' '.join(removed_unk_parts))
    return entries


lex_lines = load('data/raw/spmrl/bgulex/bgulex.utf8.hr')
with open('data/clean/spmrl/bgulex/bgulex-01.hr', 'w') as f:
    f.write('\n'.join(lex_lines))
    f.write('\n')
