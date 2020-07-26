from src.processing import morph
from src.processing.spmrl.lexicon import Lexicon


def lattice(tokens: list, lex: Lexicon) -> morph.Lattice:
    lex_entries = [lex.entry(token) for token in tokens]
    lex_lattice = morph.Lattice()
    for tid, token in enumerate(tokens):
        token_id = tid + 1
        lex_lattice[token_id] = lex_entries[tid].analyses
    return lex_lattice
