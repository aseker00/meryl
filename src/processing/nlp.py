from . import morph
# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


class Sentence:

    def __init__(self, tokens: list, lattice: morph.Lattice, gold_lattice: morph.Lattice):
        self.tokens = tokens
        self.lattice = lattice
        self.gold_lattice = gold_lattice

    def __str__(self):
        return str(str(self.tokens) + ': ' + str(self.lattice) + ": " + str(self.gold_lattice))

    def __repr__(self):
        return self.__str__()

    def copy(self):
        return Sentence([t for t in self.tokens], morph.copy_lattice(self.lattice), morph.copy_lattice(self.gold_lattice))

    def analysis(self, token_id: int) -> morph.Analysis:
        return self.gold_lattice[token_id][0]
