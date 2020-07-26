import patch
from shutil import copyfile
src = 'data/clean/spmrl/bgulex/bgulex-02.hr'
dst = 'data/clean/spmrl/bgulex/bgulex-03.hr'
copyfile(src, dst)

pset = patch.fromfile('src/cleaning/bgulex/03_lex_diff.patch')
pset.apply()
