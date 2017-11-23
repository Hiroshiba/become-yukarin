"""
ある話者のATR503サブセットを、他の話者に対応するようにコピーする。
targetは、拡張子前3文字がATR503サブセットでないといけない。
"""

import argparse
from pathlib import Path
import re
from itertools import chain, groupby

parser = argparse.ArgumentParser()
parser.add_argument('source', type=Path)
parser.add_argument('target', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--prefix', default='')
argument = parser.parse_args()

source = argument.source  # type: Path
target = argument.target  # type: Path
output = argument.output  # type: Path

# source
sources = list(sorted(source.glob('*')))
assert len(sources) == 503

names = ['{}{:02d}'.format(s, n + 1) for s in 'ABCDEFGHIJ' for n in range(50)]
names += ['J51', 'J52', 'J53']

assert all(n in s.name for s, n in zip(sources, names))

map_source = {n: s for s, n in zip(sources, names)}

# target
keyfunc = lambda t: t.stem[-3:]
targets = list(target.glob('*'))
map_targets = {n: list(vs) for n, vs in groupby(sorted(targets, key=keyfunc), key=keyfunc)}

assert all(n in names for n in map_targets.keys())
assert len(list(chain.from_iterable(map_targets.values()))) == len(targets)

# output
output.mkdir(exist_ok=True)

for n in names:
    s = map_source[n]
    for t in map_targets[n]:
        out = output / (argument.prefix + t.stem + s.suffix)
        out.symlink_to(s)
