import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--prefix', default='')
argument = parser.parse_args()

input = argument.input  # type: Path
output = argument.output  # type: Path

paths = list(sorted(input.glob('*'), key=lambda p: int(''.join(filter(str.isdigit, p.name)))))
assert len(paths) == 503

output.mkdir(exist_ok=True)

names = ['{}{:02d}'.format(s, n + 1) for s in 'ABCDEFGHIJ' for n in range(50)]
names += ['J51', 'J52', 'J53']

for p, n in zip(paths, names):
    out = output / (argument.prefix + n + p.suffix)
    out.symlink_to(p)
