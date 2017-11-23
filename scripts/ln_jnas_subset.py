import argparse
import multiprocessing
from pathlib import Path

from jnas_metadata_loader import load_from_directory
from jnas_metadata_loader.jnas_metadata import JnasMetadata

parser = argparse.ArgumentParser()
parser.add_argument('jnas', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--format', default='{sex}{text_id}_{mic}_atr_{subset}{sen_id}.wav')
argument = parser.parse_args()

jnas = argument.jnas  # type: Path
output = argument.output  # type: Path

jnas_list = load_from_directory(str(jnas))
atr_list = jnas_list.subset_news_or_atr('B')

output.mkdir(exist_ok=True)


def process(d: JnasMetadata):
    p = d.path
    out = output / argument.format.format(**d._asdict())
    out.symlink_to(p)


pool = multiprocessing.Pool()
pool.map(process, atr_list)
