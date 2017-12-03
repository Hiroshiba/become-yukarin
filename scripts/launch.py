"""
launcher for some task that have diff params
"""

import argparse
import copy
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

base_command_default = \
    "screen -d -m -S {project/name}_gpu{train/gpu} ;" + \
    "screen -S {project/name}_gpu{train/gpu} -X stuff 'python3 {python_file_path} {recipe_path} {output}\n'"

parser = argparse.ArgumentParser()
parser.add_argument('output_dir', type=Path)
parser.add_argument('--python_file_path', default='train.py')
parser.add_argument('--recipe_json_path', default='recipe/recipe.json')
parser.add_argument('--base_config_json_path', default='recipe/config.json')
parser.add_argument('--base_command', default=base_command_default)
args = parser.parse_args()

recipe = json.load(open(args.recipe_json_path, encoding='utf-8'))
recipe_each = recipe['each']
recipe_all = recipe['all']
base_config = json.load(open(args.base_config_json_path, encoding='utf-8'))


def put_config_value(config, recipe_key, value):
    key_tree = recipe_key.split('/')
    target = config
    for key in key_tree[:-1]:
        target = target[key]

    target[key_tree[-1]] = value


def _replace_name(dist):
    _format = {}
    now = datetime.datetime.now()

    if '{date}' in dist['project']['name']:
        _format['date'] = now.strftime('%Y%m%d%H%M%S')
    if '{hash}' in dist['project']['name']:
        _format['hash'] = hashlib.md5(bytes(str(now), 'utf')).hexdigest()[:6]

    if len(_format) > 0:
        dist['project']['name'] = dist['project']['name'].format(**_format)


num_task = min(len(list(value)) for value in recipe_each.values())
command_list = []

for i in range(num_task):
    config = copy.deepcopy(base_config)

    for recipe_key in recipe_all.keys():
        put_config_value(config, recipe_key, recipe_all[recipe_key])

    for recipe_key in recipe_each.keys():
        put_config_value(config, recipe_key, recipe_each[recipe_key][i])

    _replace_name(config)

    # add git branch name
    git_branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode("utf-8").strip()
    config['project']['tags'].append('git branch name:' + git_branch)

    made_recipe_path = "{}.{}.json".format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'), i)
    with open(made_recipe_path, 'w', encoding='utf') as f:
        json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)


    def make_key_chain(key_chain, value, dist):
        if not isinstance(value, dict):
            dist['/'.join(key_chain)] = value
        else:
            for key in value.keys():
                make_key_chain(key_chain + [key], value[key], dist)


    dist = {}
    make_key_chain([], config, dist)

    dist['output'] = args.output_dir / config['project']['name']
    dist['python_file_path'] = args.python_file_path
    dist['recipe_path'] = made_recipe_path

    command = args.base_command.format(**dist)
    command_list += [command]

    print(config['project']['name'])

for command in command_list:
    time.sleep(1)
    subprocess.check_output(command, shell=True)
