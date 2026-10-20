# modal runner for pa2, one runner for every task
#
# the repo code is baked into the image, and the big things that must survive between runs (course assets, the hugging
# face model cache, trained adapters, results) live on the same 'atml' volume pa0 and pa1 used, under /pa2
#
# smoke test, a handful of rows through every task 1 script on a cheap gpu, to catch bugs before the real run:
#     modal run run_modal.py --task task1 --smoke        (task1 ... task5)
#
# the real run, detached so a closed terminal or a dropped connection cannot kill it:
#     modal run --detach run_modal.py --task task1
#
# watch a detached run:   modal app logs atml-pa2
# pull the results back:  modal volume get atml /pa2/results ./

import subprocess
from pathlib import Path

import modal

APP_NAME = 'atml-pa2'
VOLUME_NAME = 'atml' #same volume as pa0/pa1, everything here stays under /pa2 so the assignments do not mix
VOLUME_PATH = '/mnt/atml'
REMOTE_STORE = f'{VOLUME_PATH}/pa2' #persistent side, on the volume
REMOTE_REPO = '/root/pa2' #code side, baked into the image

GPU = 'L4' #a single fixed gpu type for the real runs, so vram and wall-clock numbers are comparable and reportable. measured on task 1 -> an a100 sat at ~31% utilisation and 13 of 80 gb, so the cheaper 24gb card is the better fit
SMOKE_GPU = 'L4' #the smoke test only checks that everything runs
CPU_CORES = 4.0
MEMORY_MB = 32768
TIMEOUT_SECONDS = 8 * 60 * 60

#these directories are created on the volume and linked into the repo copy, so the scripts' relative paths
#(data/, checkpoints/, outputs/, results/ ...) transparently read and write the volume
PERSISTENT_DIRS = ['data', 'cached', 'checkpoints', 'manifests', 'outputs', 'results']

LOCAL_REPO = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version='3.12')
    #cuda 12.8 builds, the ones the course release was prepared with (torch 2.11.0+cu128)
    .pip_install('torch', index_url='https://download.pytorch.org/whl/cu128')
    .pip_install(
        'transformers==4.57.1', 'tokenizers==0.22.1', 'trl==0.27.2', 'peft==0.17.1',
        'datasets>=3.0,<5', 'accelerate>=1.0,<2', 'bitsandbytes>=0.45', 'huggingface_hub>=0.27',
        'safetensors>=0.4', 'pyyaml>=6.0', 'pandas>=2.0', 'numpy>=1.26', 'scikit-learn>=1.4',
        'matplotlib>=3.8', 'tqdm>=4.66', 'papermill==2.7.0', 'ipykernel',
    )
    #the model downloads land on the volume instead of the container disk, so they happen once, not once per run
    .env({'HF_HOME': f'{REMOTE_STORE}/hf_cache', 'TOKENIZERS_PARALLELISM': 'false'})
    #copy=True bakes the code into the image, so a detached run does not depend on the laptop staying up.
    #the asset directories are left out, the container gets them from the volume
    .add_local_dir(
        LOCAL_REPO, remote_path=REMOTE_REPO, copy=True,
        ignore=[*PERSISTENT_DIRS, '.git', '.cache', '**/__pycache__', '**/.ipynb_checkpoints', '*_executed.ipynb'],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

#each task is a thin notebook that calls the release's entry points in order and then builds its tables and figures.
#papermill executes it top to bottom, and the executed copy is kept next to the results as the record of the run
TASK_NOTEBOOKS = {'task1': 'task1_dpo/task1.ipynb', 'task2': 'task2_ppo/task2.ipynb', 'task3': 'task3_grpo/task3.ipynb',
                  'task4': 'task4_safety/task4.ipynb', 'task5': 'task5_feedback/task5.ipynb'}
TASK_CONFIGS = {'task1': 'configs/dpo.yaml', 'task2': 'configs/ppo.yaml', 'task3': 'configs/grpo.yaml',
                'task4': 'configs/feedback.yaml', 'task5': 'configs/feedback.yaml'}

#smoke sizes -> real models, real sequence lengths and batch sizes, just very few rows / updates.
#SMOKE_ROWS cuts data files to their first n rows, SMOKE_OVERRIDES replaces config values
SMOKE_ROWS = {
    'task1': {'dpo_standard_train': 32, 'dpo_standard_eval': 8, 'dpo_length_train': 32, 'word_limit_prompts': 2},
    'task2': {}, #the cached rollout study needs the full evaluation pool, eval_max_prompts limits the evaluation instead
    'task3': {},
    'task4': {},
    #two diagnostic problems (five variants each) and a handful of gsm8k / svamp problems
    'task5': {'gsm_eval': 8, 'math_transfer_eval': 8, 'task5_diagnostics': 10},
}
SMOKE_OVERRIDES = {
    'task1': {'short_ablation_examples': 16, 'word_limit_samples': 2},
    'task2': {'updates': 2, 'fork_updates': 1, 'eval_max_prompts': 8},
    'task3': {'updates': 2, 'fork_updates': 2, 'eval_max_prompts': 8},
    'task4': {'manual_audit_per_class': 3},
    'task5': {},
}
#config keys that hold an output location, redirected into the smoke directory
OUTPUT_KEYS = ['standard_output', 'length_output', 'output']


def link_persistent_dirs():
    #repo/data -> volume/pa2/data and so on. the tiny files the repo tracks inside those directories
    #(word_limit_prompts.jsonl) also come with the course asset download, so nothing is lost by linking over them
    for name in PERSISTENT_DIRS:
        target = Path(REMOTE_STORE) / name
        target.mkdir(parents=True, exist_ok=True)
        link = Path(REMOTE_REPO) / name
        if not link.is_symlink():
            if link.exists():
                subprocess.run(['rm', '-rf', str(link)], check=True)
            link.symlink_to(target)


def ensure_assets():
    #the course assets are downloaded once onto the volume, later runs only validate them
    if subprocess.run('python -m scripts.validate_assets', shell=True, cwd=REMOTE_REPO).returncode != 0:
        subprocess.run('python -m scripts.download_assets', shell=True, cwd=REMOTE_REPO, check=True)
        subprocess.run('python -m scripts.validate_assets', shell=True, cwd=REMOTE_REPO, check=True)
        volume.commit()


def make_smoke_config(task, config):
    #writes a copy of the task config that points at the first few rows of each data file and at separate
    #outputs/results directories under /pa2/smoke, so a smoke test can never be mistaken for, or overwrite, a real result
    import sys
    import yaml
    sys.path.insert(0, REMOTE_REPO)
    from common.data import load_yaml, read_jsonl, write_jsonl

    cfg = load_yaml(config)
    smoke_dir = Path(REMOTE_STORE) / 'smoke'
    for key, rows in SMOKE_ROWS[task].items():
        path = smoke_dir / 'data' / f'{key}.jsonl'
        write_jsonl(path, read_jsonl(cfg['paths'][key])[:rows])
        cfg['paths'][key] = str(path)
    if task == 'task1':
        #three pairs from each length stratum, so the per-stratum table has every row
        stratified = read_jsonl(cfg['paths']['dpo_length_eval'])
        picked = [r for s in ['preferred_longer', 'length_matched', 'rejected_longer']
                  for r in [x for x in stratified if x['length_stratum'] == s][:3]]
        write_jsonl(smoke_dir / 'data' / 'dpo_length_eval.jsonl', picked)
        cfg['paths']['dpo_length_eval'] = str(smoke_dir / 'data' / 'dpo_length_eval.jsonl')

    if task == 'task4':
        #six safe and six unsafe xstest prompts, and the adapters the task 1-3 smoke tests left under /pa2/smoke
        import pandas as pd
        xstest = pd.read_csv(Path(REMOTE_REPO) / cfg['paths']['xstest'])
        small = pd.concat([xstest[xstest['benchmark_class'] == cls].head(6) for cls in ['SAFE', 'UNSAFE']])
        (smoke_dir / 'data').mkdir(parents=True, exist_ok=True)
        small.to_csv(smoke_dir / 'data' / 'xstest.csv', index=False)
        cfg['paths']['xstest'] = str(smoke_dir / 'data' / 'xstest.csv')
        for policy in ['dpo', 'ppo', 'grpo']:
            cfg['policies'][policy] = str(smoke_dir / cfg['policies'][policy])

    cfg.update(SMOKE_OVERRIDES[task])
    for key in OUTPUT_KEYS:
        if key in cfg:
            cfg[key] = str(smoke_dir / cfg[key])
    cfg['results_dir'] = str(smoke_dir / cfg['results_dir'])
    path = smoke_dir / Path(config).name
    path.write_text(yaml.safe_dump(cfg))
    #the directories this task writes, wiped before a smoke run so it always starts clean
    task_dir = Path(TASK_NOTEBOOKS[task]).parent.name #task1_dpo, task4_safety ...
    return str(path), [str(smoke_dir / 'outputs' / task_dir), str(smoke_dir / 'results' / task_dir)]


@app.function(gpu=GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def run(task, smoke):
    import shutil

    volume.reload() #picks up assets, cached models and adapters written by earlier runs
    link_persistent_dirs()
    subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', shell=True)
    ensure_assets()

    import logging
    import papermill

    #papermill reports each cell's output through python logging, without this nothing reaches the modal logs
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    config, executed = TASK_CONFIGS[task], f'{REMOTE_STORE}/results/{task}_executed.ipynb'
    if smoke:
        config, stale = make_smoke_config(task, config)
        executed = f'{REMOTE_STORE}/smoke/results/{task}_executed.ipynb'
        #a smoke test always starts clean, otherwise the skip-if-already-done checks would hide the code under test
        for directory in stale:
            shutil.rmtree(directory, ignore_errors=True)
    Path(executed).parent.mkdir(parents=True, exist_ok=True)

    try:
        papermill.execute_notebook(
            f'{REMOTE_REPO}/{TASK_NOTEBOOKS[task]}', executed,
            parameters={'CONFIG': config},
            cwd=REMOTE_REPO, #the scripts resolve configs/, data/, results/ from the repo root
            kernel_name='python3',
            progress_bar=False,
            log_output=True, #every cell's output goes to the modal logs, so a detached run can still be followed
        )
    finally:
        #committed even when a cell raises, so the partially executed notebook and whatever results exist are kept
        volume.commit()
        print(f'results committed under {REMOTE_STORE}')


@app.function(gpu=SMOKE_GPU, cpu=CPU_CORES, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS, volumes={VOLUME_PATH: volume})
def shell(command):
    #runs one arbitrary command inside the same environment, for quick checks that do not need a whole task ->
    #    modal run run_modal.py --cmd "python -m task2_ppo.analyze_clipping --cached-only"
    volume.reload()
    link_persistent_dirs()
    ensure_assets()
    try:
        subprocess.run(command, shell=True, cwd=REMOTE_REPO, check=True)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(task: str = 'task1', smoke: bool = False, cmd: str = ''):
    if cmd:
        shell.remote(cmd)
        return
    #the smoke test swaps in the cheap card, the real run keeps the fixed GPU constant
    function = run.with_options(gpu=SMOKE_GPU) if smoke else run
    function.remote(task, smoke)
