# modal runner for pa2, one runner for every task
#
# the repo code is baked into the image, and the big things that must survive between runs (course assets, the hugging
# face model cache, trained adapters, results) live on the same 'atml' volume pa0 and pa1 used, under /pa2
#
# smoke test, a handful of rows through every task 1 script on a cheap gpu, to catch bugs before the real run:
#     modal run run_modal.py --task task1 --smoke
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

GPU = 'A100-40GB' #a single fixed gpu type for the real runs, so vram and wall-clock numbers are comparable and reportable
SMOKE_GPU = 'L4' #the smoke test only checks that everything runs, so the cheapest card with enough memory (24gb)
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
TASK_NOTEBOOKS = {'task1': 'task1_dpo/task1.ipynb'}
TASK_CONFIGS = {'task1': 'configs/dpo.yaml'}

#smoke sizes -> real models, real sequence lengths and batch sizes, just very few rows
SMOKE_ROWS = {'dpo_standard_train': 32, 'dpo_standard_eval': 8, 'dpo_length_train': 32, 'word_limit_prompts': 2}
SMOKE_OVERRIDES = {'short_ablation_examples': 16, 'word_limit_samples': 2}


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


def make_smoke_config(config):
    #writes a copy of the task config that points at the first few rows of each data file and at separate
    #outputs_smoke/results_smoke directories, so a smoke test can never be mistaken for, or overwrite, a real result
    import sys
    import yaml
    sys.path.insert(0, REMOTE_REPO)
    from common.data import load_yaml, read_jsonl, write_jsonl

    cfg = load_yaml(config)
    smoke_dir = Path(REMOTE_STORE) / 'smoke'
    for key, rows in SMOKE_ROWS.items():
        path = smoke_dir / 'data' / f'{key}.jsonl'
        write_jsonl(path, read_jsonl(cfg['paths'][key])[:rows])
        cfg['paths'][key] = str(path)
    #three pairs from each length stratum, so the per-stratum table has every row
    stratified = read_jsonl(cfg['paths']['dpo_length_eval'])
    picked = [r for s in ['preferred_longer', 'length_matched', 'rejected_longer']
              for r in [x for x in stratified if x['length_stratum'] == s][:3]]
    write_jsonl(smoke_dir / 'data' / 'dpo_length_eval.jsonl', picked)
    cfg['paths']['dpo_length_eval'] = str(smoke_dir / 'data' / 'dpo_length_eval.jsonl')

    cfg.update(SMOKE_OVERRIDES)
    cfg['standard_output'] = str(smoke_dir / 'outputs' / 'task1_dpo' / 'standard')
    cfg['length_output'] = str(smoke_dir / 'outputs' / 'task1_dpo' / 'length_balanced')
    cfg['results_dir'] = str(smoke_dir / cfg['results_dir'])
    path = smoke_dir / Path(config).name
    path.write_text(yaml.safe_dump(cfg))
    return str(path), str(smoke_dir / 'outputs')


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
        config, outputs = make_smoke_config(config)
        executed = f'{REMOTE_STORE}/smoke/results/{task}_executed.ipynb'
        #a smoke test always starts clean, otherwise the skip-if-adapter-exists checks would hide the training code
        shutil.rmtree(outputs, ignore_errors=True)
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


@app.local_entrypoint()
def main(task: str = 'task1', smoke: bool = False):
    #the smoke test swaps in the cheap card, the real run keeps the fixed GPU constant
    function = run.with_options(gpu=SMOKE_GPU) if smoke else run
    function.remote(task, smoke)
