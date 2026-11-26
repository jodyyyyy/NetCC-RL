"""Only the three approved adaptive graph presets; resolution never trains."""
from pathlib import Path
from netcc_rl.algorithms.proposed.config import load_config
from netcc_rl.environments import make_env

ROOT = Path(__file__).resolve().parents[1] / "proposed"
NAMES = ('kuramoto', 'sis', 'swing')


def resolved(name, device='cuda:0'):
    if name not in NAMES:
        raise ValueError('formal adaptive graph environment must be kuramoto/sis/swing')
    overrides = [f'device={device}']
    if str(device) == 'cpu':
        overrides += ['collector_backend=cpu_serial', 'collector_batch_size=1',
                      'collector_protocol=null', 'learner_cuda_graphs=false', 'graph_backend=pyg']
    cfg = load_config(ROOT/'configs/defaults'/f'{name}.yaml', overrides)
    make_env(cfg).close()
    return cfg
