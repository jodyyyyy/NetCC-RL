"""Build the optional pinned Turbo-GNN extension."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

UPSTREAM='https://github.com/yandex-research/On-Efficient-Scaling-Of-GNNs.git'
COMMIT='e26d564701c64545ae6eef6b44b2fc95dc36a93e'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cuda-home',default='/usr/local/cuda-12.2');p.add_argument('--arch',default='7.5');p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=False);source=out/'upstream';site=out/'site';cache=out/'cache';cache.mkdir()
    subprocess.run(['git','clone','--no-checkout',UPSTREAM,str(source)],check=True)
    subprocess.run(['git','checkout','--detach',COMMIT],cwd=source,check=True)
    patch=Path(__file__).resolve().with_name('turbo-sm75.patch')
    manifest=json.loads((Path(__file__).resolve().parents[1]/'netcc_rl/algorithms/proposed/TURBO_GNN_MANIFEST.json').read_text())
    patch_hash=hashlib.sha256(patch.read_bytes()).hexdigest()
    if manifest['commit']!=COMMIT or manifest['patch_sha256']!=patch_hash:raise ValueError('build manifest does not match source/patch')
    subprocess.run(['git','apply',str(patch)],cwd=source,check=True)
    env=dict(os.environ,CUDA_HOME=a.cuda_home,TORCH_CUDA_ARCH_LIST=a.arch,TURBO_GNN_FORCE_BUILD='TRUE',MAX_JOBS=str(a.jobs),TMPDIR=str(cache),PIP_CACHE_DIR=str(cache))
    with (out/'build.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--target',str(site),'.'],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'build-source.json').write_text(json.dumps(dict(upstream=UPSTREAM,commit=COMMIT,patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),python=sys.executable,arch=a.arch,cuda_home=a.cuda_home,returncode=result.returncode,site=str(site)),indent=2)+'\n')
    if result.returncode:raise RuntimeError(f'see {out}/build.log')
    package=site/'turbo_gnn'
    files=[package/'_functions.py',package/'ops.py',package/'_kernels.py',*package.glob('_C*.so')]
    stamp=dict(commit=COMMIT,patch_sha256=patch_hash,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (package/'netcc_rl_build.json').write_text(json.dumps(stamp,indent=2)+'\n')
    print(f'Prepend {site} to PYTHONPATH to enable graph_backend=turbo')
if __name__=='__main__':main()
