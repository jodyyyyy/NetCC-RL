"""Train the seven serial comparison baselines using their native update rules."""
import time, json, traceback
from pathlib import Path
import numpy as np
import torch
from .environment import CASE,CFG,SNAPSHOT,DRIVERS,RAW_DIM,NODE_COUNT,make_env,LocalGraph
from .phase_features import phase_features,FEATURE_MULTIPLIER
from .runtime import rng, restore_rng, seed_everything
from netcc_rl.comparison_output import append_json, checkpoint_index, evaluation as evaluate_rows, progress_record, runtime_record, training_row, write_json
from netcc_rl.algorithms.proposed.common.adaptive_architecture import digest

def validate_runner(runner, config):
    if CASE != config["environment"] or list(DRIVERS) != config["drivers"] or digest(SNAPSHOT) != config["snapshot_sha256"]:
        raise ValueError("comparison environment differs from frozen manifest")
    if digest(runner.profile) != digest(config["profile"]):
        raise ValueError("comparison learner profile differs from frozen manifest")

def load_runner(config, checkpoint, device, seed):
    from pathlib import Path
    if json.loads((Path(checkpoint).parent / "config.json").read_text()) != config:
        raise ValueError("checkpoint configuration identity differs")
    runner = Runner(config["algorithm"], device, seed, config["steps"])
    try:
        validate_runner(runner, config)
        if runner.algorithm in ("ppo", "sac", "td3"):
            runner.agent = type(runner.agent).load(checkpoint, device=device)
        else:
            state = torch.load(checkpoint, map_location=device, weights_only=False)
            if (state["algorithm"], state["case"]) != (runner.algorithm, CASE):
                raise ValueError("checkpoint algorithm or environment differs")
            if runner.algorithm in ("mat", "jointppo"):
                runner.agent.restore(state["training"])
            elif runner.algorithm == "happo":
                runner.agent.restore_training_state(state["training"])
            else:
                for policy, weights in zip(runner.policies, state["training"], strict=True):
                    policy.actor.load_state_dict(weights["actor"])
                    policy.critic.load_state_dict(weights["critic"])
        return runner
    except BaseException:
        runner.env.close()
        raise


class Runner:
    def __init__(self, algorithm, device, seed, budget):
        self.algorithm=algorithm;self.env=make_env();self.n=0;self.episodes=0;self.pending=[]
        self.profile={};self.device=device;self.seed=seed;self.budget=budget
        if algorithm in ('mat','jointppo'):
            from .joint_train import Learner
            self.agent=Learner(algorithm,device=self.device);self.profile=self.agent.profile
            self.predict=self.agent.policy.predict
        elif algorithm=='happo':
            from .happo import HAPPO
            from .fidelity_config import load_profile
            adjacency=SNAPSHOT.get('adjacency') or SNAPSHOT['fixed_actuators']['physical_system']['adjacency']
            self.agent=HAPPO(load_profile('happo'),self.device,adjacency,DRIVERS);self.profile=self.agent.config
            self.predict=self.agent.predict
        elif algorithm=='mappo':
            from .onpolicy import build
            self.args,self.graph,self.policies,self.trainers,self.buffers=build(algorithm,self.device,formal=True)
            self.profile=vars(self.args);self.predict=lambda o:self.map_action(o,True)[1][:,0,0]
        else:
            from .sb3_settings import build
            self.agent=build(algorithm,self.env,self.seed,self.device)
            self.profile=json.loads((Path(__file__).with_name('configs')/'formal-hyperparameters-v1.json').read_text())['algorithms'][algorithm]
            self.predict=lambda o:self.agent.predict(o,deterministic=True)[0]

    @torch.no_grad()
    def map_action(self,obs,deterministic=False):
        from .batched import onpolicy_parameters,normal_sample
        from torch.distributions import Normal
        local=torch.as_tensor(self.graph.observe(obs)[:,None,:],device=self.device)
        feat=phase_features(obs)
        central=torch.as_tensor(np.broadcast_to(feat,(len(DRIVERS),1,len(feat))).copy(),device=self.device)
        mean,std,value=onpolicy_parameters(self.policies,local,central)
        actions=mean if deterministic else torch.stack([normal_sample(mean[i],std[i]) for i in range(len(DRIVERS))])
        return tuple(torch.stack([value,actions,Normal(mean,std).log_prob(actions)]).cpu().numpy())

    def episode(self,limit):
        state,_=self.env.reset(seed=self.seed+self.episodes);ret=0.;data=[]
        if self.algorithm=='mappo':
            from .vendor.onpolicy.utils.separated_buffer import SeparatedReplayBuffer
            self.args.episode_length=limit
            self.buffers=[SeparatedReplayBuffer(self.args,p.obs_space,p.share_obs_space,p.act_space) for p in self.policies]
            for i,(b,tr) in enumerate(zip(self.buffers,self.trainers)):
                b.obs[0]=self.graph.observe(state)[i];b.share_obs[0]=phase_features(state);b.masks.fill(1);tr.prep_rollout()
        for t in range(limit):
            with torch.no_grad():
                if self.algorithm in ('mat','jointppo'):
                    raw=torch.as_tensor(state,device=self.device)[None];act,latent,lp,value=self.agent.policy.act(raw);action=act[0].cpu().numpy()
                elif self.algorithm=='happo':
                    act,lp=self.agent.sample(state);action=act.cpu().numpy()
                else:values,act,lp=self.map_action(state);action=act[:,0,0]
            nxt,r,term,trunc,info=self.env.step(action);ret+=r;self.n+=1
            if self.algorithm in ('mat','jointppo'):
                with torch.no_grad():
                    memory=self.agent.policy.encode(torch.as_tensor(nxt,device=self.device)[None]);nv=self.agent.policy.components(memory)[1]
                data.append(dict(states=raw[0],latent=latent[0],logp=lp[0],values=value[0],next_values=nv[0],rewards=torch.tensor(r),terminated=torch.tensor(term),truncated=torch.tensor(trunc)))
            elif self.algorithm=='happo':
                data.append(dict(states=torch.tensor(state),actions=act,logp=lp,rewards=torch.tensor(r),next_states=torch.tensor(nxt),terminated=torch.tensor(term),truncated=torch.tensor(trunc)))
            else:
                for i,b in enumerate(self.buffers):
                    z=np.zeros((1,self.args.recurrent_N,self.args.hidden_size),np.float32)
                    b.insert(phase_features(nxt)[None],self.graph.observe(nxt)[i:i+1],z,z,act[i],lp[i],values[i],np.array([[r]],np.float32),np.array([[0. if term else 1.]],np.float32))
            state=nxt
            if hasattr(self,'heartbeat') and self.n%25==0:self.heartbeat(self.n,'collecting')
            if term or trunc:break
        if hasattr(self,'heartbeat'):self.heartbeat(self.n,'updating')
        if self.algorithm in ('mat','jointppo'):
            metrics=self.agent.update({k:[x[k].detach().cpu() for x in data] for k in data[0]})
        elif self.algorithm=='happo':
            from .trajectories import trajectory
            self.pending.append(trajectory(**{k:torch.stack([x[k].detach().cpu() for x in data]) for k in data[0]}))
            metrics={}
            if len(self.pending)==self.profile['batch_episode'] or self.n==self.budget:
                updates=self.agent.update(self.pending);self.pending=[]
                metrics=dict(sequential_minibatches=len(updates),last_value_loss=updates[-1]['value_loss'])
        else:
            # Native environments truncate at 150; final partial rollout bootstraps.
            for p,b,tr in zip(self.policies,self.buffers,self.trainers):
                with torch.no_grad():tail=p.get_values(b.share_obs[t+1],b.rnn_states_critic[t+1],b.masks[t+1]).cpu().numpy()
                if term:tail*=0
                if t+1!=limit:raise AssertionError('unexpected early boundary requires resized MAPPO buffer')
                b.compute_returns(tail,tr.value_normalizer);tr.prep_training()
            from .onpolicy_fast import train_independent
            stats=train_independent(self.trainers,self.buffers)
            metrics={k:float(np.mean([s[k] for s in stats])) for k in stats[0]}
        modules=[self.agent.policy] if self.algorithm in ('mat','jointppo') else ([self.agent] if self.algorithm=='happo' else [m for p in self.policies for m in (p.actor,p.critic)])
        if not all(torch.isfinite(p).all() for m in modules for p in m.parameters()):raise FloatingPointError('nonfinite learner parameters')
        self.episodes+=1
        return dict(env_steps=self.n,agent_steps=self.n*len(DRIVERS),episode=self.episodes,episode_return=ret,episode_steps=t+1,terminated=term,truncated=trunc,collector_cutoff=not(term or trunc),metrics=metrics)

    def state(self):
        if self.algorithm=='mappo':
            training=[dict(actor=p.actor.state_dict(),critic=p.critic.state_dict(),actor_optimizer=p.actor_optimizer.state_dict(),critic_optimizer=p.critic_optimizer.state_dict()) for p in self.policies]
        else:training=self.agent.training_state()
        return dict(algorithm=self.algorithm,case=CASE,config=CFG,profile=self.profile,env_steps=self.n,episodes=self.episodes,training=training,pending=self.pending,rng=rng())


def checkpoint(r,out,name):
    if r.algorithm in ('ppo','sac','td3'):
        r.agent.save(out/(name+'.zip'))
        if r.algorithm!='ppo':r.agent.save_replay_buffer(out/'replay.pkl')
        torch.save(dict(rng=rng(),env_steps=r.n),out/'rng.pt')
    else:
        tmp=out/(name+'.tmp');torch.save(r.state(),tmp);tmp.replace(out/(name+'.pt'))

def evaluation(r,config,out=None,episodes=50,seed_start=202600):
    saved=rng();env=make_env();rows=[]
    seeds = list(range(seed_start, seed_start + episodes))
    try:
        for seed in seeds:
            o,_=env.reset(seed=seed);ret=0.;steps=0
            while True:
                o,reward,term,trunc,_=env.step(r.predict(o));ret+=reward;steps+=1
                if term or trunc:break
            rows.append({"seed":seed,"return":ret,"transitions":steps,"terminated":bool(term),"truncated":bool(trunc)})
        result = evaluate_rows(config, rows)
        if out is not None:write_json(out, result)
        return result
    finally:env.close();restore_rng(saved)

def train(config, output, device, seed, smoke):
    from types import SimpleNamespace
    a = SimpleNamespace(output=output, algorithm=config["algorithm"], check=smoke)
    budget = config["steps"]
    if smoke:
        budget = 5002 if a.algorithm in ("sac", "td3") else (1500 if a.algorithm == "happo" else 150)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    seed_everything(seed);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started = time.time()
    status = dict(phase="initializing", env_steps=0, learner_updates=None)
    def publish():
        now = time.time()
        value = progress_record(config, status["phase"], status["env_steps"], budget,
                                learner_updates=status.get("learner_updates"),
                                wall_seconds=now-started, updated_at=now, error=status.get("error"))
        write_json(out/"progress.json", value)
    publish()
    try:
        r=Runner(a.algorithm, device, seed, budget)
        validate_runner(r, config)
        def heartbeat(n,phase):
            status.update(env_steps=n, phase=phase);publish()
        r.heartbeat=heartbeat
        write_json(out / 'config.json', config)
        write_json(out / 'runtime.json', runtime_record(config,seed,device,str(torch.__version__),budget,smoke))
        next_save=100000;status['phase']='collecting';publish()
        with (out/'metrics.jsonl').open('w',buffering=1) as log:
            def progress(row):
                episodes = row.get('episodes', [])
                if 'episode_return' in row:
                    episodes = [{'return':row['episode_return'], 'transitions':row['episode_steps'],
                                 'terminated':row['terminated'], 'truncated':row['truncated']}]
                if episodes and 'r' in episodes[0]:
                    episodes = [{'return':e['r'], 'transitions':e['l']} for e in episodes]
                normalized = training_row(config, row['env_steps'], row.get('metrics', {}),
                                          learner_updates=row.get('learner_updates'), episodes=episodes)
                log.write(json.dumps(normalized,allow_nan=False)+'\n')
                status.update(row);publish()
            if a.algorithm in ('ppo','sac','td3'):
                from stable_baselines3.common.callbacks import BaseCallback
                class Callback(BaseCallback):
                    def _on_step(self):
                        nonlocal next_save
                        r.n=self.num_timesteps
                        if r.n%100==0 or any('episode' in i for i in self.locals['infos']):
                            row=dict(env_steps=r.n,agent_steps=r.n,learner_updates=int(r.agent._n_updates))
                            episodes=[i['episode'] for i in self.locals['infos'] if 'episode' in i]
                            if episodes:row['episodes']=episodes
                            progress(row)
                        if r.n>=next_save:checkpoint(r,out,f'checkpoint-{r.n:09d}');next_save+=100000
                        return True
                # Finish exactly at budget with a partial final rollout, then update it.
                if a.algorithm=='ppo':
                    from stable_baselines3.common.buffers import RolloutBuffer
                    full=r.agent.n_steps;aligned=budget//full*full
                    if aligned:r.agent.learn(total_timesteps=aligned,callback=Callback())
                    left=budget-aligned
                    if left:
                        r.agent.n_steps=left;r.agent.rollout_buffer=RolloutBuffer(left,r.agent.observation_space,r.agent.action_space,device=r.agent.device,gae_lambda=r.agent.gae_lambda,gamma=r.agent.gamma,n_envs=1)
                        r.agent.learn(total_timesteps=left,reset_num_timesteps=False,callback=Callback())
                else:r.agent.learn(total_timesteps=budget,callback=Callback())
                r.n=int(r.agent.num_timesteps)
                if not all(torch.isfinite(x).all() for x in r.agent.policy.parameters()):raise FloatingPointError('nonfinite SB3 parameters')
                if r.agent._n_updates<1:raise AssertionError('no SB3 learner update')
            else:
                while r.n<budget:
                    row=r.episode(min(150,budget-r.n));progress(row)
                    if r.n>=next_save:checkpoint(r,out,f'checkpoint-{r.n:09d}');next_save+=100000
            if r.n!=budget:raise AssertionError('real transition budget differs')
            checkpoint(r,out,'final')
            status['phase']='evaluating';publish()
            result = evaluation(r,config,out/'evaluation.json',episodes=1 if smoke else 50)
            append_json(out/'evaluation.jsonl', dict(real_steps=r.n, final=True, **result))
            weights = out / ('final.zip' if a.algorithm in ('ppo','sac','td3') else 'final.pt')
            checkpoint_index(config, out, weights, r.n, seed)
            status.update(phase='completed',env_steps=r.n);publish()
    except BaseException:
        status.update(phase='failed',error=traceback.format_exc());publish();raise
    finally:
        if 'r' in locals():r.env.close()
