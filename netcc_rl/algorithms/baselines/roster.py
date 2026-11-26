TRANSFORMERS=('mat','jointppo')
BASELINES=('mat','jointppo','mappo','happo','ppo','sac','td3')
def require_active(a,training=False):
    if a not in BASELINES:raise ValueError(a)
