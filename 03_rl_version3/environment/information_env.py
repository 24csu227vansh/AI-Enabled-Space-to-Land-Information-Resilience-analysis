import numpy as np
import gymnasium as gym
from gymnasium import spaces

class InformationEnv(gym.Env):
    """Non-operational environmental information-management simulator.

    With image features, observed valid-pixel fractions influence coverage.
    Freshness, reliability and degradation remain synthetic. No combat actions.
    """
    metadata = {'render_modes': []}
    ALLOCATIONS = np.array([[1/3,1/3,1/3], [1,0,0], [0,1,0], [0,0,1]],dtype=np.float32)
    WEIGHTS = np.array([0.4,0.3,0.3],dtype=np.float32)

    def __init__(self, condition='mixed', episode_length=40, seed=0, feature_rows=None):
        super().__init__()
        if condition not in ('mixed','normal','mild','moderate','severe'):
            raise ValueError('Unknown condition')
        self.condition=condition;self.episode_length=episode_length
        self.action_space=spaces.Discrete(4)
        self.observation_space=spaces.Box(0,1,shape=(4,),dtype=np.float32)
        self.rng=np.random.default_rng(seed)
        self.t=0;self.quality=np.ones(3,dtype=np.float32);self.level=0.0
        self.feature_rows=feature_rows

    def _draw(self):
        base=self.rng.uniform(0.70,0.98,size=3)
        if self.feature_rows:
            # Real imagery only informs observed data completeness (coverage).
            s1,s2=self.feature_rows[int(self.rng.integers(len(self.feature_rows)))]
            base[1]=0.5*(s1+s2)
        loss=self.rng.uniform(0.25,1.0,size=3)*self.level
        return np.clip(base-loss,0,1).astype(np.float32)

    def _obs(self):
        return np.r_[self.quality,np.float32(self.level)].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.t=0
        levels={'normal':0.0,'mild':0.15,'moderate':0.35,'severe':0.55}
        self.level=float(self.rng.choice([0.0,0.15,0.35,0.55])) if self.condition=='mixed' else levels[self.condition]
        self.quality=self._draw()
        return self._obs(),{}

    def step(self,action):
        action=int(action)
        allocation=self.ALLOCATIONS[action]
        improved=np.clip(self.quality + 0.18*allocation*(1-self.quality),0,1)
        performance=float(self.WEIGHTS@improved)
        cost=0.008 if action !=0 else 0.0
        reward=performance-cost
        self.t+=1;done=self.t>=self.episode_length
        self.quality=self._draw()
        return self._obs(),float(reward),done,False,{
            'performance':performance,'freshness':float(improved[0]),
            'coverage':float(improved[1]),'reliability':float(improved[2]),
            'processing_cost':cost}
