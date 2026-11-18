"""Run independent training seeds and matched evaluation scenarios on user's machine.
Execute from SpaceToLand_RL_v3 project root after copying this file there.
"""
import argparse
from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO, DQN
from environment.information_env import InformationEnv
from evaluation.compare import evaluate
from utils.sen12ms_features import load_features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default=None)
    ap.add_argument('--seeds',type=int,default=5)
    ap.add_argument('--timesteps',type=int,default=30000)
    ap.add_argument('--episodes',type=int,default=50)
    args=ap.parse_args()
    if args.seeds<2 or args.episodes<2: ap.error('Use at least two seeds and two episodes')
    rows=load_features(args.features) if args.features else None
    output=Path('results/multi_seed');output.mkdir(parents=True,exist_ok=True)
    all_raw=[];all_reports=[]
    for i in range(args.seeds):
        seed=17+101*i
        print(f'Training seed {i+1}/{args.seeds}: {seed}',flush=True)
        ppo=PPO('MlpPolicy',InformationEnv(condition='mixed',feature_rows=rows),seed=seed,
                verbose=0,n_steps=256,batch_size=64,learning_rate=3e-4)
        ppo.learn(total_timesteps=args.timesteps)
        dqn=DQN('MlpPolicy',InformationEnv(condition='mixed',feature_rows=rows),seed=seed,
                verbose=0,learning_starts=500,buffer_size=20000,batch_size=64,train_freq=4)
        dqn.learn(total_timesteps=args.timesteps)
        raw,summary,report=evaluate({'PPO':ppo,'DQN':dqn,'Rule-Based':None},
                                    episodes=args.episodes,seed=1000,feature_rows=rows)
        raw['training_seed']=seed;report['training_seed']=seed
        all_raw.append(raw);all_reports.append(report)
        raw.to_csv(output/f'episodes_seed_{seed}.csv',index=False)
    episodes=pd.concat(all_raw,ignore_index=True)
    reports=pd.concat(all_reports,ignore_index=True)
    episodes.to_csv(output/'all_episodes.csv',index=False)
    reports.to_csv(output/'all_seed_results.csv',index=False)
    reports.groupby(['method','condition'],as_index=False).agg(
        performance_mean=('mean','mean'),performance_sd_between_seeds=('mean','std'),
        RI_mean=('RI','mean'),RI_sd_between_seeds=('RI','std')
    ).to_csv(output/'across_seed_summary.csv',index=False)
    print('Saved multi-seed results to',output.resolve())
    print('NOTE: With no --features, observations are synthetic. With features, only coverage is image-informed.')
if __name__=='__main__':main()
