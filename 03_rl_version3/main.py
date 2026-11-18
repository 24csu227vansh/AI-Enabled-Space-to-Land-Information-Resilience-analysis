import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO,DQN
from environment.information_env import InformationEnv
from evaluation.compare import evaluate
from utils.sen12ms_features import load_features

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps',type=int,default=15000)
    parser.add_argument('--episodes',type=int,default=30)
    parser.add_argument('--features',default=None,help='Optional CSV extracted from real SEN12MS TIFF patches')
    args=parser.parse_args()
    if args.timesteps<1000 or args.episodes<2: parser.error('Use at least 1000 timesteps and 2 evaluation episodes.')
    rows=load_features(args.features) if args.features else None
    output=Path('results');output.mkdir(exist_ok=True)
    mode='image-informed coverage + simulated other factors' if rows else 'fully synthetic'
    print('Mode:',mode)
    print('Training PPO...')
    ppo=PPO('MlpPolicy',InformationEnv(condition='mixed',feature_rows=rows),seed=7,
            verbose=0,n_steps=256,batch_size=64,learning_rate=3e-4)
    ppo.learn(total_timesteps=args.timesteps)
    print('Training DQN...')
    dqn=DQN('MlpPolicy',InformationEnv(condition='mixed',feature_rows=rows),seed=7,
            verbose=0,learning_starts=500,buffer_size=20000,batch_size=64,train_freq=4)
    dqn.learn(total_timesteps=args.timesteps)
    print('Evaluating...')
    raw,summary,report=evaluate({'PPO':ppo,'DQN':dqn,'Rule-Based':None},episodes=args.episodes,feature_rows=rows)
    raw.to_csv(output/'episode_results.csv',index=False)
    summary.to_csv(output/'performance_summary.csv',index=False)
    report.to_csv(output/'resilience_index.csv',index=False)
    ppo.save(output/'ppo_model');dqn.save(output/'dqn_model')
    report.pivot(index='condition',columns='method',values='mean').reindex(['normal','mild','moderate','severe']).plot(marker='o')
    plt.ylabel('Fixed-weight performance');plt.title('Information quality vs degradation')
    plt.tight_layout();plt.savefig(output/'performance_comparison.png',dpi=160);plt.close()
    report[report.condition!='normal'].pivot(index='condition',columns='method',values='RI').reindex(['mild','moderate','severe']).plot(marker='o')
    plt.ylabel('Resilience Index');plt.title('Performance retention by degradation')
    plt.tight_layout();plt.savefig(output/'resilience_index.png',dpi=160);plt.close()
    print(report[['method','condition','mean','std','RI']].round(4).to_string(index=False))
    print('Saved to',output.resolve())
    print('DATA LIMITATION: Only image-derived coverage is observed. Other indicators and degradation are simulated.')

if __name__=='__main__':main()
