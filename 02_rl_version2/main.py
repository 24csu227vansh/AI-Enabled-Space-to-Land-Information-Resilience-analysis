import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from environment.information_env import InformationEnv
from evaluation.compare import evaluate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps',type=int,default=15000)
    parser.add_argument('--episodes',type=int,default=30)
    args=parser.parse_args()
    if args.timesteps < 1000 or args.episodes < 2:
        parser.error('Use at least 1000 timesteps and 2 evaluation episodes.')
    output=Path('results'); output.mkdir(exist_ok=True)
    print('Training PPO on mixed normal/degraded synthetic scenarios...')
    ppo=PPO('MlpPolicy',InformationEnv(condition='mixed'),seed=7,verbose=0,
            n_steps=256,batch_size=64,learning_rate=3e-4)
    ppo.learn(total_timesteps=args.timesteps)
    print('Training DQN on mixed normal/degraded synthetic scenarios...')
    dqn=DQN('MlpPolicy',InformationEnv(condition='mixed'),seed=7,verbose=0,
            learning_starts=500,buffer_size=20000,batch_size=64,train_freq=4)
    dqn.learn(total_timesteps=args.timesteps)
    print('Evaluating on matched held-out scenarios...')
    raw,summary,report=evaluate({'PPO':ppo,'DQN':dqn,'Rule-Based':None},episodes=args.episodes)
    raw.to_csv(output/'episode_results.csv',index=False)
    summary.to_csv(output/'performance_summary.csv',index=False)
    report.to_csv(output/'resilience_index.csv',index=False)
    ppo.save(output/'ppo_model'); dqn.save(output/'dqn_model')
    report.pivot(index='condition',columns='method',values='mean').reindex(['normal','mild','moderate','severe']).plot(marker='o')
    plt.ylabel('Fixed-weight performance'); plt.title('Information quality across degradation levels')
    plt.tight_layout(); plt.savefig(output/'performance_comparison.png',dpi=160); plt.close()
    report[report.condition!='normal'].pivot(index='condition',columns='method',values='RI').reindex(['mild','moderate','severe']).plot(marker='o')
    plt.ylabel('Resilience Index'); plt.title('Performance retention by degradation level')
    plt.tight_layout(); plt.savefig(output/'resilience_index.png',dpi=160); plt.close()
    print(report[['method','condition','mean','std','RI']].round(4).to_string(index=False))
    print('Results saved to',output.resolve())
    print('Synthetic benchmark only: not actual SEN12MS findings.')

if __name__=='__main__':
    main()
