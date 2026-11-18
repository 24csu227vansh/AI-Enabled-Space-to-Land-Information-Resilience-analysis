import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from pathlib import Path
out=Path('/mnt/data/Research_Final_Validation');out.mkdir(exist_ok=True)
df=pd.read_csv('/mnt/data/Pasted text(3).txt'); assert len(df)==1080
seed=df.groupby(['method','condition','training_seed'],as_index=False).performance.mean()
seed.to_csv(out/'rl_seed_means.csv',index=False)
means=seed.groupby(['method','condition']).performance.agg(['mean','std','count']).reset_index()
means['ci95_low']=means['mean']-t.ppf(.975,means['count']-1)*means['std']/np.sqrt(means['count'])
means['ci95_high']=means['mean']+t.ppf(.975,means['count']-1)*means['std']/np.sqrt(means['count'])
means.to_csv(out/'rl_condition_statistics.csv',index=False)
wide=seed.pivot(index=['method','training_seed'],columns='condition',values='performance').reset_index()
for cond in ['mild','moderate','severe']:wide[cond+'_RI']=wide[cond]/wide['normal']
wide.to_csv(out/'rl_seed_resilience.csv',index=False)
ri=wide.groupby('method')[[x+'_RI' for x in ['mild','moderate','severe']]].agg(['mean','std']);ri.to_csv(out/'rl_resilience_statistics.csv')
comparisons=[]
for cond in ['normal','mild','moderate','severe']:
 for a,b in [('Rule-Based','DQN'),('Rule-Based','PPO'),('DQN','PPO')]:
  p=seed[seed.condition==cond].pivot(index='training_seed',columns='method',values='performance');delta=p[a]-p[b]; n=len(delta); sd=delta.std(ddof=1)
  comparisons.append(dict(condition=cond,method_a=a,method_b=b,n_seeds=n,mean_difference=delta.mean(),sd_seed_difference=sd,ci95_low=delta.mean()-t.ppf(.975,n-1)*sd/np.sqrt(n),ci95_high=delta.mean()+t.ppf(.975,n-1)*sd/np.sqrt(n)))
pd.DataFrame(comparisons).to_csv(out/'rl_paired_seed_differences.csv',index=False)
fusion=pd.read_csv('/mnt/data/Satellite_Fusion_Validation/model_metrics_by_split.csv');fusion.to_csv(out/'satellite_split_metrics.csv',index=False)
fs=fusion.groupby('method').agg(mae_mean=('mae_cloud_percentage_points','mean'),mae_sd=('mae_cloud_percentage_points','std'),r2_mean=('r2','mean')).reset_index();fs.to_csv(out/'satellite_model_statistics.csv',index=False)
cor=pd.read_csv('/mnt/data/Satellite_Experiment/correlated_dropout_summary.csv');cor.to_csv(out/'satellite_correlated_dropout.csv',index=False)
# charts
fig,ax=plt.subplots(figsize=(8.2,4.5)); order=['normal','mild','moderate','severe']; methods=['Rule-Based','DQN','PPO']
for m in methods:
 d=means[means.method==m].set_index('condition').loc[order];ax.plot(order,d['mean'],marker='o',label=m)
ax.set(ylabel='Normalized proxy performance',xlabel='Simulated information degradation',ylim=(.45,.92),title='RL methods: observed seed-mean performance');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'figure1_rl_performance.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,4.5));
for m in methods:
 d=wide[wide.method==m];ax.plot(['mild','moderate','severe'],[d[c+'_RI'].mean() for c in ['mild','moderate','severe']],marker='o',label=m)
ax.set(ylabel='Resilience Index (degraded / normal)',ylim=(.55,1),title='Resilience across simulated degradation');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'figure2_resilience.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,4.5)); fs2=fs.set_index('method').loc[['Train-mean baseline','S1 only','S2 only','S1 + S2']];ax.bar(fs2.index,fs2.mae_mean);ax.set(ylabel='MAE, cloud metadata percentage points',title='Held-out location cloud-metadata prediction');ax.tick_params(axis='x',rotation=12);fig.tight_layout();fig.savefig(out/'figure3_satellite_prediction.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.2,4.5));d=cor[np.isclose(cor.dropout_probability,.5)].sort_values('shared_failure_fraction');ax.plot(d.shared_failure_fraction*100,d.either_mean*100,marker='o');ax.set(xlabel='Shared-failure fraction (%)',ylabel='Either modality available (%)',title='Simulated 50% dropout: sensitivity to shared failures',ylim=(40,85));ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'figure4_shared_failures.png',dpi=180);plt.close(fig)
# document
D=Document();sec=D.sections[0];sec.header.paragraphs[0].text='RESEARCH VALIDATION  |  18 NOVEMBER 2026'
styles=D.styles;styles['Normal'].font.name='Aptos';styles['Normal'].font.size=Pt(9)
D.add_heading('AI-Enabled Space-to-Land Information Resilience: A Reinforcement Learning Framework for Situational Awareness',0)
D.add_paragraph('Consolidated research paper | Synthetic RL evaluation + preliminary real Sentinel-1/Sentinel-2 imagery analysis')
D.add_heading('Abstract',1)
D.add_paragraph('This study examines resilience of environmental information support under simulated observation degradation. Three information-management methods—Proximal Policy Optimization (PPO), Deep Q-Network (DQN), and a fixed Rule-Based policy—were compared using 1,080 recorded evaluation episodes spanning three training seeds, four conditions, and 30 episodes per seed-condition-method. The primary measure is a higher-is-better normalized proxy performance score and its Resilience Index (RI), defined as degraded/normal performance. Separately, user-provided SSL4EO-S12 Sentinel-1 and Sentinel-2 samples supported real image-feature extraction, synthetic sensor-availability stress tests, and preliminary prediction of Sentinel-2 cloud-percentage metadata. The Rule-Based method had the highest mean synthetic performance in all four conditions. Sentinel-2-only features slightly outperformed combined S1/S2 features on the metadata-prediction task. These findings do not establish reinforcement-learning superiority, accurate land-cover mapping, or operational decision advantage.')
D.add_paragraph('Keywords: Earth observation; Sentinel-1; Sentinel-2; information availability; reinforcement learning; resilience; multimodal sensing.')
D.add_heading('3.1 Introduction',1)
D.add_paragraph('Optical and radar Earth observations provide complementary views of land-surface conditions, but observation gaps, noise, and delayed acquisitions limit reliable environmental information. This paper studies non-operational information processing and human-supervised decision support. It does not model targeting, combat routing, weapon systems, or autonomous physical action.')
D.add_paragraph('Research questions: RQ1—How does satellite information contribute to environmental awareness? RQ2—Does S1/S2 fusion outperform individual modalities? RQ3—Do adaptive RL policies retain performance under information degradation? RQ4—How do individual modalities and failure correlations influence resilience? Current experiments provide only partial answers: simulated RQ3, and preliminary availability/prediction evidence for RQ2 and RQ4. A terrestrial-only comparison for RQ1 remains untested.')
D.add_heading('3.2 Related work',1)
D.add_paragraph('Deep learning for remote sensing is surveyed by Zhu et al. [1]. The SEN12MS benchmark describes co-registered Sentinel-1 radar, Sentinel-2 optical imagery, and land-cover labels [2]; this project instead uses a small SSL4EO-S12 sample, which does not supply independently validated land-cover labels for the present experiments [3]. PPO [4] and DQN [5] motivate alternative learning policies, but neither is guaranteed to exceed a well-designed fixed policy. Remote-sensing image fusion is relevant to complementary sensing [6].')
D.add_heading('3.3 Methodology',1)
D.add_heading('3.3.1 Data and design',2)
D.add_paragraph('RL input: 1,080 evaluation rows (3 methods × 4 degradation conditions × 3 seeds × 30 episodes), provided as a CSV-formatted text export. Training seeds were 17, 118, and 219. The synthetic normalized score is interpreted as higher-is-better. The proposed score is P = 0.4F + 0.3C + 0.3Q for normalized freshness, coverage, and reliability; these weights require independent sensitivity validation. Real-image input: 100 shared SSL4EO-S12 sample IDs, with 400 acquisitions per modality. S1 VV/VH and S2 B2/B3/B4/B8 summary features were extracted. Matching was by sample ID and nearest acquisition date; this does not prove pixel-level georegistration.')
D.add_heading('3.3.2 Evaluation and uncertainty',2)
D.add_paragraph('RI = mean(P_degraded) / mean(P_normal), with a strictly positive denominator. Method comparisons use means of 30 episodes within each training seed; the unit for between-policy uncertainty is the training seed (n=3), not all 90 episode observations. Descriptive 95% Student-t intervals are reported where defined; with three seeds these intervals are unstable and should not be treated as strong inferential evidence. Rule-Based results across seeds can be identical when the baseline is deterministic and evaluation scenarios are reused. Satellite prediction used five overlapping 75/25 location-disjoint splits (300 train/100 test records per split), so the five split metrics are not independent external validations.')
D.add_heading('3.3.3 Simulated dropout',2)
D.add_paragraph('At each location, each sensor can be unavailable with probability p. An independent-failure model and a shared-failure mixture model were tested; the shared-failure fraction is a chosen simulation parameter, not a measured outage correlation. The quantity reported is availability of at least one modality, not classification accuracy, learned fusion quality, or task-performance RI.')
D.add_heading('3.4 Results and discussion',1)
D.add_heading('3.4.1 RL performance',2)
def table(headers,rows):
 tb=D.add_table(rows=1,cols=len(headers));tb.style='Light Shading Accent 1';tb.alignment=WD_TABLE_ALIGNMENT.CENTER
 for i,h in enumerate(headers):tb.rows[0].cells[i].text=str(h)
 for row in rows:
  c=tb.add_row().cells
  for i,x in enumerate(row):c[i].text=str(x)
 return tb
rows=[]
for m in methods:
 p=means[means.method==m].set_index('condition');rows.append([m]+[f'{p.loc[c,"mean"]:.4f}' for c in order])
table(['Method']+[x.title() for x in order],rows)
D.add_picture(str(out/'figure1_rl_performance.png'),width=Inches(5.9))
D.add_paragraph('Figure 1. Means across the three training seeds for each condition. Differences are small; this plot is descriptive.')
rows=[]
for m in methods:
 q=wide[wide.method==m];rows.append([m]+[f'{q[c+"_RI"].mean():.4f}' for c in ['mild','moderate','severe']])
table(['Method','Mild RI','Moderate RI','Severe RI'],rows)
D.add_picture(str(out/'figure2_resilience.png'),width=Inches(5.9))
D.add_paragraph('Figure 2. Mean seed-level RI; higher indicates greater retained normalized proxy performance. The Rule-Based method had the highest mean performance in each condition; differences are not evidence of general superiority.')
D.add_heading('3.4.2 Satellite availability',2)
D.add_paragraph('For a marginal dropout probability of 0.5, the theoretical probability that at least one modality is available is 0.75 under independent failures, 0.625 under a 50% shared-failure mixture, and 0.50 under fully shared failures. The corresponding Monte Carlo experiment tested the same assumptions; results demonstrate redundancy benefits only under incomplete failure correlation.')
D.add_picture(str(out/'figure4_shared_failures.png'),width=Inches(5.9))
D.add_paragraph('Figure 3. Availability under simulated failures; these are not empirical outage estimates.')
D.add_heading('3.4.3 Real-image feature prediction',2)
rows=[]
for _,r in fs2.reset_index().iterrows():rows.append([r['method'],f'{r.mae_mean:.3f}',f'{r.r2_mean:.3f}'])
table(['Feature set','Mean MAE (percentage points)','Mean R²'],rows)
D.add_picture(str(out/'figure3_satellite_prediction.png'),width=Inches(5.9))
D.add_paragraph('Figure 4. Prediction of S2 cloud-percentage metadata. S2-only had the lowest mean absolute error (2.481 percentage points); S1+S2 (2.502) did not improve on S2-only. All mean R² values were near zero or negative, indicating limited predictive explanatory power. This is neither pixel-level cloud mapping nor land-cover classification.')
D.add_heading('3.4.4 Statistical interpretation',2)
D.add_paragraph('The accompanying CSV files contain seed-level scores, descriptive t intervals, paired seed differences, and split-level satellite metrics. Repeated episode scores within the same training run and repeated train/test splits must not be counted as independent training replications. No unqualified p-value-based superiority claim is made. The existing record also does not independently verify model convergence, training budget equivalence, or robustness to alternative reward weights.')
D.add_heading('3.5 Conclusion',1)
D.add_paragraph('This study establishes a reproducible preliminary evidence chain spanning synthetic RL information management, real satellite-image feature extraction, and simulated observation availability. The Rule-Based baseline slightly exceeded PPO and DQN on the current synthetic proxy score; S2-only slightly exceeded S1+S2 on the cloud-metadata prediction task. Fusion redundancy increased availability only when sensor failures were not fully shared. The work supports cautious evaluation of environmental information resilience, not claims of operational effectiveness or validated land-cover decision superiority.')
D.add_heading('Limitations and future work',1)
D.add_paragraph('Obtain reliable independent land-cover labels and a larger geographically diverse S1/S2 sample; verify georegistration and acquisition-date matching; conduct location-disjoint and geographic holdout evaluation; evaluate classification macro-F1 and per-class metrics; retrain PPO/DQN over more independent seeds with fixed budgets; report sensitivity to score weights and dropout mechanisms; and connect learned policies to image-derived quality indicators before asserting end-to-end improvement.')
D.add_heading('References',1)
refs=['[1] Zhu, X. X., et al. (2017). Deep Learning in Remote Sensing: A Comprehensive Review and List of Resources. IEEE GRSM. DOI: 10.1109/MGRS.2017.2762307.','[2] Schmitt, M., et al. (2019). SEN12MS: A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion. DOI: 10.5194/isprs-annals-IV-2-W7-153-2019.','[3] SSL4EO-S12 dataset project. https://github.com/zhu-xlab/SSL4EO-S12 (dataset documentation; verify release/version when citing).','[4] Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.','[5] Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature. DOI: 10.1038/nature14236.','[6] Li, et al. (2022). Deep learning in multimodal remote sensing data fusion: A comprehensive review. DOI: 10.1016/j.jag.2022.102926.']
for x in refs:D.add_paragraph(x,style='Normal')
D.add_heading('Reproducibility note',1)
D.add_paragraph('The analysis script and derived CSVs are provided alongside this document. The original uploaded imagery is not redistributed in the results package. This report does not assert any unperformed image-label validation or newly trained RL policies.')
D.save(out/'Research_Paper_Consolidated.docx')
with open(out/'README.txt','w') as f:f.write('Consolidated validation of user-supplied V3 episode CSV and previously generated satellite experiment CSVs. Run make_final_validation.py from /mnt/data with original input paths. Data: three training seeds; five overlapping satellite splits. Figures and DOCX derived directly from CSV. No new PPO/DQN training performed.\n')
print('rl means\n',means[['method','condition','mean']].to_string(index=False));print('satellite\n',fs.to_string(index=False));print('created',out)
