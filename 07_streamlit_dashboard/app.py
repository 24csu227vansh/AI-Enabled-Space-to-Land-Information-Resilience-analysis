"""Streamlit dashboard: descriptive research statistics and reproducible figures."""
from pathlib import Path
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
METHODS = ['Rule-Based', 'DQN', 'PPO']
CONDITIONS = ['normal', 'mild', 'moderate', 'severe']

@st.cache_data
def read_csv(name):
    return pd.read_csv(DATA / name)

def figure_bytes(fig):
    output = io.BytesIO()
    fig.savefig(output, format='png', dpi=200, bbox_inches='tight')
    output.seek(0)
    return output.getvalue()

def show_figure(fig, filename, caption):
    st.pyplot(fig)
    st.caption(caption)
    st.download_button('Download PNG', figure_bytes(fig), file_name=filename, mime='image/png', key=filename)
    plt.close(fig)

def chart_rl():
    df = read_csv('rl_seed_means.csv')
    stats = df.groupby(['method','condition'], as_index=False)['performance'].agg(['mean','std']).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(CONDITIONS))
    for j, method in enumerate(METHODS):
        part = stats[stats.method.eq(method)].set_index('condition').reindex(CONDITIONS)
        ax.errorbar(x, part['mean'], yerr=part['std'].fillna(0), marker='o', capsize=3, label=method)
    ax.set_xticks(x, ['Normal','Mild','Moderate','Severe'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Normalized proxy performance (higher is better)')
    ax.set_title('Figure 2. RL performance across degradation conditions')
    ax.grid(axis='y', alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig, stats

def chart_ri():
    df = read_csv('rl_seed_resilience.csv')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(3)
    w = .24
    for j, method in enumerate(METHODS):
        part = df[df.method.eq(method)]
        means = [part[f'{c}_RI'].mean() for c in ('mild','moderate','severe')]
        stds = [part[f'{c}_RI'].std(ddof=1) for c in ('mild','moderate','severe')]
        ax.bar(x+(j-1)*w, means, width=w, yerr=stds, capsize=3, label=method)
    ax.set_xticks(x, ['Mild','Moderate','Severe'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Resilience Index = degraded / normal')
    ax.set_title('Figure 3. Retained performance by degradation level')
    ax.legend()
    fig.tight_layout()
    return fig, df

def chart_dropout():
    df = read_csv('dropout_summary.csv')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for method, label in [('S1_only','S1 only'),('S2_only','S2 only'),('S1_or_S2','S1 or S2')]:
        part = df[df.method.eq(method)].sort_values('dropout_probability')
        ax.plot(part.dropout_probability * 100, part['mean'] * 100, marker='o', label=label)
    ax.set(xlabel='Simulated dropout probability (%)', ylabel='Information availability (%)', ylim=(0,105), title='Supplementary. Availability under independent sensor dropout')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig, df

def chart_correlated():
    df = read_csv('satellite_correlated_dropout.csv')
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = {0.0:'Independent',0.5:'50% shared failure',1.0:'Fully shared'}
    for fraction in [0.,.5,1.]:
        part = df[np.isclose(df.shared_failure_fraction, fraction)].sort_values('dropout_probability')
        ax.plot(part.dropout_probability*100, part.either_mean*100, marker='o', label=labels[fraction])
    ax.set(xlabel='Marginal dropout probability (%)',ylabel='Either sensor available (%)',ylim=(0,105),title='Figure 4. Availability under shared sensor failures')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    return fig, df

def chart_prediction():
    df = read_csv('satellite_split_metrics.csv')
    labels = ['Train-mean baseline','S1 only','S2 only','S1 + S2']
    part = df.groupby('method')['mae_cloud_percentage_points'].agg(['mean','std']).reindex(labels)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, part['mean'], yerr=part['std'], capsize=4)
    ax.set(ylabel='MAE (cloud-percentage points; lower is better)',title='Figure 5. Sentinel-2 cloud-metadata prediction')
    ax.tick_params(axis='x', labelrotation=12)
    fig.tight_layout()
    return fig, df

st.set_page_config(page_title='Space-to-Land | Research Statistics', layout='wide')
st.title('Satellite Information Resilience — Statistical Dashboard')
st.caption('Reproducible figures from recorded RL episodes and preliminary S1/S2 experiments. Environmental information analysis only.')
st.info('RL experiments are synthetic. Satellite availability uses simulated failures. The cloud-metadata model is not a land-cover classifier. Three RL training seeds and five overlapping satellite splits limit inference.')
page = st.sidebar.radio('Choose analysis', ['Overview','RL performance','Resilience Index','Independent dropout','Correlated failures','Satellite prediction','Statistical tables'])

if page == 'Overview':
    st.subheader('Study design')
    st.markdown('**RL:** PPO, DQN, Rule-Based; 3 training seeds × 4 conditions × 30 episodes per seed/method/condition. **Satellite:** 100 sample locations and 400 observations per modality; separate dropout and metadata-prediction analyses.')
    c1,c2,c3 = st.columns(3)
    c1.metric('RL evaluation episodes', '1,080')
    c2.metric('RL training seeds', '3')
    c3.metric('Satellite sample locations', '100')
    st.markdown('**Paper mapping:** Figure 1 is the methodology architecture (not a statistical graph). The dashboard reproduces statistical Figures 2–5 and an additional independent-dropout plot.')
elif page == 'RL performance':
    fig, table = chart_rl()
    show_figure(fig, 'figure2_rl_performance_python.png', 'Error bars are standard deviations across 3 seed means, not independent-episode confidence intervals.')
    st.dataframe(table, use_container_width=True)
elif page == 'Resilience Index':
    fig, table = chart_ri()
    show_figure(fig, 'figure3_resilience_python.png', 'RI is computed within each seed using the same higher-is-better proxy score.')
    st.dataframe(table, use_container_width=True)
elif page == 'Independent dropout':
    fig, table = chart_dropout()
    show_figure(fig, 'supplementary_independent_dropout_python.png', 'Information availability only; not prediction accuracy.')
    st.dataframe(table, use_container_width=True)
elif page == 'Correlated failures':
    fig, table = chart_correlated()
    show_figure(fig, 'figure4_correlated_dropout_python.png', 'Failure correlations are simulated assumptions, not observed sensor outage rates.')
    st.dataframe(table, use_container_width=True)
elif page == 'Satellite prediction':
    fig, table = chart_prediction()
    show_figure(fig, 'figure5_satellite_prediction_python.png', 'Error bars show standard deviation across 5 overlapping train/test splits; these are not independent replications.')
    st.dataframe(table, use_container_width=True)
    st.subheader('Mean R² by method')
    st.dataframe(read_csv('satellite_model_statistics.csv')[['method','r2_mean','mae_mean']], use_container_width=True)
else:
    filename = st.selectbox('Source CSV', sorted(x.name for x in DATA.glob('*.csv')))
    df = read_csv(filename)
    st.dataframe(df, use_container_width=True)
    st.download_button('Download source CSV', (DATA/filename).read_bytes(), file_name=filename, mime='text/csv')

st.divider()
st.caption('Source files are bundled locally. The dashboard does not fetch external datasets or alter the underlying experimental records.')
