# %% imports
from collections import OrderedDict
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.gofplots import qqplot, qqplot_2samples

from project_data import get_stored_anova_data

# %% globals

VAR_NAME_FOR_ANALYSIS = 't_remaining_s'


# %% data manipulation fx
def _slice_timewindows_for_analysis(df: pd.DataFrame) -> OrderedDict:
    d_datasets = OrderedDict()
    max_t_in_data = df['t_remaining_s'].max()
    if max_t_in_data > 120:
        d_datasets['Last 3 Minute'] = df.copy()
    if max_t_in_data > 60:
        d_datasets['Last 2 Minute'] = df[df['t_remaining_s'] <= 120].copy()
    
    df_1_min = df[df['t_remaining_s'] <= 60] if max_t_in_data > 60 else df
    
    if max_t_in_data >= 60:
        d_datasets['Last 1 Minute'] = df_1_min.copy()
    if max_t_in_data > 30:
        d_datasets['Last 30 Seconds'] = df_1_min[df_1_min['t_remaining_s'] <= 30].copy()
    if max_t_in_data > 20:
        d_datasets['Last 20 Seconds'] = df_1_min[df_1_min['t_remaining_s'] <= 20].copy()
    if max_t_in_data > 10:
        d_datasets['Last 10 Seconds'] = df_1_min[df_1_min['t_remaining_s'] <= 10].copy()
    
    assert len(d_datasets) > 0
    return d_datasets


def _split_into_anova_buckets(df: pd.DataFrame, var_name: str = 't_remaining_s') -> OrderedDict:
    d_group = OrderedDict()
    d_group['Following Timeout (Home)'] = df[(df['is_home']) & (df['poss_follow_to'])][var_name].copy()
    d_group['Following Timeout (Away)'] = df[(~df['is_home']) & (df['poss_follow_to'])][var_name].copy()
    d_group['Regular Play (Home)'] = df[(df['is_home']) & (~df['poss_follow_to'])][var_name].copy()
    d_group['Regular Play (Away)'] = df[(~df['is_home']) & (~df['poss_follow_to'])][var_name].copy()
    return d_group


def _rearrange_into_anova_bucket_pairs(d_datasets: OrderedDict) -> OrderedDict:
    d_datasets_pairs = OrderedDict()
    for groupname, group in d_datasets.items():
        d_datasets_pairs[groupname] = OrderedDict()
        d_datasets_pairs[groupname]['Following Timeout (Home vs. Away)'] = [group['Following Timeout (Home)'].copy(),
                                                                            group['Following Timeout (Away)'].copy()]
        d_datasets_pairs[groupname]['Regular Play (Home vs. Away)'] = [group['Regular Play (Home)'].copy(),
                                                                       group['Regular Play (Away)'].copy()]
        d_datasets_pairs[groupname]['Timeout vs. Regular Play (Home)'] = [group['Following Timeout (Home)'].copy(),
                                                                          group['Regular Play (Home)'].copy()]
        d_datasets_pairs[groupname]['Timeout vs. Regular Play (Away)'] = [group['Following Timeout (Away)'].copy(),
                                                                          group['Regular Play (Away)'].copy()]
    return d_datasets_pairs


# %% plotting fx

def plot_qq(d_datasets: OrderedDict, var_name: str = 't_remaining_s', **kwargs):
    # https://www.statsmodels.org/devel/generated/statsmodels.graphics.gofplots.qqplot_2samples.html
    # https://www.statsmodels.org/devel/generated/statsmodels.graphics.gofplots.qqplot.html
    
    n_datasets = len(d_datasets)
    m_per_dataset = len(d_datasets[list(d_datasets.keys())[0]])
    total = n_datasets * m_per_dataset
    
    plt.figure(figsize=(18, 16))
    i = 0
    y_lims = (0, 120)
    x_lims = (-2.5, 2.5)
    for groupname, group in d_datasets.items():
        for sample_name, sample in group.items():
            i += 1
            _ax = plt.subplot(n_datasets, m_per_dataset, i)
            # _res = stats.probplot(sample, plot=plt)
            _res = qqplot(data=sample, ax=_ax, line='r', markersize=1)
            _ax.set_title("")
            _ax.grid(True)
            _ax.set_xlim(x_lims)
            _ax.xaxis.set_label_position('top')
            plt.axvline(x=0)
            
            if ((i - 1) % m_per_dataset) == 0:
                y_lims = (min(sample), max(sample))
                _ax.set_ylabel(groupname)
                _ax.set_ylim(y_lims)
            else:
                _ax.set_ylabel("")
                _ax.set_yticklabels([])
                _ax.set_ylim(y_lims)
            
            if i <= m_per_dataset:
                _ax.set_xlabel(sample_name)
                _ax.set_xticklabels([])
            elif i > (total - m_per_dataset):
                _ax.set_xlabel("")
            else:
                _ax.set_xlabel("")
                _ax.set_xticklabels([])
            # plt.setp(visible=False)
    
    plt.suptitle(
        "1-sample QQ Plots by ANOVA group\nvariable: '{}'\nrows = time window filter\ncolumns = ANOVA group\nx-axis = Theoretical Quantiles\ny-axis=Sample Quantiles".format(
            var_name))
    # plt.tight_layout()
    plt.savefig('images/qq_plot_timewindows.png')
    plt.show()
    
    return


def plot_qq_pairs(d_datasets_pairs: OrderedDict, var_name: str = 't_remaining_s', **kwargs):
    # https://www.statsmodels.org/devel/generated/statsmodels.graphics.gofplots.qqplot_2samples.html
    # https://www.statsmodels.org/devel/generated/statsmodels.graphics.gofplots.qqplot.html
    
    n_datasets = len(d_datasets_pairs)
    m_per_dataset = len(d_datasets_pairs[list(d_datasets_pairs.keys())[0]])
    total = n_datasets * m_per_dataset
    
    plt.figure(figsize=(18, 16))
    i = 0
    y_lims = (-2.5, 2.5)
    x_lims = (-2.5, 2.5)
    for groupname, group in d_datasets_pairs.items():
        for sample_name, sample in group.items():
            i += 1
            _ax = plt.subplot(n_datasets, m_per_dataset, i)
            # _res = stats.probplot(sample, plot=plt)
            _res = qqplot_2samples(data1=sample[0], data2=sample[1], ax=_ax, line='45')
            _ax.set_title("")
            _ax.grid(True)
            _ax.set_xticklabels([])
            # _ax.set_xlim(x_lims)
            # _ax.set_ylim(y_lims)
            _ax.xaxis.set_label_position('top')
            plt.axvline(x=0)
            
            if ((i - 1) % m_per_dataset) == 0:
                _ax.set_ylabel(groupname)
            else:
                _ax.set_ylabel("")
                _ax.set_yticklabels([])
            
            if i <= m_per_dataset:
                _ax.set_xlabel(sample_name)
            elif i > (total - m_per_dataset):
                _ax.set_xlabel("")
            else:
                _ax.set_xlabel("")
            # plt.setp(visible=False)
    
    plt.suptitle(
        "2-sample QQ Plots by ANOVA group\nvariable: '{}'\nrows = time window filter\ncolumns = ANOVA group pairs\nx-axis = ANOVA group #1\ny-axis=ANOVA group #2".format(
            var_name))
    # plt.tight_layout()
    plt.savefig('images/qq_plot_timewindows_pairs.png')
    plt.show()
    
    return


# %% main

def main():
    d_datasets = _slice_timewindows_for_analysis(df=get_stored_anova_data())
    for groupname, group in d_datasets.items():
        d_datasets[groupname] = _split_into_anova_buckets(df=group, var_name=VAR_NAME_FOR_ANALYSIS)
    plot_qq(d_datasets=d_datasets, var_name=VAR_NAME_FOR_ANALYSIS)
    
    d_datasets_pairs = _rearrange_into_anova_bucket_pairs(d_datasets=d_datasets)
    plot_qq_pairs(d_datasets_pairs=d_datasets_pairs, var_name=VAR_NAME_FOR_ANALYSIS)
    
    return


if __name__ == '__main__':
    main()
