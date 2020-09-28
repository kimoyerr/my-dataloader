#!/usr/bin/env python
"""App to run some simulations to test PyTorch DataLoader performance


"""

import os
from itertools import product

import pandas as pd
import streamlit as st
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

from simulate import gen_sim_data
from dataset import SeqDataset, KmerTokenize
from doc2vec_DM import run_DM_model

from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_file, show

@st.cache # Only run once per session
def sim_data():
    gen_sim_data()

@st.cache # Only run once per session
def create_metrics_df():
    df = pd.DataFrame()


# Create dataset
def create_dataset(file_name):
    batch_size = 1
    context_size = 3
    kmer_hypers = {'k': 4, 'overlap': False, 'merge': True}
    ds = SeqDataset(file_name, kmer_size=kmer_hypers['k'], context_size=context_size,
                      transform=transforms.Compose([KmerTokenize(kmer_hypers)]))

    return ds

# Create empty metrics dataframe
metrics_df = create_metrics_df()

# Parameter search grids
param_grid = {'use_gpu': [True, False],
              'pin_memory': [True, False],
              'num_workers': [0, 1, 2, 4, 8],
              'batch_size': [1, 2, 4, 8, 16, 32]
              }

all_params = list(ParameterGrid(param_grid))
num_epochs = 1
shuffle = True

# Create dataset
SeqData = create_dataset('/home/ubuntu/pytorch-models/my-dataloader/data/01_raw/sim_1.txt')
# Perform runs against all the parameters
my_bar = st.progress(0.0)
count = 0
metrics_df = pd.DataFrame()

for param in all_params:
    print(count)
    print(len(all_params))
    my_bar.progress(count/len(all_params))
    st.info(param)
    tmp = run_DM_model(SeqData, num_epochs=num_epochs, batch_size=param['batch_size'], shuffle=shuffle,
                       pin_memory=param['pin_memory'],  num_workers=param['num_workers'], use_gpu=param['use_gpu'], non_blocking=True)
    data = pd.DataFrame({'batch_size' : param['batch_size'],
                       'num_workers' : param['num_workers'],
                       'shuffle': shuffle,
                       'pin_memory': param['pin_memory'],
                       'use_gpu': param['use_gpu'],
                        'total_time': tmp[0].sum + tmp[1].sum,
                        'data_load_time_avg': tmp[0].avg,
                        'data_load_time_count': tmp[0].count,
                        'DM_run_time_avg': tmp[1].avg,
                        'DM_run_time_count': tmp[1].count}, index=[metrics_df.shape[0]])
    metrics_df = metrics_df.append(data)
    st.write(metrics_df)
    count += 1
    metrics_df.to_csv('data/02_model_outputs/metrics_df.csv', index=False)


# Plots
batch_sizes = metrics_df.batch_size.unique().tolist()
worker_sizes = metrics_df.num_workers.unique().tolist()

# Time vs Num of Workers
sel_batch_size = st.selectbox('Select Batch Size', batch_sizes)
p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Time vs Number of Workers for Dataloader for Batch Size ' + str(sel_batch_size)
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.batch_size==sel_batch_size,:]
    p.line(df['num_workers'], df['total_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.click_policy="hide"
st.bokeh_chart(p)

p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Data Load Time vs Number of Workers for Dataloader for Batch Size ' + str(sel_batch_size)
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.batch_size==sel_batch_size,:]
    df['total_load_time'] = df['data_load_time_avg']*df['data_load_time_count']
    p.line(df['num_workers'], df['total_load_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.click_policy="hide"
st.bokeh_chart(p)

p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Model Time vs Number of Workers for Dataloader for Batch Size ' + str(sel_batch_size)
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.batch_size==sel_batch_size,:]
    df['total_model_time'] = df['DM_run_time_avg']*df['DM_run_time_count']
    p.line(df['num_workers'], df['total_model_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.click_policy="hide"
p.legend.background_fill_alpha = 0.5
st.bokeh_chart(p)


# Time vs Batch Sizes
sel_num_workers = st.selectbox('Select Number of Workers', worker_sizes)

# Time vs Num of Workers
p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Time vs Batch-Size for Dataloader for ' + str(sel_num_workers) + ' Number of Workers'
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.num_workers==sel_num_workers,:]
    p.line(df['batch_size'], df['total_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.background_fill_alpha = 0.5
p.legend.click_policy="hide"
st.bokeh_chart(p)

p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Data Load Time vs Batch-Size for Dataloader for ' + str(sel_num_workers) + ' Number of Workers'
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.num_workers==sel_num_workers,:]
    df['total_load_time'] = df['data_load_time_avg']*df['data_load_time_count']
    p.line(df['batch_size'], df['total_load_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.background_fill_alpha = 0.5
p.legend.click_policy="hide"
st.bokeh_chart(p)

p = figure(plot_width=800, plot_height=250)
p.title.text = 'Total Model Time vs Batch-Size for Dataloader for ' + str(sel_num_workers) + ' Number of Workers'
gpu_pin_combi = list(product(metrics_df.use_gpu.unique().tolist(), metrics_df.pin_memory.unique().tolist()))
for name, color in zip(gpu_pin_combi, Spectral4):
    print(name)
    df = metrics_df.loc[(metrics_df.use_gpu==name[0]) & (metrics_df.pin_memory==name[1]),:]
    df = df.loc[metrics_df.num_workers == sel_num_workers, :]
    df['total_model_time'] = df['DM_run_time_avg']*df['DM_run_time_count']
    p.line(df['batch_size'], df['total_model_time'], line_width=2, color=color, alpha=0.8,
           legend_label='use_gpu='+str(name[0])+' pin_memory='+str(name[1]))

p.legend.location = "top_right"
p.legend.background_fill_alpha = 0.5
p.legend.click_policy="hide"
st.bokeh_chart(p)

# DO PCA
SeqData = create_dataset('/home/ubuntu/pytorch-models/my-dataloader/data/01_raw/sim_1.txt')
model = run_DM_model(SeqData, num_epochs=5, batch_size=4, shuffle=True, pin_memory=True,
                   num_workers=4, use_gpu=True, non_blocking=True, do_pca=True)
# Plots kmer
plt.clf()
colors = {0: 'red', 1: 'blue'}
plt.scatter(model[3][:, 0], model[3][:, 1], c=[colors[i] for i in model[1].labels_])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
st.pyplot(plt)

# Plots Docs
plt.clf()
colors = {0: 'red', 1: 'blue'}
plt.scatter(model[4][:, 0], model[4][:, 1], c=[colors[i] for i in model[2].labels_])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
st.pyplot(plt)


# User selections
num_epochs = st.slider('Number of Epochs', min_value=1, max_value=10, value=1)
batch_size = st.slider('Data Loader Batch Size', min_value=1, max_value=50, value=1)
num_workers = st.slider('Number of Workers for DataLoader', min_value=0, max_value=8, value=0)
shuffle = st.radio("Shuffle Data for DataLoader?", ("False","True"))
pin_memory = st.radio("Pin Memory?", ("False","True"))
non_blocking = st.radio("Non Blocking?", ("False","True"))
use_gpu = st.radio("Use GPU?", ("False","True"))

submit = st.button('Run DM Model')
if submit:
    SeqData = create_dataset('/home/ubuntu/pytorch-models/my-dataloader/data/01_raw/sim_1.txt')
    tmp = run_DM_model(SeqData, num_epochs=num_epochs, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,  num_workers=num_workers, use_gpu=use_gpu, non_blocking=True)
    data = pd.DataFrame({'batch_size' : batch_size,
                       'num_workers' : num_workers,
                       'shuffle': shuffle,
                       'pin_memory': pin_memory,
                       'use_gpu': use_gpu,
                        'total_time': tmp[0].sum + tmp[1].sum,
                        'data_load_time_avg': tmp[0].avg,
                        'data_load_time_count': tmp[0].count,
                        'DM_run_time_avg': tmp[1].avg,
                        'DM_run_time_count': tmp[1].count}, index=[metrics_df.shape[0]])
    metrics_df = metrics_df.append(data)
    st.write(metrics_df)







