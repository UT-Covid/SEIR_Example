#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray
from src.seir import *
from src.implicit_node import *
from copy import deepcopy


# requirements for model run
# - travel data frame
# - baseline contact
# - starting population sizes
# - epi params

# In[68]:


# validation tolerances:
abs_tol= 0.1 #0.001 max persons difference
rel_tol = 0.05 #0.05% max difference


# In[69]:


params_template = {
  "mu": 0.0,
  "sigma": 0.5,
  "beta": 0.1,
  "gamma": 0.2,
  "omega": 0.1,
  "start_S": [[24, 0], [49, 0]],
  "start_E": [[0, 0], [0, 0]],
  "start_I": [[1, 0], [1, 0]],
  "start_R": [[0, 0], [0, 0]],
  "days": 30,
  "outpath": "outputs/multiple_nodes",
  "phi": [], # fill in after partitioning
  "n_sims": 1,
  "stochastic": "False",
  "n_age": 2,
  "n_nodes": 2,
  "sim_idx": 0, # single deterministic run
  "interval_per_day": 10
}


# In[70]:


travel = pd.read_csv('inputs/travel2.csv')


# In[71]:


travel


# In[72]:


contact = pd.read_csv('inputs/contact.csv')


# In[73]:


partition = partition_contacts(travel, contact, daily_timesteps=10)
phi_matrix = contact_matrix(partition)


# In[74]:


phi_matrix


# In[75]:


test1 = deepcopy(params_template)
test1['phi'] = phi_matrix
test1_model = SEIR(test1)
test1_model.seir()
test1_model.final


# In[76]:


test1_model.beta


# In[77]:


test1_model.stochastic


# In[78]:


test1_model.plot_timeseries()


# In[79]:


def discrete_time_approx(rate, timestep):
    """

    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """

    return (1 - (1 - rate)**(1/timestep))


# In[80]:


ref_params = {
  "mu": 0.0,
  "sigma": 0.5,
  "beta": 0.1,
  "gamma": 0.2,
  "omega": 0.1,
  "start_S": [[73, 0]],
  "start_E": [[0, 0]],
  "start_I": [[2, 0]],
  "start_R": [[0, 0]],
  "days": 30,
  "outpath": "outputs/single_node",
  "phi": [[[[5/10, 0], [0, 0]]]],
  "n_sims": 1,
  "stochastic": "False",
  "n_age": 2,
  "n_nodes": 1,
  "sim_idx": 0,
  "interval_per_day": 10
}


# In[81]:


ref_model = SEIR(ref_params)
ref_model.seir()


# In[82]:


ref_model.plot_timeseries()


# In[83]:


def test_partition(test, ref, method=xarray.testing.assert_allclose, **kwargs):

    test_s = xr_summary(test.final.S, sel={'age': 'young'}, timeslice=slice(0, test.duration), sum_over='node')
    ref_s = xr_summary(ref.final.S, sel={'age': 'young'}, timeslice=slice(0, ref.duration), sum_over='node')

    diff_s = test_s - ref_s

    test_e = xr_summary(test.final.E, sel={'age': 'young'}, timeslice=slice(0, test.duration), sum_over='node')
    ref_e = xr_summary(ref.final.E, sel={'age': 'young'}, timeslice=slice(0, ref.duration), sum_over='node')

    diff_e = test_e - ref_e

    test_i = xr_summary(test.final.I, sel={'age': 'young'}, timeslice=slice(0, test.duration), sum_over='node')
    ref_i = xr_summary(ref.final.I, sel={'age': 'young'}, timeslice=slice(0, ref.duration), sum_over='node')

    diff_i = test_i - ref_i

    test_r = xr_summary(test.final.R, sel={'age': 'young'}, timeslice=slice(0, test.duration), sum_over='node')
    ref_r = xr_summary(ref.final.R, sel={'age': 'young'}, timeslice=slice(0, ref.duration), sum_over='node')

    diff_r = test_r - ref_r

    try:
        method(test_s, ref_s, **kwargs)
    except AssertionError:
        print('Differing values for susceptible timeseries.')

    try:
        method(test_e, ref_e, **kwargs)
    except AssertionError:
        print('Differing values for exposed timeseries.')

    try:
        method(test_i, ref_i, **kwargs)
    except AssertionError:
        print('Differing values for infected timeseries.')

    try:
        method(test_r, ref_r, **kwargs)
    except AssertionError:
        print('Differing values for recovered timeseries.')

    return diff_s, diff_e, diff_i, diff_r


# In[84]:


s_diff, e_diff, i_diff, r_diff = test_partition(test1_model, ref_model, atol=abs_tol, rtol=rel_tol)


# In[85]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

s_diff.plot(ax=ax, color='b')
e_diff.plot(ax=ax, color='g')
i_diff.plot(ax=ax, color='r')
r_diff.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("N Partition - N Baseline")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Two Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
plt.tight_layout()
#ax.set_ylim(-2, 3)
plt.axhline(y=0, c='gray', ls='dotted')

plt.show()


# In[86]:


test_partition(test1_model, ref_model, method=xarray.testing.assert_equal)


# # Sensitivity analysis: contact rate

# In[87]:


def update_contact_rate(contact_df, new_rate, age1='young', age2='young'):

    contact_idx = contact_df.set_index(['age1', 'age2'])
    contact_dict = contact_idx.to_dict()
    contact_dict['daily_per_capita_contacts'][(age1, age2)] = new_rate
    updated_df = pd.DataFrame.from_dict(contact_dict).reset_index()
    updated_df.columns = ['age1', 'age2', 'daily_per_capita_contacts']

    return updated_df


# In[88]:


update_contact_rate(contact, 0.3)


# In[89]:


polymod = pd.read_csv('./data/Cities_Data/ContactMatrixAll_5AgeGroups.csv', header=None)


# In[90]:


max_cr = polymod.max().max()
min_cr = polymod.min().min()
print(min_cr, max_cr)


# In[91]:


max_contact = update_contact_rate(contact, max_cr)
max_partition = partition_contacts(travel, max_contact, daily_timesteps=10)
max_phi_matrix = contact_matrix(max_partition)


# In[92]:


test_max = deepcopy(params_template)
test_max['phi'] = max_phi_matrix
test_max_model = SEIR(test_max)
test_max_model.seir()


# In[93]:


ref_max = deepcopy(ref_params)
ref_max['phi'] = [[[[max_cr/10, 0], [0, 0]]]]
ref_max_model = SEIR(ref_max)
ref_max_model.seir()


# In[94]:


ds_max, de_max, di_max, dr_max = test_partition(test_max_model, ref_max_model, atol=abs_tol, rtol=rel_tol)


# In[95]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ds_max.plot(ax=ax, color='b')
de_max.plot(ax=ax, color='g')
di_max.plot(ax=ax, color='r')
dr_max.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("N Partition - N Baseline")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Two Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
plt.tight_layout()
#ax.set_ylim(-2, 3)
plt.axhline(y=0, c='gray', ls='dotted')

plt.show()


# In[96]:


min_contact = update_contact_rate(contact, min_cr)
min_partition = partition_contacts(travel, min_contact, daily_timesteps=10)
min_phi_matrix = contact_matrix(min_partition)

test_min = deepcopy(params_template)
test_min['phi'] = min_phi_matrix
test_min_model = SEIR(test_min)
test_min_model.seir()

ref_min = deepcopy(ref_params)
ref_min['phi'] = [[[[min_cr/10, 0], [0, 0]]]]
ref_min_model = SEIR(ref_min)
ref_min_model.seir()


# In[97]:


ds_min, de_min, di_min, dr_min = test_partition(test_min_model, ref_min_model, atol=abs_tol, rtol=rel_tol)


# In[98]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ds_min.plot(ax=ax, color='b')
de_min.plot(ax=ax, color='g')
di_min.plot(ax=ax, color='r')
dr_min.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("N Partition - N Baseline")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Two Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
plt.tight_layout()
#ax.set_ylim(-2,3)
plt.axhline(y=0, c='gray', ls='dotted')

plt.show()


# In[99]:


ref_min_model.plot_timeseries()


# In[100]:


test_min_model.plot_timeseries()


# In[101]:


ref_max_model.plot_timeseries()


# In[102]:


test_max_model.plot_timeseries()


# # Sensitivity analysis: population size

# In[103]:


pop_template = {
  "mu": 0.0,
  "sigma": 0.5,
  "beta": 0.1,
  "gamma": 0.2,
  "omega": 0.1,
  "start_S": [[24, 0], [49, 0]],
  "start_E": [[0, 0], [0, 0]],
  "start_I": [[1, 0], [1, 0]],
  "start_R": [[0, 0], [0, 0]],
  "days": 30,
  "outpath": "outputs/multiple_nodes",
  "phi": [], # fill in after partitioning
  "n_sims": 1,
  "stochastic": "False",
  "n_age": 2,
  "n_nodes": 2,
  "sim_idx": 0, # single deterministic run
  "interval_per_day": 10
}


# In[104]:


def update_travel(travel_df, new_count, source, destination, age_src='young', age_dest='young'):

    travel_idx = travel_df.set_index(['source', 'destination', 'age_src', 'age_dest'])
    travel_dict = travel_idx.to_dict()
    travel_dict['n'][(source, destination, age_src, age_dest)] = new_count
    updated_df = pd.DataFrame.from_dict(travel_dict).reset_index()
    updated_df.columns = ['source', 'destination', 'age_src', 'age_dest', 'destination_type', 'n']

    return updated_df


# In[105]:


def update_start_pop(travel_df):

    grouped = travel_df.groupby(['source', 'age_src'])['n'].sum().reset_index()

    nodes = sorted(travel_df['source'].unique())
    ages = sorted(travel_df['age_src'].unique(), reverse=True)

    pop_arr_s = np.zeros([len(nodes), len(ages)])
    pop_arr_e = np.zeros([len(nodes), len(ages)])
    pop_arr_i = np.zeros([len(nodes), len(ages)])
    pop_arr_r = np.zeros([len(nodes), len(ages)])

    for i, node in enumerate(nodes):
        for j, age in enumerate(ages):
            new_total = grouped[(grouped['source']==node) & (grouped['age_src']==age)]['n'].item()
            if new_total > 2:
                pop_arr_s[i, j] = new_total-1
                pop_arr_i[i, j] = 1

    return pop_arr_s, pop_arr_e, pop_arr_i, pop_arr_r


# In[106]:


new_travel = update_travel(travel, 50, source='A', destination='A')
pop_s, pop_e, pop_i, pop_r = update_start_pop(new_travel)


# In[ ]:





# # Sensitivity analysis: additional nodes

# In[107]:


travel3 = pd.read_csv('inputs/travel3.csv')


# In[108]:


travel3


# In[109]:


partition3 = partition_contacts(travel3, contact, daily_timesteps=10)
phi_matrix3 = contact_matrix(partition3)
pop_s, pop_e, pop_i, pop_r = update_start_pop(travel3)
test3 = deepcopy(params_template)
test3['phi'] = phi_matrix3
test3['start_S'] = pop_s
test3['start_E'] = pop_e
test3['start_I'] = pop_i
test3['start_R'] = pop_r
test3['n_nodes'] = 3


# In[110]:


test3_model = SEIR(test3)
test3_model.seir()


# In[134]:


test3


# In[111]:


ref3 = deepcopy(params_template)
ref3['phi'] = [[[[5/10, 0], [0, 0]]]]
ref3_s = np.array([sorted(travel3.groupby(['age_src'])['n'].sum(), reverse=True)])
ref3['start_S'] = ref3_s[0, 0] - 3
ref3['start_E'] = np.array([[0, 0]])
ref3['start_I'] = np.array([[3, 0]])
ref3['start_R'] = np.array([[0, 0]])
ref3['n_nodes'] = 1


# In[112]:


ref3_model = SEIR(ref3)
ref3_model.seir()


# In[113]:


test3_model.plot_timeseries()


# In[114]:


ref3_model.plot_timeseries()


# In[115]:


ref3['start_S']


# In[116]:


s3_diff, e3_diff, i3_diff, r3_diff = test_partition(test3_model, ref3_model, atol=abs_tol, rtol=rel_tol)


# In[117]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

s3_diff.plot(ax=ax, color='b')
e3_diff.plot(ax=ax, color='g')
i3_diff.plot(ax=ax, color='r')
r3_diff.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("N Partition - N Baseline")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Three Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
plt.tight_layout()
#ax.set_ylim(-2, 3)
plt.axhline(y=0, c='gray', ls='dotted')

plt.show()


# In[118]:


local3v2_contact5_s = s3_diff - s_diff
local3v2_contact5_e = e3_diff - e_diff
local3v2_contact5_i = i3_diff - i_diff
local3v2_contact5_r = r3_diff - r_diff


# In[119]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

local3v2_contact5_s.plot(ax=ax, color='b')
local3v2_contact5_e.plot(ax=ax, color='g')
local3v2_contact5_i.plot(ax=ax, color='r')
local3v2_contact5_r.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("Population Size")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Three Local, One Contextual vs Two Local, One Contextual")
plt.tight_layout()

plt.show()


# In[120]:


ref3_model.plot_timeseries()


# In[121]:


test3_model.final


# In[122]:


test3_model.plot_timeseries()


# In[123]:


partition3_min = partition_contacts(travel3, min_contact, daily_timesteps=10)
phi_matrix3_min = contact_matrix(partition3_min)

test3_min = deepcopy(test3)
test3_min['phi'] = phi_matrix3_min

ref3_min = deepcopy(ref3)
ref3_min['phi'] = [[[[min_cr/10, 0], [0, 0]]]]

test3_min_model = SEIR(test3_min)
test3_min_model.seir()

ref3_min_model = SEIR(ref3_min)
ref3_min_model.seir()


# In[124]:


s3_diff_min, e3_diff_min, i3_diff_min, r3_diff_min = test_partition(test3_min_model, ref3_min_model, atol=abs_tol, rtol=rel_tol)


# In[125]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

s3_diff_min.plot(ax=ax, color='b')
e3_diff_min.plot(ax=ax, color='g')
i3_diff_min.plot(ax=ax, color='r')
r3_diff_min.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.ylabel("N Partition - N Baseline")
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Three Local, One Contextual vs Baseline Mixing = 0.2 contacts/person/day")
plt.tight_layout()
plt.axhline(y=0, c='gray', ls='dotted')
#ax.set_ylim(-2, 3)

plt.show()


# In[126]:


partition3_max = partition_contacts(travel3, max_contact, daily_timesteps=10)
phi_matrix3_max = contact_matrix(partition3_max)

test3_max = deepcopy(test3)
test3_max['phi'] = phi_matrix3_max

ref3_max = deepcopy(ref3)
ref3_max['phi'] = [[[[max_cr/10, 0], [0, 0]]]]

test3_max_model = SEIR(test3_max)
test3_max_model.seir()

ref3_max_model = SEIR(ref3_max)
ref3_max_model.seir()


# In[127]:


s3_diff_max, e3_diff_max, i3_diff_max, r3_diff_max = test_partition(test3_max_model, ref3_max_model, atol=abs_tol, rtol=rel_tol)


# In[128]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

s3_diff_max.plot(ax=ax, color='b')
e3_diff_max.plot(ax=ax, color='g')
i3_diff_max.plot(ax=ax, color='r')
r3_diff_max.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("Three Local, One Contextual vs Baseline Mixing = 10.2 contacts/person/day")
plt.tight_layout()
plt.ylabel("N Partition - N Baseline")
plt.axhline(y=0, c='gray', ls='dotted')
#ax.set_ylim(-2, 3)

plt.show()

# DEBUG
# TODO
sys.exit(0)

# # 16 Nodes

# In[129]:


travel16 = pd.read_csv('inputs/travel16.csv')
partition16 = partition_contacts(travel16, contact, daily_timesteps=10)
phi_matrix16 = contact_matrix(partition16)
pop_s16, pop_e16, pop_i16, pop_r16 = update_start_pop(travel16)

test16 = deepcopy(params_template)
test16['phi'] = phi_matrix16
test16['start_S'] = pop_s16
test16['start_E'] = pop_e16
test16['start_I'] = pop_i16
test16['start_R'] = pop_r16
test16['n_nodes'] = 16

ref16 = deepcopy(params_template)
ref16['phi'] = [[[[5/10, 0], [0, 0]]]]
ref16_s = np.array([sorted(travel16.groupby(['age_src'])['n'].sum(), reverse=True)])
ref16['start_S'] = ref16_s[0, 0] - 16
ref16['start_E'] = np.array([[0, 0]])
ref16['start_I'] = np.array([[16, 0]])
ref16['start_R'] = np.array([[0, 0]])
ref16['n_nodes'] = 1

test16_model = SEIR(test16)
test16_model.seir()

ref16_model = SEIR(ref16)
ref16_model.seir()


# In[130]:


ref16_s


# In[131]:


s16_diff_max, e16_diff_max, i16_diff_max, r16_diff_max = test_partition(test16_model, ref16_model, atol=abs_tol, rtol=rel_tol)


# In[132]:


fig, ax = plt.subplots(1, 1, figsize=(8, 5))

s16_diff_max.plot(ax=ax, color='b')
e16_diff_max.plot(ax=ax, color='g')
i16_diff_max.plot(ax=ax, color='r')
r16_diff_max.plot(ax=ax, color='k')

plt.legend(("S", "E", "I", "R"), loc=0)
plt.xlabel("Time")
plt.xticks(rotation=45)
plt.title("16 Local, One Contextual vs Baseline Mixing = 5 contacts/person/day")
plt.tight_layout()
plt.ylabel("N Partition - N Baseline")
plt.axhline(y=0, c='gray', ls='dotted')
#ax.set_ylim(-2, 3)

plt.show()


# In[ ]:





# In[ ]:
