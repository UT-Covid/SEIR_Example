#############################################
# Calculate contact rates in implicit nodes #
#############################################

"""
Explicit nodes are nodes that have axes in the contact matrix. Implicit nodes are nodes that are implied by
contact between individuals in two different explicit nodes. For example, explicit nodes may be spatially defined,
non-overlapping areas like census block groups. The presence of schools in the model is "implied" by contact between
two or more explicit nodes that reflects some of the population in those nodes interacting.

Inputs needed
1. number of people from node A (example: 25 people)
2. number of people from node B (example: 50 people)
3. contact rate in implied node (example: 5 daily contacts per person)
"""

import pandas as pd
import numpy as np
from collections import defaultdict

# breaks single defintion rule -- also present in seir.py
def discrete_time_approx(rate, timestep):
    """

    :param rate: daily rate
    :param timestep: timesteps per day
    :return: rate rescaled by time step
    """

    return (1 - (1 - rate)**(1/timestep))

def load_travel(path):
    """
    Load the number of people traveling between nodes (both implicit and explicit)
    :return:
    """
    travel = pd.read_csv(path, header=0)
    return travel

def load_contact(path):
    """
    Load the node-independent structured per-capita daily contact matrix
    :return:
    """
    contact = pd.read_csv(path, header=0)
    return contact

def load_population(path):
    """
    Load the node-specific population sizes
    :return:
    """
    population = pd.read_csv(path, header=0)
    return population

def contact_probability(n_i, n_j, n_i_total, n_k_total):

    # contacts between members of source node within the destination node
    fraction_i_to_j = n_i/n_i_total
    fraction_j_in_k = n_j/n_k_total
    pr_ii_in_j = fraction_i_to_j * fraction_j_in_k

    return pr_ii_in_j

def probabilistic_partition(travel_df, daily_timesteps):

    total_pop = travel_df.groupby(['source'])['n'].sum().to_dict()
    total_contextual_dest = travel_df[travel_df['destination_type'] == 'contextual'].groupby(['destination'])['n'].sum().to_dict()

    if len(set(total_pop.keys()).intersection(set(total_contextual_dest.keys()))) > 0:
        raise ValueError('Contextual nodes cannot also be source nodes.')
    if len(set(total_contextual_dest.keys()).intersection(set(total_pop.keys()))) > 0:
        raise ValueError('Contextual nodes cannot also be source nodes.')

    total_pop.update(total_contextual_dest)

    mapping = travel_df.groupby(['destination']).aggregate(lambda tdf: tdf.unique().tolist()).reset_index()
    mapping['source'] = [set(i) for i in mapping['source']]
    mapping['destination_type'] = [i[0] if len(i) == 1 else i for i in mapping['destination_type']]
    implicit2source = mapping[mapping['destination_type'] == 'contextual'][['source', 'destination']].set_index('destination').to_dict(orient='index')

    contact_dict = {
        'i': [],
        'j': [],
        'age_i': [],
        'age_j': [],
        'pr_contact_ij': []
    }

    # if it's local contact, or contact in contextual location within local pop only, it's straightforward
    for i, row in travel_df.iterrows():
        if row['destination_type'] == 'local':
            contact_dict['i'].append(row['source'])
            contact_dict['j'].append(row['destination'])
            contact_dict['age_i'].append(row['age_src'])
            contact_dict['age_j'].append(row['age_dest'])
            # if it's local within-node contact, the pr(contact) = n stay in node / n total in node (no need to multiply by another fraction)
            if row['source'] == row['destination']:
                daily_pr = contact_probability(n_i=row['n'], n_j=1, n_i_total=total_pop[row['source']], n_k_total=1)
                contact_dict['pr_contact_ij'].append(daily_pr)
            else:
                daily_pr = contact_probability(n_i=row['n'], n_j=row['n'], n_i_total=total_pop[row['source']], n_k_total=total_pop[row['destination']])
                contact_dict['pr_contact_ij'].append(daily_pr)

        # partitioning contacts between two different nodes within a contextual node requires a bit more parsing
        elif row['destination_type'] == 'contextual':
            other_sources = implicit2source[row['destination']]['source']
            for j in other_sources:
                contact_dict['i'].append(row['source'])
                contact_dict['j'].append(j)
                contact_dict['age_i'].append(row['age_src'])
                contact_dict['age_j'].append(row['age_dest'])
                j_to_dest = travel_df[(travel_df['source'] == j) \
                                      & (travel_df['destination'] == row['destination']) \
                                      & (travel_df['age_src'] == row['age_src']) \
                                      & (travel_df['age_dest'] == row['age_dest'])]['n'].item()
                daily_pr = contact_probability(n_i=row['n'], n_j=j_to_dest, n_i_total=total_pop[row['source']], n_k_total=total_pop[row['destination']])
                contact_dict['pr_contact_ij'].append(daily_pr)

    contact_df = pd.DataFrame.from_dict(contact_dict)
    contact_df = contact_df.groupby(['i', 'j', 'age_i', 'age_j'])['pr_contact_ij'].sum().reset_index()

    return contact_df

def partition_contacts(travel, contacts, daily_timesteps):

    tr_partitions = probabilistic_partition(travel, daily_timesteps)

    tc = pd.merge(tr_partitions, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2'])
    tc['interval_per_capita_contacts'] = tc['daily_per_capita_contacts'] / daily_timesteps
    tc['partitioned_per_capita_contacts'] = tc['pr_contact_ij'] * tc['interval_per_capita_contacts']

    recalc = tc.groupby(['age_i', 'age_j'])['partitioned_per_capita_contacts'].sum().reset_index()
    recalc = pd.merge(recalc, contacts, how='outer', left_on=['age_i', 'age_j'], right_on=['age1', 'age2']).dropna()
    # this check is broken
    for i, row in recalc.iterrows():
        try:
            assert row['partitioned_per_capita_contacts'] == row['daily_per_capita_contacts']
        except AssertionError:
            print('mismatched partitioned and baseline contacts')
            print(row)
    tc = tc.dropna()

    tc_final = tc[['i', 'j', 'age_i', 'age_j', 'partitioned_per_capita_contacts']]

    return tc_final

def contact_matrix(contact_df):

    sources = contact_df['i'].unique()
    destinations = contact_df['j'].unique()
    nodes = []
    for i in sources:
        nodes.append(i)
    for j in destinations:
        nodes.append(j)
    nodes = sorted(list(set(nodes)))

    ages = ['young', 'old']

    new_arr = np.zeros([len(nodes), len(nodes), len(ages), len(ages)])

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            for k, a1 in enumerate(ages):
                for l, a2 in enumerate(ages):
                    subset = contact_df[(contact_df['i'] == n1) \
                        & (contact_df['j'] == n2) \
                        & (contact_df['age_i'] == a1) \
                        & (contact_df['age_j'] == a2)]
                    if subset.empty:
                        val = 0
                    else:
                        val = subset['partitioned_per_capita_contacts'].item()
                    new_arr[i, j, k, l] = val

    return new_arr

def main():

    travel = load_travel('../inputs/travel3.csv')
    contacts = load_contact('../inputs/contact.csv')
    partition_df = partition_contacts(travel, contacts, daily_timesteps=10)
    partition_array = contact_matrix(partition_df)

if __name__ == '__main__':

    main()




