import os
import pandas as pd
import numpy as np

colnames = ['visitid',
            'size',
            'speed',
            'branded',
            'price',
            'outofstock',
            'wholecost',
            'choice']

individual = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../data/raw.csv'),
                         names=colnames,
                         header=None)

# remove unseen products
individual = individual[individual['outofstock'] == 0]

individual['product'] = individual['size'].astype(str) + \
    individual['speed'].astype(str) + \
    individual['branded'].astype(str)

individual['product'] = pd.to_numeric(individual['product'])

ppl = individual['visitid'].nunique()

aggregate = individual[individual['choice'] == 1] \
    .groupby('product') \
    .agg(people=('visitid', 'nunique'),
         size=('size', 'mean'),
         speed=('speed', 'mean'),
         branded=('branded', 'mean'),
         price=('price', 'mean'),
         wholecost=('wholecost', 'mean')) \
    .reset_index()

ppl_0 = round((1 - aggregate['people'].sum() / ppl) * ppl)

aggregate.loc[-1] = [0, ppl_0, 0, 0, 0, 0, 0]

aggregate.index = aggregate.index + 1

aggregate = aggregate.sort_index()

aggregate['share'] = aggregate['people'] / aggregate['people'].sum()

aggregate['log_share'] = np.log(aggregate['share'])

log_share_0 = aggregate[aggregate['product'] ==
                        0]['log_share'][0]  # outside good market share

aggregate['y'] = aggregate['log_share'] - log_share_0  # estimation target

aggregate.drop([0], axis=0, inplace=True)

aggregate['log_price'] = np.log(aggregate['price'])

aggregate['log_wholecost'] = np.log(aggregate['wholecost'])

individual.loc[individual['choice'] == 0, 'product'] == 0

individual['intercept'] = 1
