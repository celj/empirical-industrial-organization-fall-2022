from data.aggregate import aggregate
from data.aggregate import individual
from models.IV2SLS import IV2SLS
from models.OLS import OLS
# import pandas as pd
# import numpy as np

# print(aggregate[['price', 'log_price']])

# print(aggregate[['y', 'constant', 'price', 'size', 'speed', 'branded']].describe())

# print(individual[individual['visitid'] == 1]['choice'].sum())

aggregate[['product', 'share']].to_clipboard()

OLS(aggregate[['price', 'size', 'speed', 'branded']],
    aggregate[['y']],
    results=False)

IV2SLS(aggregate[['size', 'speed', 'branded', 'log_price']],
       aggregate[['y']],
       aggregate[['size', 'speed', 'branded', 'wholecost']],
       results=False)

IV2SLS(aggregate[['size', 'speed', 'branded', 'log_price']],
       aggregate[['y']],
       aggregate[['size', 'speed', 'branded', 'log_wholecost']],
       results=False)
