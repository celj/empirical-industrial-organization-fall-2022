import copy
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

choice_sum = individual.groupby('visitid')['choice'].sum().reset_index()
choice_sum.rename(columns={'choice': 'intercept'}, inplace=True)

individual = individual.merge(choice_sum, on='visitid', how='left')

individual = individual[['visitid',
                         'product',
                         'choice',
                         'intercept',
                         'size',
                         'speed',
                         'branded',
                         'price']]

users = list(individual['visitid'].unique())
products = list(individual['product'].unique())


def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):

    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres


def logll(x):
    choice = individual['visitid'].to_numpy()
    A = individual[['intercept', 'size',
                    'speed', 'branded', 'price']].to_numpy()
    P = np.divide(np.exp(A @ np.array([x[0], x[1], x[2], x[3], x[4]]).transpose(
    )), individual['visitid'].to_numpy() @ np.exp(A @ np.array([x[0], x[1], x[2], x[3], x[4]]).transpose()))

    f = -choice @ np.log(P)

    return f


print(nelder_mead(logll, np.array([0, 0, 0, 0, 0])))

print(products)
