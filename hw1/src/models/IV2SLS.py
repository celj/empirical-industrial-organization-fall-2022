import numpy as np
import pandas as pd


def IV2SLS(x, y, z, results=True):
    '''
    x: explanatory variables (dataframe)
    y: response variables (dataframe)
    z: instrumental variables (dataframe)
    '''

    print('2SLS \n')

    parameters = list(x.columns)

    x = x.to_numpy()
    y = y.to_numpy()
    z = z.to_numpy()

    n = y.size
    k = len(parameters)
    y_ = np.mean(y)

    x = np.concatenate([np.ones((n, 1)), x], axis=1)
    z = np.concatenate([np.ones((n, 1)), z], axis=1)

    w = z @ np.linalg.inv(z.transpose() @ z) @ z.transpose()

    x_hat = w @ x
    beta_hat = np.linalg.inv(x.transpose() @ w @ x) @ x.transpose() @ w @ y
    y_hat = x_hat @ beta_hat
    err = y - y_hat

    s_sq = (err.transpose() @ err) / (n - k - 1)
    se = np.sqrt(s_sq)[0, 0]
    se_beta = np.sqrt(np.diag(s_sq * np.linalg.inv(x.transpose() @ x)))
    r_sq = 1 - (np.sum(err**2) / np.sum((y - y_)**2))

    parameters.insert(0, '(intercept)')
    beta_hat = beta_hat[:, 0]

    if results == False:
        output = {'': parameters, 'Estimate': beta_hat, 'Std. Error': se_beta}
        output = pd.DataFrame.from_dict(output)

        print('Standard Error:', se, '\n')
        print('R Squared', r_sq, '\n')
        print(output.to_string(index=False), '\n')

    else:
        output = {'estimations': beta_hat, 'residuals': err}

        return output
