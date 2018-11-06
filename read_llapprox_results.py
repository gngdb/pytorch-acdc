import json
import math
import numpy as np

if __name__ == '__main__':
    with open('linear_layer_approx_results.json', 'r') as f:
        results = json.load(f)

    # Results formatting: dictionary:
    #   Key: string "<layer_type>_<options>"
    #   Value: [(M, N, no. parameters, final MSE), ...]
    # list contains replicates

    for k in results:
        mse_vals = [v[3] for v in results[k]]
        n_params = results[k][0][2]
        mu = np.mean(mse_vals)
        sigma = math.sqrt(np.var(mse_vals))
        print(f"{k} with {n_params} parameters: {mu} +/- {sigma}")
