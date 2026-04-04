import pandas as pd
import numpy as np

def generate_data(rows, cols):
    data = np.random.rand(rows, cols)
    df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(cols)])
    return df
