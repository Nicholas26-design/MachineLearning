import pandas as pd
import numpy as np

def get_dataset(size):
    df = pd.DataFrame()
    df['position'] = np.random.choice(['offense', 'center', 'defense'], size)
    df['age'] = np.random.randint(1, 50, size)
    df['team'] = np.random.choice(['red', 'blue', 'yellow'], size)
    df['win'] = np.random.choice(['yes', 'no'], size)
    df['prob'] = np.random.uniform(0, 1, size)
    return df

df = get_dataset(1_000_000)
df.info

# https://www.youtube.com/watch?v=u4_c2LDi4b8
