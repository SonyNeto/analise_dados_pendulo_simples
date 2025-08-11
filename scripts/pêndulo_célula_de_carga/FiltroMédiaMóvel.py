import sys
import numpy as np
import pandas as pd

s = int(sys.argv[2])
df = pd.read_csv(sys.argv[1], sep =',', names=['t', 'theta'], header=2)
df['Média móvel'] = df['theta'].rolling(s).mean()
dft = df['t']
dfm = pd.merge(dft, df['Média móvel'].dropna(), right_index=True, left_index=True)
print(df['Média móvel'])
print(dfm)
arq = sys.argv[1].replace('.csv','') + 'Filtrado.csv'
dfm.to_csv(arq, index=False)
