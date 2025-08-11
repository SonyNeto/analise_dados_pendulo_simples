import sys
import numpy as np
import pandas as pd

#s = int(sys.argv[2])
df = pd.read_csv(sys.argv[1])
df = df.truncate(before=483000, after=500000)
#df.drop(df.index[0:125000], axis=0, inplace=True)
#df['Média móvel'] = df['theta'].rolling(s).mean()
#dft = df['t']
#dfm = pd.merge(dft, df['Média móvel'].dropna(), right_index=True, left_index=True)
#print(df['Média móvel'])
print(df)
arq = 'frequencianatural.csv'
df.to_csv(arq, index=False)
