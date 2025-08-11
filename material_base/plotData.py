import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("2022-04-03_08h07m_SerieTempPend.csv", sep =',', names=['t', 'theta'], header=None)
#pd.display(df)
plt.figure()
plt.plot(df['t'], df['theta'], 'r-o', mfc='w', mec='r', ms=3)
plt.xlabel('$t$')
plt.ylabel(r'$\theta$')
plt.xlim([-0.5,6.12])
plt.grid()
plt.show()
