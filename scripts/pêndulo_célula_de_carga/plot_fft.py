import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack
plt.rcParams['axes.formatter.use_locale'] = True

df = pd.read_csv(sys.argv[1], sep =',', names=['t', 'x'], header=2)
#pd.display(df)

plt.figure()
plt.plot(df['t'], df['x'], mec='black', ms=1, label = 'Dados não filtrados')
plt.xlabel('$t (s)$')
plt.ylabel(r'$V(t)$')
plt.legend()
plt.grid()
#print(df)
#figura = sys.argv[1].replace('.csv', '.pdf')
#plt.savefig(figura)
#print(figura)
# Transformada de Fourier
def FFT(signal, time):
    N_fft = len(signal)
    fft = 2*np.abs(scipy.fftpack.fft(signal))/N_fft
    dt = time[1]-time[0]
    freqs = scipy.fftpack.fftfreq(signal.size, dt)
    return [freqs, fft]
#####################################################################
# taxa de amostragem para fft
sampRate = 1
time = np.array(df['t'])
signal = np.array(df['x'])
time = time[::sampRate]
signal = signal[::sampRate]
# plotar serie temporal
fig, ax = plt.subplots(1, figsize=(8, 8))
# Transformada de Fourier
freqs, FFT = FFT(signal, time)
N_fft = len(freqs)
fft_max = np.amax(FFT[:N_fft//2])
n_max = np.argmax(FFT[:N_fft//2])
f_max = n_max*(freqs[1]-freqs[0])
print ("f_max = ", f_max, freqs[1]-freqs[0])
ax.set_yscale('log')
ax.set_xticks(np.arange(0., 10., 1.))
#ax.set_title('Transformada de Fourier',fontsize=14)
ax.set_xlim(0., 10.)
ax.set_ylim([10**-6, 10**0])
ax.plot(freqs[:N_fft//2], FFT[:N_fft//2], 'b-', linewidth = 0.5)
ax.grid()
plt.show()

