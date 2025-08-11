import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack
plt.rcParams['axes.formatter.use_locale'] = True

def freqop(s, bf):
    df = pd.read_csv(sys.argv[1], sep =',', names=['t', 'x'], header=2)
    df = df.truncate(before=bf, after=500000)
    df['Média móvel'] = df['x'].rolling(s).mean()
    #df.drop(df.index[0:s], axis=0, inplace=True)
    dft = df['t']
    dfm = pd.merge(dft, df['Média móvel'].dropna(), right_index=True, left_index=True)
    #arq = sys.argv[1].replace('.csv','') + 'Filtrado.csv'
    #dfm.to_csv(arq, index=False)
   # print(dfm)
#pd.display(df)

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
    print(FFT[:N_fft//2][18])
    f_max = n_max*(freqs[1]-freqs[0])
    print ("f_max = ", f_max, freqs[1]-freqs[0])
    ax.set_yscale('log')
    ax.set_xticks(np.arange(0., 10., 1.))
#ax.set_title('Transformada de Fourier',fontsize=14)
    ax.set_xlim(0., 30.)
    ax.set_ylim([10**-6, 10**1])
    ax.plot(freqs[:N_fft//2], FFT[:N_fft//2], 'b-', linewidth = 0.5)
    ax.grid()
    x = np.linspace(-5.15,-4.7,1000)
    y = (0.0216938+0.488275)*np.exp(-0.223449*(x))*np.cos(2*np.pi*26*(x)+0.4)
    plt.figure()
    plt.plot(dfm['t'], dfm['Média móvel'], mec='black', ms=1, label = 'Dados não filtrados')
    plt.plot(x,y)
    plt.xlabel('$t (s)$')
    plt.ylabel(r'$V(t)$')
    plt.legend()
    plt.grid()
    return f_max
    print(bf)
freqop(414, 483000)
#a = np.array([0])
#for bf in range(495000,480000,-1000):
#    a = np.append(a, freqop(1,bf))
#print(np.max(a))
#print(np.where(a == np.max(a)))
plt.show()

