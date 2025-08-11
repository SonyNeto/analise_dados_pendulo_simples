# -*- coding: utf-8 -*-
#Locale settings
import locale, sys
# Set to pt_BR to get comma decimal separater
locale.setlocale(locale.LC_NUMERIC, 'pt_BR.utf8')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from scipy.optimize import least_squares
import scipy.fftpack
import pandas as pd
plt.rcParams['axes.formatter.use_locale'] = True
#####################################################################
# Leitura de dados
def lerDados(arquivo):
    dados = arquivo
    dataset = np.loadtxt(dados, delimiter=',', skiprows=2)
    signal = dataset [:, 1]
    time = dataset[:,0]
    time = time - time[0]
    return[time, signal]

nPer = 1704 #1491
nPoints = 180*nPer
time, signal = lerDados(sys.argv[1])
time = time[-nPoints:]
time = time-time[0]
signal = signal[-nPoints:]
avg = np.average(signal[4*int(len(signal)/5):])
# Plota grafico
# plot data
fig=plt.figure(figsize=(8, 10))
plt.subplots_adjust(hspace=0.35)
ax1=fig.add_subplot(211)
ax1.plot(time, signal-avg, 'b-.', linewidth=0.2, markersize=0.2, label= "Dados experimentais")
#legenda = u"$f(t)=$ $%g+%g e^{-%g t}$" % (C, A,G)
t_max = time[-1]
y_max = np.amax(signal-avg)
#ax1.axhline(y=y_max, ls='--')
ax1.set_xlim(0, t_max)
ax1.set_ylim(-1.1*y_max, 1.1*y_max)
ax1.set_xlabel("Tempo $t$, $s$", fontsize=12)
ax1.set_ylabel(" Deslocamento $x(t)$, $mm$", fontsize=12)
ax1.text(0.2, 1.3, '(a)')

ax1.grid()

# second subplot: the Fourier transform
#####################################################################
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
time = time[::sampRate]
signal = signal[::sampRate]
# plotar serie temporal
ax2=fig.add_subplot(212)
# Transformada de Fourier
freqs, FFT = FFT(signal, time)
N_fft = len(freqs)
fft_max = np.amax(FFT[:N_fft//2])
n_max = np.argmax(FFT[:N_fft//2])
print (freqs[1])
f_max = n_max*(freqs[1]-freqs[0])
print ("f_max = ", f_max, freqs[1]-freqs[0])
ax2.set_yscale('log')
ax2.set_xticks(np.arange(0., 150., 10.))
ax2.set_title(u'Transformada de Fourier',fontsize=14)
ax2.set_xlim(0., 150.)
ax2.set_ylim([10**-6, 10**0])
ax2.plot(freqs[:N_fft//2], FFT[:N_fft//2], 'b-', linewidth = 0.5)
ax2.grid()
#plt.legend(numpoints = '1', loc = "upper left")
ax2.set_xlabel(u"Frequência $f$, $Hz$", fontsize=12)
ax2.set_ylabel(r"$|\tilde x(f)|$ ", fontsize=12)
ax1.text(-0.12, 1.05, '(a)', transform=ax1.transAxes, size=12, weight='bold')
ax2.text(-0.12, 1.05, '(b)', transform=ax2.transAxes, size=12, weight='bold')
# inset of the peak
# previsão teórica
f_0 = f_max
#f_0 = 10.5832
om_0 = 2*np.pi*f_0
freqInt = np.linspace(0.90*f_0, 1.1*f_0, 1024)
om = 2*np.pi*freqInt
# sub region of the original image
x1, x2, y1, y2 = 0.98*f_0, 1.08*f_0, 0.01*fft_max, 1.8*fft_max
#x1, x2, y1, y2 = 10.45, 10.69, 0.01, 0.33
axins = zoomed_inset_axes(ax2, zoom=4, loc=4) # zoom = 6
ip = InsetPosition(ax2, [.4, .65, .5, .3])
axins.set_axes_locator(ip)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
for axis in ['top','bottom','left','right']:
    axins.spines[axis].set_linewidth(1)
    axins.spines[axis].set_color('g')
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", lw=1, ec="g")
xdata = freqs[n_max-10:n_max+10] 
ydata = FFT[n_max-10:n_max+10]
print('n_max',n_max)
#####################################################################
def yMod(gamma, om):
  ganho = 1.0/np.sqrt((om**2-om_0**2)**2+(gamma*om)**2)
  ganho_max = np.amax(ganho)
  coef = fft_max/ganho_max
  ganho *= coef
  return ganho
#####################################################################
def dyMod(gamma, om):
  dy = -gamma*om*om/np.cbrt((om**2-om_0**2)**2+(gamma*om)**2)
  return dy
#####################################################################
def residual(gamma):
    yM = yMod(gamma, 2*np.pi*xdata)
    dif = np.sum((yM-ydata)**2)
    return dif
#####################################################################
def dres(gamma):
    yM = yMod(gamma, 2*np.pi*xdata)
    dy = dyMod(gamma, 2*np.pi*xdata)
    dif = np.sum((yM-ydata)*dy)
    return dif
#####################################################################
#axins.plot(xdata, yMod(xdata, *gammaFit), 'r-', label = u"Teórico", linewidth = 1)
axins.plot(freqs[1:N_fft//2], FFT[1:N_fft//2], 'bo', label = "Experimental",\
        linewidth = 1.5, ms = 2.5)
ga1 = 0.1
ga2 = 1.0
f1 = dres(ga1)
print(f1)
f2 = dres(ga2)
print(f2)
if f1*f2 < 0:
  rt=scipy.optimize.bisect (dres, ga1, ga2)
  ga_fit = rt
  print('f_0', f_0, 'ga_fit', ga_fit)
ganho = yMod(ga_fit, om)
legenda = 'Teórico $f_0=%.5g$Hz\n $\gamma=%.4g$Hz' % (f_0, ga_fit)
axins.plot(freqInt, ganho, 'r-', label = legenda, linewidth = 1)
axins.legend(fontsize=8, loc = "upper right")
X = time
dt= time[1]-time[0]
T = 1.0/f_0
nPer = int(T/dt)
print('nPer', nPer)
C = 0.3*np.amax(signal[-nPer:])
#ax1.axvline(time[-nPer])
#C = np.amax(signal[15*int(len(signal)/16):])
x = np.linspace(0,12,17000)
y = (0.0216938+0.488275)*np.exp(-0.223449*x)*np.cos(2*np.pi*26.472*x+1.84)
#ax1.plot(X, y, c='green')
envelope = C+(y_max-C)*np.exp(-ga_fit*(X-X[0])/2)
legenda = u"$f(t)=$ $%g+%g e^{-%g t}$" % (C, y_max-C, ga_fit/2)
ax1.plot(X, envelope, 'r-', label = legenda)
ax1.plot(X, -envelope,'r-')
ax1.legend(loc="upper right")
Q = 2*np.pi*f_0/ga_fit
print('Q', Q)
ax1.set_title(u'Série temporal, $Q=%.4g$' %(Q),fontsize=14)
figura = sys.argv[1].replace('.csv', '')
figura = figura+'f_0%.5gga%.4g.pdf' %(f_0, ga_fit)
print(figura)
plt.savefig(figura)
plt.show()
