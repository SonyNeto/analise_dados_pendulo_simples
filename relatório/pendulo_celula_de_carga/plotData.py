###CÓDIGO DO FILTRO DE MÉDIA MÓVEL E PLOTAGEM DOS GRÁFICOS###
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

##Função para o Algoritmo do melhor valor de n da Janela do Filtro Móvel
#def freqop(s):
s = int(sys.argv[2]) #Valor n da Janela do Filtro Móvel
df = pd.read_csv(sys.argv[1], sep =',', names=['t', 'theta'], header=2) #Leitura dos Dados para DataFrame
data = (pd.merge(df['t'], df['theta'].rolling(s).mean(), right_index=True, left_index=True)).dropna()#Filtro Móvel

datat = data['t'].to_numpy()#Conversão do DataFrame de Dados de Tempo para um Array

datam = data['theta'].to_numpy()#Conversão do DataFrame de Dados de Sinais para um Array
datam1 = (-data['theta']-(np.max(datam) - np.min(datam))).to_numpy()#Conversão dos Dados de Sinais Espelhados para um Array 

p = scipy.signal.find_peaks(datam, distance=1500)[0] #Achando os máximos
p1 = scipy.signal.find_peaks(datam1, distance=1500)[0] #Achando os mínimos

dt = pd.DataFrame(datat[p], columns=['t'])#Conversão de volta dos tempos de máximos para um DataFrame
dt1 = pd.DataFrame(datat[p1], columns=['t'])#Conversão de volta dos tempos de mínimos para um DataFrame

d = pd.DataFrame(datam[p], columns=['peaks'])#Conversão de volta dos máximos para um DataFrame
d1 = pd.DataFrame(datam1[p1], columns=['peaks'])#Conversão de volta dos mínimos um DataFrame
dtm = pd.merge(dt, d, right_index=True, left_index=True)#DataFrame dos Dados de máximos
dtm1 = pd.merge(dt1, d1, right_index=True, left_index=True)#DataFrame dos Dados de mínimos

dtmf = pd.concat([dtm, dtm1]).sort_values(by=['t','peaks'], ascending=[1,1])#Concatenação dos Dados de máximos e mínimos
#print(dtmf)
## Cálculo da Frequência de oscilação do pêndulo
n = 1
l = 0
while n + 2 < len(p):
   l_n = abs(datat[p1[n]] - datat[p[n]])
   l = l+l_n
   n = n+1
  # print(l)
periodo = 2*l/(n-2)
frequencia = float(1/periodo)
freq_s=frequencia/2
  # return freq_s
#print(freq)
  
## Cálculo do melhor valor de n
#a = np.array([0])
#for s in range(1,1024):
 #  a = np.append(a, freqop(s))
#print(np.max(a))
#print(np.where(a == np.max(a)))
print('A frequência é: ', freq_s)

#Exportação dos dados de máximos e mínimos
arq = sys.argv[1].replace('.csv','') + ' Pontos máximos e mínimos.csv'
dtmf.to_csv(arq, index=False)

#def func(t,a,b,c):
   # return a*np.sin(b*t+c)
#x = data['t'].to_numpy
#y = data['theta'].to_numpy
#scipy.optimize.curve_fit(func, x, y)

#pd.display(df)

#x = np.linspace(-10,10,1000)
#y = 0.009*np.cos(2*np.pi*1.19*x + 1.5) + 0.009  - 0.006*np.cos(2*np.pi*0.06*x + 2) - 0.005

#Plotagem dos gráficos
plt.figure()
plt.scatter(dtmf['t'], dtmf['peaks'], c='green', label='Máximos e Mínimos')
plt.plot(dtmf['t'], dtmf['peaks'])
#plt.title('Frequência de oscilação do pêndulo = %f Hz'%freq)
#plt.plot(df['t'], df['theta'], 'r-o', mfc='w', mec='r', ms=3, label='Dados filtrados')
#plt.plot(x, y, 'r-o', mfc='w', mec='g', ms=1)
plt.plot(data['t'], data['theta'],'r-o', mfc='w', mec='r', ms=3, label='Dados filtrados')
plt.xlabel('$t (s)$')
plt.ylabel(r'$V(t)$')
plt.grid()
plt.legend()
plt.show()
