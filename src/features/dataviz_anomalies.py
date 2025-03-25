import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

meteobydate = pd.read_csv('data/processed/meteo_pivot_cleaned_2010-2024_0.1.csv', sep = ';', parse_dates = ['datemesure'])

parametres = ['ETP', 'GLOT', 'RR', 'TN', 'TX']

path    = './data/reports/figures'

std_threshold = {
    param: meteobydate[f"{param}_origine"].std() * 0.4 for param in parametres
}  # 42% of std deviation
print(std_threshold)

std_threshold_min = {
    param: meteobydate[f"{param}_origine"].std() * 0.1 for param in parametres
}  # 10% of std deviation

for param in parametres:
    differences = (meteobydate[param + '_origine'] - meteobydate[param])
    # on retire toutes les valeurs qui n'ont pas été corrigées
    differences = np.abs(differences[differences !=0])
    plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1,2, 1)
    ax1.hist(differences, bins = 4, label = param)
    ax1.set_xlabel(f"{param} : Différences absolues entre valeur brute et corrigée")
    ax1.set_ylabel("Effectif")
    #plt.show()
    bins = meteobydate[f"{param}_origine"].std() * np.arange(0.0, 1.1, 0.1)
    ax2 = plt.subplot(1,2, 2)
    ax2.hist(differences, bins = bins, label = param)
    ax2.axvline(std_threshold[param], ymax = ax2.get_ylim()[1] * 0.9, linestyle = ':', color = 'red')
    ax2.text(std_threshold[param], ax2.get_ylim()[1] * 0.9, 'Seuil anomalie 40% std', size = 11, color = 'red')
    ax2.axvline(std_threshold_min[param], ymax = ax2.get_ylim()[1] * 0.9, linestyle = ':', color = 'green')
    ax2.text(std_threshold_min[param], ax2.get_ylim()[1] * 0.9, 'Seuil anomalie minimal', size = 11, color = 'green')
    
    ax2.set_xlabel(f"{param} : Différences absolues entre valeur brute et corrigée")
    ax2.set_xticks(bins)
    ax2.set_ylabel("Effectif")
    
    plt.savefig(f"{path}/{param}_differences.png")
    #plt.show()
    
# fréquence des anomalies
anomaly_day = meteobydate.groupby('datemesure')['anomaly'].agg(['sum', 'count'])
anomaly_day['rate'] = anomaly_day['sum'] / anomaly_day['count']
plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.plot(anomaly_day.index, anomaly_day['sum'], label = 'sum')
ax1.legend()
ax2 = plt.subplot(2, 1, 2)
ax2.plot(anomaly_day.index, anomaly_day['rate'], label = 'rate')
ax2.legend()
plt.savefig(f"{path}/anomalies_time.png")
