import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',keep_default_na=False, na_values=['-','','None','< 0,03'],decimal=",")
data_trié=data[['Energie, Règlement UE N° 1169/2011 (kcal/100 g)',
                'Eau (g/100 g)']],'Protéines, N x facteur de Jones (g/100 g)','Glucides (g/100 g)',
                'Lipides (g/100 g)']]


#for column in data_trié:
 #   print(data_trié[column].value_counts())

data_trié=data_trié.dropna()
pd.set_option('display.max_rows', data_trié.shape[0]+1)

#plt.scatter(data_trié['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'],data_trié['Eau (g/100 g)'])
#plt.show()

score=[]
for i in range(3,20):
    model = KMeans(n_clusters=i)
    model.fit(data_trié)
    model.predict(data_trié)
    #plt.scatter(data_trié['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'],data_trié['Eau (g/100 g)'], c=model.predict(data_trié))
    #plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
    score.append(model.score(data_trié))

#plt.plot(np.linspace(3,19,17),score)
#plt.show()

model = KMeans(n_clusters=7)
model.fit(data_trié)
model.predict(data_trié)
plt.scatter(data_trié['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'],data_trié['Eau (g/100 g)'], c=model.predict(data_trié))
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
plt.show()