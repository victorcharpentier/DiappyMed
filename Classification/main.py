import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


data=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',keep_default_na=False,
                   na_values=['-','','None','< 0,5','< 0,6','< 0,022','< 0,3','<0,3','< 0,1','< 0,03','< 0,85','< 0,55',
                              '< 0,46', '< 0,57','< 0,42','< 0,05','< 0,2','traces'],decimal=",")
selection_features=['Energie, Règlement UE N° 1169/2011 (kcal/100 g)',
                'Eau (g/100 g)','Protéines, N x facteur de Jones (g/100 g)','Glucides (g/100 g)',
                'Lipides (g/100 g)']
data_trié=data[selection_features]
comparaison=data['alim_grp_code']



pd.set_option('display.max_rows',data_trié.shape[0]+1)

#for column in data_trié:
#    print(data_trié[column].value_counts().sort_values())

data_trié=data_trié.dropna()

print(data_trié)

inertia=[]
K_range=range(9,10)
for k in K_range:
    model=KMeans(n_clusters=k)
    model.fit(data_trié)
    prediction=model.predict(data_trié)
    print(model.cluster_centers_)
    #liste_prediction=[]
    #for k in range(prediction.shape[0]):
    #    liste_prediction.append(((prediction[k]*10)+comparaison[k]))
    #liste_prediction = pd.DataFrame(liste_prediction,columns=['prediction'])
    #print(liste_prediction['prediction'].value_counts())
    inertia.append(model.inertia_)
plt.plot(K_range,inertia)
plt.xlabel("KMeans")
plt.show()






