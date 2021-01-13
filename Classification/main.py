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
                'Lipides (g/100 g)','alim_ssgrp_code']
data_trié=data[selection_features]



pd.set_option('display.max_rows',data_trié.shape[0]+1)

#for column in data_trié:
#    print(data_trié[column].value_counts().sort_values())

data_trié=data_trié.dropna()
y=data_trié['alim_ssgrp_code']
for ligne in y:
    ligne=float(ligne)
data_trié=data_trié.drop(['alim_ssgrp_code'],axis=1)
print(data_trié)
print(y)

def model_KMeans(data_trié):
    #score=[]
    for i in range(3,4):
        model = KMeans(n_clusters=i)
        model.fit(data_trié)
        model.predict(data_trié)
        print(model.predict(data_trié))
        #plt.scatter(data_trié['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'],data_trié['Eau (g/100 g)'], c=model.predict(data_trié))
        #plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
        #score.append(model.score(data_trié))
    x_pca = model.transform(data_trié)
    for i in range(x_pca.shape[1]):
        plt.text(x_pca[i, 0], x_pca[i, 1], str(y[i]))
        plt.show()

model_KMeans(data_trié)




# model = KMeans(n_clusters=7)
# model.fit(data_trié)
# model.predict(data_trié)
# plt.scatter(data_trié['Energie, Règlement UE N° 1169/2011 (kcal/100 g)'],data_trié['Eau (g/100 g)'], c=model.predict(data_trié))
# plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
# plt.show()


## MODELISATION EN 2D d'un d'un espace à plusieurs dimensions grace à PCA
def model_PCA(data_trié):
    model = PCA(n_components=2)
    model.fit(data_trié)
    x_pca = model.transform(data_trié)
    plt.scatter(x_pca[:,0], x_pca[:,1], c=y)
    plt.show()
    return model

#model=model_PCA(data_trié)

#plt.plot(np.cumsum(model.explained_variance_ratio))
#plt.show()