import pandas as pd
from sklearn.cluster import KMeans
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import unique
import math
from sklearn.decomposition import PCA

list=[]
for i in range(0,1000,1):
    list.append("< " + str(i / 1000000).replace('.', ','))
    list.append("< " + str(i / 100000).replace('.', ','))
    list.append("< " + str(i / 10000).replace('.', ','))
    list.append ("< "+str(i/1000).replace('.',','))
    list.append("< " + str(i / 100).replace('.', ','))
    list.append("< " + str(i / 10).replace('.', ','))
    list.append("< " + str(i).replace('.', ','))


list_na_values=['-','','traces']
for i in list_na_values:
    list.append(i)

data=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',
                   keep_default_na=False,
                    na_values=list_na_values,
                   decimal=",")




#plt.imshow(data.isna())
#plt.show()

features=['Energie, Règlement UE N° 1169/2011 (kJ/100 g)','Energie, Règlement UE N° 1169/2011 (kcal/100 g)',
          'Energie, N x facteur Jones, avec fibres  (kJ/100 g)',
          'Energie, N x facteur Jones, avec fibres  (kcal/100 g)',
          'Eau (g/100 g)','Protéines, N x facteur de Jones (g/100 g)',
          'Protéines, N x 6.25 (g/100 g)','Glucides (g/100 g)','Lipides (g/100 g)',
          'Sucres (g/100 g)','Fructose (g/100 g)','Galactose (g/100 g)',
          'Glucose (g/100 g)','Lactose (g/100 g)','Maltose (g/100 g)',
          'Saccharose (g/100 g)','Amidon (g/100 g)','Fibres alimentaires (g/100 g)',
          'Polyols totaux (g/100 g)','Cendres (g/100 g)','Alcool (g/100 g)',
          'Acides organiques (g/100 g)','AG saturés (g/100 g)','AG monoinsaturés (g/100 g)',
          'AG polyinsaturés (g/100 g)','AG 4:0, butyrique (g/100 g)',
          'AG 6:0, caproïque (g/100 g)','AG 8:0, caprylique (g/100 g)',
          'AG 10:0, caprique (g/100 g)','AG 12:0, laurique (g/100 g)',
          'AG 14:0, myristique (g/100 g)','AG 16:0, palmitique (g/100 g)',
          'AG 18:0, stéarique (g/100 g)','AG 18:1 9c (n-9), oléique (g/100 g)',
          'AG 18:2 9c,12c (n-6), linoléique (g/100 g)',
          'AG 18:3 c9,c12,c15 (n-3), alpha-linolénique (g/100 g)',
          'AG 20:4 5c,8c,11c,14c (n-6), arachidonique (g/100 g)',
          'AG 20:5 5c,8c,11c,14c,17c (n-3) EPA (g/100 g)',
          'AG 22:6 4c,7c,10c,13c,16c,19c (n-3) DHA (g/100 g)',
          'Cholestérol (mg/100 g)','Sel chlorure de sodium (g/100 g)',
          'Calcium (mg/100 g)','Chlorure (mg/100 g)','Cuivre (mg/100 g)',
          'Fer (mg/100 g)','Iode (µg/100 g)','Magnésium (mg/100 g)','Manganèse (mg/100 g)',
          'Phosphore (mg/100 g)','Potassium (mg/100 g)','Sélénium (µg/100 g)',
          'Sodium (mg/100 g)','Zinc (mg/100 g)','Rétinol (µg/100 g)',
          'Beta-Carotène (µg/100 g)','Vitamine D (µg/100 g)','Vitamine E (mg/100 g)',
          'Vitamine K1 (µg/100 g)','Vitamine K2 (µg/100 g)','Vitamine C (mg/100 g)',
          'Vitamine B1 ou Thiamine (mg/100 g)','Vitamine B2 ou Riboflavine (mg/100 g)',
          'Vitamine B3 ou PP ou Niacine (mg/100 g)',
          'Vitamine B5 ou Acide pantothénique (mg/100 g)','Vitamine B6 (mg/100 g)',
          'Vitamine B9 ou Folates totaux (µg/100 g)','Vitamine B12 (µg/100 g)']

#selection des data

for column in features:
    for k in data[column]:
        if "<" in str(k) and k not in list_na_values :
            list_na_values.append(k)
print(list_na_values)

data=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',
                   keep_default_na=False,
                    na_values=list_na_values,
                   decimal=",")

#selection des features

def infos(column):
    list=[]
    for k in column:
        if not math.isnan(float(k)):
            list.append(float(k))
    list.sort()
    list=np.array(list)
    list=np.unique(list)
    return np.var(list)/np.mean(list),(np.var(list)),np.mean(list),len(list),\
           (list[1],list[2],list[3])


list_na_values_min=['-','','traces']
data_min=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',
                   keep_default_na=False,
                    na_values=list_na_values_min,
                   decimal=",")

for column in features:
    list=[]
    a,b,c,d,e=infos(data[column])
    for k in data_min[column]:
        if k in list_na_values:
            list.append(str(k)+" apparaît ")
    print(column)
    list = pd.Series(list)
    print(list.value_counts())
    print("les minimums sont "+str(e)+" et la moyenne -2x la variance est de "+str(c-(2*b))+"\n")





for column in features:
    print(column + str(infos(data[column])))



selection_features=[]
for k in features:
    ratio_var_mean,var,mean,samples=infos(data[k])
    if ratio_var_mean>1 and samples>2000:
        selection_features.append(k)
        print(k)

print(selection_features)



data_trié=data[selection_features]
label=data[['alim_nom_fr','alim_grp_nom_fr']]

pd.set_option('display.max_rows',data_trié.shape[0]+1)

#for column in data_trié:
#    print(data_trié[column].value_counts().sort_values())

data_trié=data_trié.dropna()

print(data_trié)

inertia=[]
i_range=range(8,9)
for i in i_range:
    model=KMeans(n_clusters=i)
    model.fit(data_trié)
    prediction=model.predict(data_trié)
    for k in range(len(prediction)):
        print(str(label['alim_nom_fr'][k])+" est un/une "+str(label['alim_grp_nom_fr'][k])+" classé dans le cluster "+str(prediction[k]))
    print(model.cluster_centers_)
    inertia.append(model.inertia_)
plt.plot(i_range,inertia)
plt.show()


# inertia_aglo=[]
# K_range=range(1,10)
# for k in K_range:
#     model_aglo=sk.cluster.AgglomerativeClustering(n_clusters=k)
#     model_aglo.fit(data_trié)
#     inertia_aglo.append(model_aglo.distances_)
# plt.plot(K_range,inertia_aglo)
#
# plt.show()



