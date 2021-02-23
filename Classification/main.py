import pandas as pd
from sklearn.cluster import KMeans
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import unique
import math
from sklearn.decomposition import PCA
import statistics

# list=[]
# for i in range(0,1000,1):
#     list.append("< " + str(i / 1000000).replace('.', ','))
#     list.append("< " + str(i / 100000).replace('.', ','))
#     list.append("< " + str(i / 10000).replace('.', ','))
#     list.append("< " +str(i/1000).replace('.',','))
#     list.append("< " + str(i / 100).replace('.', ','))
#     list.append("< " + str(i / 10).replace('.', ','))
#     list.append("< " + str(i).replace('.', ','))


list_na_values=['-','','traces']
#for i in list_na_values:
#    list.append(i)

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

#data['Eau (g/100 g)'].str.strip('< ').astype(float)

for column in features:
    for k in data[column]:
         if "< " in str(k):
             k=k[2:]
             for i in range(len(k)):
                 if k[i]==',':
                    k=k[0:i]+'.'+k[i+1:len(k)+1]
             k = float(k)/2
             print(k)
         # if "<" in str(k):
         #     k=k[1:]
         #     for i in range(len(k)):
         #         if k[i]==',':
         #            k=k[0:i]+'.'+k[i+1:len(k)+1]
         #     k=float(k)/2
         #     k = str(k)
         #     for i in range(len(k)):
         #         if k[i]==',':
         #            k=k[0:i]+'.'+k[i+1:len(k)+1]
         #     print(k)


         if " " in str(k):
             k = k[1:]


#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#    print(data)



#on tranforme les "<x" par x/2


print(list_na_values)

# data=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',
#                    keep_default_na=False,
#                     na_values=list_na_values,
#                    decimal=",")

#selection des features

def infos(column):
    list=[]
    for k in column:
        if not math.isnan(float(k)):
            list.append(float(k))
    moyenne=statistics.mean(list)
    variance=statistics.variance(list)
    ecart_type=math.sqrt(variance)
    return ecart_type/moyenne,ecart_type,moyenne,len(list),\
           list


# list_na_values_min=['-','','traces']
# data_min=pd.read_excel('C:\\Users\\charp\\Downloads\\Table Ciqual 2020_FR_2020 07 07.xls',
#                    keep_default_na=False,
#                     na_values=list_na_values_min,
#                    decimal=",")

# def les_x_minimums(list,x):
#     list_1=[]
#     list_2=[]
#     for i in list:
#         if i not in list_1:
#             list_1.append(i)
#     list_1.sort()
#     if x>len(list_1):
#         for k in range(len(list_1)):
#             list_2.append((list_1[k], list.count(list_1[k])))
#     else:
#         for k in range(x):
#             list_2.append((list_1[k],list.count(list_1[k])))
#     return list_2

#print(les_x_minimums([1,4,8,5,4,2,3,5,8,9,4,5,1,2,3,5,8],2))



# for column in features:
#     list=[]
#     ratio_ecart_type_moyenne, ecart_type, moyenne, nb_individus,list_min=infos(data[column])
#     for k in data_min[column]:
#         if k in list_na_values:
#             list.append(str(k)+" apparaît ")
#     print(column)
#     list=pd.Series(list)
#     print(list.value_counts())
#     print("les minimums sont ")
#     print(les_x_minimums(list_min,10))
#     print("la moyenne est de "+str(moyenne))





for column in features:
    print(column + str(infos(data[column])))



selection_features=[]
for k in features:
    ratio_var_mean,var,mean,samples,a=infos(data[k])
    if ratio_var_mean<0.05 and samples>2000:
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
    result=[]
    for k in range(i):
        result.append([])
    model=KMeans(n_clusters=i)
    model.fit(data_trié)
    prediction=model.predict(data_trié)
    for k in range(len(prediction)):
        result[prediction[k]].append(str(label['alim_nom_fr'][k])+" est un/une "+str(label['alim_grp_nom_fr'][k]))
        print(str(label['alim_nom_fr'][k])+" est un/une "+str(label['alim_grp_nom_fr'][k])+" classé dans le cluster "+str(prediction[k]))
    print(model.cluster_centers_)
    for j in range(len(result)):
        print(result[j])
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



#DecisionTree
scoring="accuracy")
print("score DecisionTree : "+str(from sklearn import tree

clf_DecisionTree = tree.DecisionTreeClassifier()
clf_DecisionTree = clf.fit(X, Y)

scores_DecisionTree = cross_val_score(clf_DecisionTree, X, Y, cv=5, scores_DecisionTree.mean()))


#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve

#choix de l'hyperparamètre "n_estimators"
model= RandomForestClassifier()
k=np.arange(1,50)
train_score, val_score = validation_curve(model, X, Y, "n_estimators", k, cv=5)
plt.plot(k,train_score(axis=1))
plt.plot(k,val_score(axis=1))
plt.ylabel("score")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

clf_RandomForestClassifier = RandomForestClassifier(n_estimators=10)
clf_RandomForestClassifier = clf.fit(X, y)

scores_RandomForest = cross_val_score(clf_RandomForestClassifier, X, Y, cv=5, scoring="accuracy")
print("score RandomForest : "+str(scores_RandomForest.mean()))

#Multi-Layer Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {"alpha":n}
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

>>> clf.fit(X, y)




