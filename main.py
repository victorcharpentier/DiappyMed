import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

%matplotlib inline
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None
df = pd.read_excel('C:\\Users\\nisri\\Downloads\\ciqual_table.xls')
alim_nom_fr = df['alim_nom_fr']
features = df.loc[:,'Energie, Règlement UE N° 1169/2011 (kJ/100 g)':]
df = df.loc[:,'Energie, Règlement UE N° 1169/2011 (kJ/100 g)':]


print(df.shape)
print(df.dtypes)

# on selectionne les colonnes numeriques
df_numeriques = df.select_dtypes(include=[np.number])
colonnes_numeriques = df_numeriques.columns.values
print(colonnes_numeriques)

# on selectionne les colonnes non numeriques
df_non_numeriques = df.select_dtypes(exclude=[np.number])
colonnes_non_numeriques = df_non_numeriques.columns.values
print(colonnes_non_numeriques)

# Les 10 premieres colonnes ne sont pas intérressantes s'elles sont manquantes
#on s'interesse à savoir si le reste des colonnes contient des valeurs ou pas
colonnes = df.columns[10:]
couleurs = ['#000099', '#ffff00'] # jaune : pas de données, bleu : il y a des données
sns.heatmap(df[colonnes].isnull(), cmap=sns.color_palette(couleurs))


#Ici on peut voir le pourcentage des données vides qu'on a sur la table
for col in df.columns:
    cellules_vides = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(cellules_vides*100)))

# on crée un indicateur pour voir les données manquantes pour nos features
for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        df['{}_ismissing'.format(col)] = missing

# en se basant sur ces indicateurs, nous allons dessiner l'histogramme
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['num_missing'] = df[ismissing_cols].sum(axis=1)

df['num_missing'].value_counts().reset_index().sort_values(by='index').plot.bar(x='index', y='num_missing')

df = df.apply(pd.to_numeric, errors='coerce', downcast='float')

for col in df.columns:
    missing = df[col].isnull()
    num_missing = np.sum(missing)

    if num_missing > 0:
        df['{}_ismissing'.format(col)] = missing
        med = df[col].median()# on remplace les valeurs manquantes par des médianes
        df[col] = df[col].fillna(med)

df = df.loc[:,:'Vitamine B12 (µg/100 g)']

table = pd.concat([alim_nom_fr, df], axis=1)

plt.scatter(df['Vitamine B6 (mg/100 g)'], alim_nom_fr)
plt.xlabel('features')
plt.ylabel('aliments')
plt.show()

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()