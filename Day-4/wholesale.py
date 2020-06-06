import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


try:
    data = pd.read_csv("Wholesale customers data.csv")
    data.drop(labels=(['Channel','Region']),axis=1,inplace=True)
    print('Wholesale customers has {} samples with {} features each'.format(*data.shape))
except:
    print('Sorry! Dataset could not be loaded.')

data.head()

data.describe()

data.info()

indices = [22,154,398]

samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
print("Chosen samples of wholesale customers dataset:")
display(samples)

pcts = 100. * data.rank(axis=0, pct=True).iloc[indices].round(decimals=3)

sns.heatmap(pcts, annot=True, vmin=1, vmax=99, fmt='.1f', cmap='YlGnBu')
plt.title('Percentile ranks of\nsamples\' category spending')
plt.xticks(rotation=45, ha='center');

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

new_data = data.drop('Milk',axis=1)

X_train, X_test, y_train, y_test = train_test_split(new_data, data['Milk'], test_size=0.25, random_state=1)

regressor =  DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

score = regressor.score(X_test, y_test)
print(score)

pd.plotting.scatter_matrix(data, alpha=0.3,figsize=(15,8),diagonal='kde' )
plt.tight_layout()

log_data = np.log(data.copy())

log_samples = np.log(samples)

pd.plotting.scatter_matrix(log_data, alpha=0.5, figsize=(14,8),diagonal='kde')
plt.tight_layout()

print("Original chosen samples of wholesale customers dataset:")
display(samples)

print("Log-transformed samples of wholesale customers dataset:")
display(log_samples)

for feature in log_data.keys():

    
    Q1 = np.percentile(log_data, 25)

    
    Q3 = np.percentile(log_data, 75)

    
    step = (Q3 - Q1) * 1.5
    

    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
    
outliers  = [66, 75, 338, 142, 154, 289]


good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

from sklearn.decomposition import PCA

pca = PCA(n_components=6)
pca.fit(good_data)

pca_samples = pca.transform(log_samples)

print(pca.components_)

print(pca.explained_variance_)

pca_samples