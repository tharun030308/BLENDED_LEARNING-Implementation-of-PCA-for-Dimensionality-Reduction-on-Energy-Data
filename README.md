# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('HeightsWeights.csv')
X=data[['Height(Inches)', 'Weight(Pounds)']]


plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(6,5))


sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

```

## Output:
<img width="829" height="634" alt="image" src="https://github.com/user-attachments/assets/8283450a-cea8-436d-b00c-88e0f61e4370" />
<img width="772" height="507" alt="image" src="https://github.com/user-attachments/assets/e0d91b32-2258-4872-ac6c-119505101ef2" />

## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
