import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv('./data/opioids_data.csv', sep=',',engine='python')

# Drops unnecessary columns
X = data.drop(['WorldwideUniqueCaseIdentification', 'Category', 'date_received', 'Outcome', 'OutcomeCodeSystemVersion', 'OutcomeText', 'reaction_impact',
           'ID', 'CultureID', 'SOCAbbreviation', 'IsCurrent', 'IsDefaultSOC', 'Primary Source Description', 'summary', 'narrative' ], axis=1)
X_clean = X.dropna(axis=0)

# Specify the columns to be transformed and the transformers to use
numeric_cols = ['Status', 'BodyWeight', 'Height', 'age_year', 'LLTCode', 'PTCode', 'HLTCode', 'SOCCode']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_cols = ['sex', 'ATCText', 'GenericDrugName', 'LLTName', 'PTName', 'HLTName', 'SOCName']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the transformers into a ColumnTransformer object
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_transformed = preprocessor.fit_transform(X_clean)


# apply K-Means
km = KMeans(n_clusters=10, init="k-means++", random_state=5)
y_means = km.fit(X_transformed)
labels = y_means.labels_

# use pca to reduce dimensionality to visualize clusters later on
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_transformed.toarray())

# plotting the clusters
fig, ax = plt.subplots()
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
legend = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend)
plt.show()