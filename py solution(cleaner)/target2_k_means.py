import pandas as pd
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

mindset = pd.read_csv("sumOfTechHumTagsInGroupsAndPosts2.csv")
user = pd.read_parquet("user2.parquet")
user_copy = user.copy()
gender = pd.read_parquet("predicted_genders2.parquet")
user = user.drop(columns=['vk_id',"first_name","last_name", "birth_date"])
user["gender"] = gender["predicted_gender"]
users_new = user.merge(mindset, left_on="id", right_on="user_id", how="left")
users_new = users_new.dropna()
train = users_new.copy()
df_final = pd.get_dummies(users_new, columns=['gender'])
new_columns = ['age_14.0', 'age_15.0', 'age_16.0', 'age_17.0', 'age_18.0']

# Присваиваем значение 0 для всех новых столбцов
for col in new_columns:
    df_final[col] = False

df_final2 = pd.get_dummies(df_final, columns=['age'])
users_new = df_final2.copy()
users_new = users_new.drop(["id","user_id"], axis=1)
users_new["tech"] = users_new["tech"].replace(0, np.nan).apply(lambda x: np.log1p(x))
users_new["hum"] = users_new["hum"].replace(0, np.nan).apply(lambda x: np.log1p(x))
users_new.fillna(0, inplace=True)

# Отдельные скейлеры для каждого столбца
scaler_tech = StandardScaler()
scaler_hum = StandardScaler()

# Стандартизуем каждый столбец по отдельности
users_new['tech'] = scaler_tech.fit_transform(users_new[['tech']])
users_new['hum'] = scaler_hum.fit_transform(users_new[['hum']])

# Вычисляем разность после стандартизации
users_new['diff'] = (users_new['tech'] - users_new['hum'])*10

users_new = users_new.drop("tech", axis=1)
users_new = users_new.drop("hum", axis=1)

sns.heatmap(round(users_new.corr(), 2), annot=True)

#---------------------------model---------------------
# Применение KMeans для кластеризации на 2 кластера
kmeans = KMeans(n_clusters=2, random_state=42)
users_new['cluster'] = kmeans.fit_predict(users_new)

# Добавление целевой переменной на основе кластера (0 - экстраверт, 1 - интроверт)
train['target'] = users_new['cluster'].apply(lambda x: 0 if x == 0 else 1)
users_new.to_parquet("train2.parquet", index=False)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(users_new.drop("cluster", axis=1))
principal_df = pd.DataFrame(data = pca_result, columns=["PC1","PC2"])
principal_df["cluster"] = kmeans.labels_

# Получение центроидов PCA пространства
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Визуализация кластеров и центроидов
plt.figure(figsize=(8, 6))
scatter = plt.scatter(principal_df['PC1'], principal_df['PC2'],
                      c=principal_df['cluster'], cmap='viridis', alpha=0.7)

plt.title('Визуализация кластеров с помощью PCA')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend(*scatter.legend_elements(), title='Кластеры')
plt.grid(True)
plt.show()

test =pd.read_parquet("feature_target2.parquet")

# Делаем предсказания для нового датасета на обученной модели KMeans
predicted_clusters = kmeans.predict(test)


test["tech"] = test["tech"].replace(0, np.nan).apply(lambda x: np.log1p(x))
test["hum"] = test["hum"].replace(0, np.nan).apply(lambda x: np.log1p(x))
test.fillna(0, inplace=True)

# Стандартизуем каждый столбец по отдельности
test['tech'] = scaler_tech.fit_transform(test[['tech']])
test['hum'] = scaler_hum.fit_transform(test[['hum']])

# Вычисляем разность после стандартизации
test['diff'] = (test['tech'] - test['hum'])*10

test = test.drop("tech", axis=1)
test = test.drop("hum", axis=1)

test.rename(columns={'age_19': 'age_19.0'}, inplace=True)

test = test.drop("Unnamed: 0",axis=1)

# Добавляем результат предсказаний к новому датасету
test['predicted_cluster'] = predicted_clusters
user_copy['predicted_cluster'] = predicted_clusters

test_predict2 = pd.DataFrame()
test_predict2["vk_id"] = user_copy["vk_id"]
test_predict2["target2"] = user_copy["predicted_cluster"]

test_predict2.to_csv("test_predict2.csv", index=False)