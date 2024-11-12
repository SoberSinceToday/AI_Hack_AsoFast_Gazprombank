import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

user = pd.read_parquet("user2.parquet")
friend = pd.read_parquet("friend2.parquet")
photo = pd.read_parquet("photo2.parquet")
biography = pd.read_parquet("biography2.parquet")
post = pd.read_parquet("post2.parquet")
group = pd.read_parquet("group_table2.parquet")

newd = pd.DataFrame(columns=['user_id', 'anon_pab'])
group2 = group
# Группировка по user_id
for user_id, gr in group2.groupby('user_id'):  # Исправленный цикл
    gr['name'].fillna('', inplace=True)
    counter = gr[gr['name'].str.lower().str.contains('аноним')].shape[0]
    counter += gr[gr['name'].str.lower().str.contains('знакомств')].shape[0]
    new_row = pd.DataFrame({'user_id': [user_id], 'anon_pab': [counter]})
    newd = pd.concat([newd, new_row])
anon_group = newd

biography2 = biography.drop(columns=["id", "user_id"]).applymap(lambda x: 0 if (x is None) else 1)
biography2["user_id"] = biography["user_id"]

friend["city"] = friend["city"].apply(lambda x: None if (x is None or x == "") else x)
friend = friend[friend["city"].notna()]

diff_city_count = friend.groupby('user_id').agg(diff_city_count=("city", "nunique"))
friends_count = friend.groupby('user_id').agg(friend_count=("id", "count"))
avg_photo_likes = photo.groupby('user_id').agg(avg_photo_likes=("like_count", "mean"))
photo_count = photo.groupby('user_id').agg(photo_count=("id", "count"))
post_count_owner = post[post["isowner"] == 1].groupby('user_id').agg(post_count_owner=("id", "count"))
post_count_friend = post[post["isowner"] == 0].groupby('user_id').agg(post_count_friend=("id", "count"))
post_count_delete = post[post["text"] == "Запись удалена"].groupby('user_id').agg(post_count_delete=("id", "count"))
group_count = group.groupby('user_id').agg(group_count=("id", "count"))
user_new = user
user_new["user_id"] = user["id"]
user_new = user_new.drop("id", axis=1)
user_new = user_new.merge(avg_photo_likes, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(photo_count, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(post_count_owner, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(post_count_friend, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(post_count_delete, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(group_count, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(anon_group, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(diff_city_count, left_on="user_id", right_on="user_id", how="left")
user_new = user_new.merge(friends_count, left_on="user_id", right_on="user_id", how="left")
user_new["like_friend_ratio"] = (user_new["friend_count"]) / (user_new["avg_photo_likes"] + 1)
user_new["friend_count"] = user_new["friend_count"].fillna(-1)
user_new.to_parquet("base_test.parquet", index=False)

# Определение IQR
Q1 = user_new['friend_count'].quantile(0.25)
Q3 = user_new['friend_count'].quantile(0.85)
IQR = Q3 - Q1

# Верхняя граница по IQR
upper_bound_IQR = Q3 + 1.5 * IQR

print(f"Upper bound by IQR: {upper_bound_IQR}")

# Фильтрация данных
filtered_IQR = user_new[user_new['friend_count'] <= upper_bound_IQR]

print(f"Number of rows after IQR filtering: {len(filtered_IQR)}")

filtered_IQR["friend_count"] = filtered_IQR["friend_count"].apply(lambda x: None if x == -1 else x)
user_new = filtered_IQR

# Среднее и стандартное отклонение для правила 3-х сигм
mean_friend_count = user_new[user_new["friend_count"].isna()]['avg_photo_likes'].mean()
std_friend_count = user_new[user_new["friend_count"].isna()]['avg_photo_likes'].std()

# Среднее и стандартное отклонение для правила 3-х сигм 2
mean_friend_count2 = user_new[~user_new["friend_count"].isna()]['like_friend_ratio'].mean()
std_friend_count2 = user_new[~user_new["friend_count"].isna()]['like_friend_ratio'].std()

# Верхняя граница по правилу 3-х сигм
upper_bound_3sigma = mean_friend_count + 3 * std_friend_count

# Верхняя граница по правилу 3-х сигм 2
upper_bound_3sigma2 = mean_friend_count2 + 3 * std_friend_count2

print(f"Upper bound by 3-sigma rule: {upper_bound_3sigma}")
print(f"Upper bound by 3-sigma rule2: {upper_bound_3sigma2}")

# Фильтрация данных
filtered_3sigma_with_nan = user_new[user_new["friend_count"].isna()].reset_index()
filtered_3sigma_without_nan = user_new[~user_new["friend_count"].isna()].reset_index()

filtered_3sigma_with_nan = filtered_3sigma_with_nan[
    filtered_3sigma_with_nan['avg_photo_likes'] <= upper_bound_3sigma].reset_index()
filtered_3sigma_without_nan = filtered_3sigma_without_nan[
    filtered_3sigma_without_nan['like_friend_ratio'] <= upper_bound_3sigma2].reset_index()

# Объединение результатов
all_filtered_3sigma = pd.concat([filtered_3sigma_with_nan, filtered_3sigma_without_nan])

print(f"Number of rows after 3-sigma filtering: {len(all_filtered_3sigma)}")
user_new = all_filtered_3sigma
user_new.drop(columns=["level_0", "index"], inplace=True)
user_new["avg_photo_likes"] = user_new["avg_photo_likes"].fillna(0)
user_new["photo_count"] = user_new["photo_count"].fillna(0)
user_new["post_count_owner"] = user_new["post_count_owner"].fillna(0)
user_new["post_count_friend"] = user_new["post_count_friend"].fillna(0)
user_new["post_count_delete"] = user_new["post_count_delete"].fillna(0)

user_new.fillna(-3, inplace=True)
user_new.drop("like_friend_ratio", axis=1, inplace=True)
features = user_new.drop(columns=["vk_id", "age", "user_id", "first_name", "last_name", "birth_date"])
features_old = features.copy()
features_columns = features.columns

# Стандартизация только видимых друзей
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

features["friend_count"] = features["friend_count"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
features["friend_count"] = features["friend_count"].replace(np.nan, 0)
features["friend_count"] = features["friend_count"].replace(-3, np.nan)

f_sc = StandardScaler()
features['friend_count'] = f_sc.fit_transform(features['friend_count'].values.reshape(-1, 1))

features["diff_city_count"] = features["diff_city_count"].replace(0, np.nan).apply(
    lambda x: np.log1p(x) if x != -3 else x)
features["diff_city_count"] = features["diff_city_count"].replace(np.nan, 0)
features["diff_city_count"] = features["diff_city_count"].replace(-3, np.nan)
diff_sc = StandardScaler()
features['diff_city_count'] = diff_sc.fit_transform(features['diff_city_count'].values.reshape(-1, 1))

features["group_count"] = features["group_count"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
features["group_count"] = features["group_count"].replace(np.nan, 0)
features["group_count"] = features["group_count"].replace(-3, np.nan)
gr_sc = StandardScaler()
features['group_count'] = gr_sc.fit_transform(features['group_count'].values.reshape(-1, 1))

features["anon_pab"] = features["anon_pab"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
features["anon_pab"] = features["anon_pab"].replace(np.nan, 0)
features["anon_pab"] = features["anon_pab"].replace(-3, np.nan)
anon_sc = RobustScaler()
features['anon_pab'] = anon_sc.fit_transform(features['anon_pab'].values.reshape(-1, 1))

features.fillna(-2, inplace=True)

features["avg_photo_likes"] = features["avg_photo_likes"].replace(0, np.nan).apply(lambda x: np.log1p(x))
features["photo_count"] = features["photo_count"].replace(0, np.nan).apply(lambda x: np.log1p(x))
postO_sc = MinMaxScaler()
features['post_count_owner'] = postO_sc.fit_transform(features['post_count_owner'].values.reshape(-1, 1))
postF_sc = MinMaxScaler()
features['post_count_friend'] = postF_sc.fit_transform(features['post_count_friend'].values.reshape(-1, 1))
postD_sc = MinMaxScaler()
features['post_count_delete'] = postD_sc.fit_transform(features['post_count_delete'].values.reshape(-1, 1))

like_sc = StandardScaler()
features['avg_photo_likes'] = like_sc.fit_transform(features['avg_photo_likes'].values.reshape(-1, 1))
like_sc = RobustScaler()
features['photo_count'] = like_sc.fit_transform(features['photo_count'].values.reshape(-1, 1))
features.fillna(0, inplace=True)

# Применение KMeans для кластеризации на 2 кластера
kmeans = KMeans(n_clusters=2, random_state=42)
features['cluster'] = kmeans.fit_predict(features)

# Добавление целевой переменной на основе кластера (0 - экстраверт, 1 - интроверт)
train = user_new
train['target'] = features['cluster']
features.to_parquet("train.parquet", index=False)
# Применение PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features.drop("cluster", axis=1))
principal_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
principal_df["cluster"] = kmeans.labels_

# Получение центров кластеров
centers = kmeans.cluster_centers_
# Нормализация центров для PCA
centers_pca = pca.transform(centers)

# Вычисляем расстояния от точек до центров кластеров
distances_to_centers = np.array(
    [np.linalg.norm(principal_df[['PC1', 'PC2']].values - center, axis=1) for center in centers_pca])

# Находим разницу между расстояниями до обоих кластеров
distance_diff = np.abs(distances_to_centers[0] - distances_to_centers[1])

# Устанавливаем порог для определения "практически одинакового" расстояния
threshold = 0.25  # Порог, определяющий, насколько близки расстояния

# Фильтрация точек, находящихся практически одинаково от двух кластеров
filtered_df = principal_df[distance_diff > threshold]

# Визуализация отфильтрованных кластеров
plt.figure(figsize=(8, 6))
scatter = plt.scatter(filtered_df['PC1'], filtered_df['PC2'],
                      c=filtered_df['cluster'], cmap='viridis', alpha=0.7)
plt.title('Визуализация кластеров с помощью PCA (отфильтровано)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend(*scatter.legend_elements(), title='Кластеры')
plt.grid(True)
plt.show()

test = pd.read_parquet("features_test.parquet")

test["friend_count"] = test["friend_count"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
test["friend_count"] = test["friend_count"].replace(np.nan, 0)
test["friend_count"] = test["friend_count"].replace(-3, np.nan)

f_sc = StandardScaler()
test['friend_count'] = f_sc.fit_transform(test['friend_count'].values.reshape(-1, 1))

test["diff_city_count"] = test["diff_city_count"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
test["diff_city_count"] = test["diff_city_count"].replace(np.nan, 0)
test["diff_city_count"] = test["diff_city_count"].replace(-3, np.nan)
diff_sc = StandardScaler()
test['diff_city_count'] = diff_sc.fit_transform(test['diff_city_count'].values.reshape(-1, 1))

test["group_count"] = test["group_count"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
test["group_count"] = test["group_count"].replace(np.nan, 0)
test["group_count"] = test["group_count"].replace(-3, np.nan)
gr_sc = StandardScaler()
test['group_count'] = gr_sc.fit_transform(test['group_count'].values.reshape(-1, 1))

test["anon_pab"] = test["anon_pab"].replace(0, np.nan).apply(lambda x: np.log1p(x) if x != -3 else x)
test["anon_pab"] = test["anon_pab"].replace(np.nan, 0)
test["anon_pab"] = test["anon_pab"].replace(-3, np.nan)
anon_sc = RobustScaler()
test['anon_pab'] = anon_sc.fit_transform(test['anon_pab'].values.reshape(-1, 1))

test.fillna(-2, inplace=True)

test["avg_photo_likes"] = test["avg_photo_likes"].replace(0, np.nan).apply(lambda x: np.log1p(x))
test["photo_count"] = test["photo_count"].replace(0, np.nan).apply(lambda x: np.log1p(x))
postO_sc = MinMaxScaler()
test['post_count_owner'] = postO_sc.fit_transform(test['post_count_owner'].values.reshape(-1, 1))
postF_sc = MinMaxScaler()
test['post_count_friend'] = postF_sc.fit_transform(test['post_count_friend'].values.reshape(-1, 1))
postD_sc = MinMaxScaler()
test['post_count_delete'] = postD_sc.fit_transform(test['post_count_delete'].values.reshape(-1, 1))

like_sc = StandardScaler()
test['avg_photo_likes'] = like_sc.fit_transform(test['avg_photo_likes'].values.reshape(-1, 1))
like_sc = RobustScaler()
test['photo_count'] = like_sc.fit_transform(test['photo_count'].values.reshape(-1, 1))
test.fillna(0, inplace=True)

predicted_clusters = kmeans.predict(test)

# Добавляем результат предсказаний к новому датасету
test['predicted_cluster'] = predicted_clusters
base_test = pd.read_parquet("base_test.parquet")
base_test['predicted_cluster'] = predicted_clusters
base_test["predicted_cluster"] = base_test["predicted_cluster"].apply(lambda x: "Э" if x == 1 else "И")

test_predict1 = pd.DataFrame()
test_predict1["vk_id"] = base_test["vk_id"]
test_predict1["target1"] = base_test["predicted_cluster"]
test_predict1.to_csv("test_predict1.csv", index=False)
