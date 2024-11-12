import pandas as pd
import numpy as np
import torch
import re
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import spacy
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier, Pool


# 1. Загрузка языковых моделей Spacy
def load_spacy_models():
    try:
        nlp_en = spacy.load('en_core_web_sm')
    except OSError:
        print("Модель en_core_web_sm не найдена. Установите её с помощью команды:")
        print("python -m spacy download en_core_web_sm")
        raise

    try:
        nlp_ru = spacy.load('ru_core_news_sm')
    except OSError:
        print("Модель ru_core_news_sm не найдена. Установите её с помощью команды:")
        print("python -m spacy download ru_core_news_sm")
        raise

    return nlp_en, nlp_ru


# 2. Функция для очистки и лемматизации текста
def clean_and_lemmatize(text, nlp_en, nlp_ru, lang='ru'):
    """
    Очищает и лемматизирует текст.

    :param text: Исходный текст.
    :param nlp_en: Загруженная модель spaCy для английского.
    :param nlp_ru: Загруженная модель spaCy для русского.
    :param lang: Язык текста ('en' для английского, 'ru' для русского).
    :return: Лемматизированный текст.
    """
    if pd.isna(text):
        return ""

    # Обрезка до 100 символов
    text = text[:100]

    # Удаление специальных символов и приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()

    # Выбор модели в зависимости от языка
    if lang == 'en':
        doc = nlp_en(text)
    elif lang == 'ru':
        doc = nlp_ru(text)
    else:
        raise ValueError("Поддерживаемые языки: 'en' для английского и 'ru' для русского.")

    # Лемматизация и удаление стоп-слов
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    return ' '.join(lemmas)


# 3. Функция для генерации эмбеддингов
def generate_embeddings(text_series, model, nlp_en, nlp_ru, lang='ru', batch_size=64):
    """
    Генерирует эмбеддинги для серии текстовых данных.

    :param text_series: pandas Series с текстовыми данными.
    :param model: Модель SentenceTransformer.
    :param nlp_en: Загруженная модель spaCy для английского.
    :param nlp_ru: Загруженная модель spaCy для русского.
    :param lang: Язык текста ('en' для английского, 'ru' для русского).
    :param batch_size: Размер батча для генерации эмбеддингов.
    :return: numpy.ndarray с эмбеддингами.
    """
    cleaned_texts = text_series.apply(lambda x: clean_and_lemmatize(x, nlp_en, nlp_ru, lang))
    embeddings = model.encode(cleaned_texts.tolist(), batch_size=batch_size, show_progress_bar=True)
    return embeddings


# 4. Основной пайплайн
def main(skip_ready=False):
    DATA_PATH = '/kaggle/input/forhypee/users_with_target.parquet'  # Замените на путь к вашему датасету
    user_new = pd.read_parquet(DATA_PATH)
    user_new = user_new[~user_new["is_extravert"].isna()]
    user_new["is_extravert"] = user_new["is_extravert"].astype(int)
    user_new["is_extravert"] = user_new["is_extravert"].replace({
        'True': 1, 'False': 0,
        'yes': 1, 'no': 0
    })

    # Параметры
    DATA_PATH = '/kaggle/input/forhypee/users_with_target.parquet'  # Замените на путь к вашему датасету
    PROCESSED_DATA_PATH = 'processed_data.pkl'  # Путь для сохранения обработанного DataFrame
    MODEL_SAVE_PATH = 'catboost_models'  # Папка для сохранения моделей
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    TARGETS = ['is_extravert', 'is_analyst', 'is_rigid']
    TEXT_COLUMNS = {
        'groups_names': 'ru',
        'groups_descriptions': 'ru',
        'posts_text': 'ru'  # Измените язык при необходимости
    }

    # 1. Загрузка данных
    print("Загрузка данных...")
    if not skip_ready:
        df = pd.read_parquet(DATA_PATH)
        df = df[~df["is_extravert"].isna()]
        print(f"Датасет загружен с размером {df.shape}")
        # 1.1 Кодирование целевых переменных в бинарный формат (0 и 1)
        for target in TARGETS:
            if True:
                df[target] = df[target].astype(int)
            elif df[target].dtype == 'object':
                # Преобразование строковых меток в 0 и 1
                df[target] = df[target].map({
                    'True': 1, 'False': 0,
                    'true': 1, 'false': 0,
                    'Yes': 1, 'No': 0
                })
            else:
                # Преобразование числовых меток в 0 и 1
                df[target] = df[target].astype(int)

            # Замена пропусков на 0 (или другое значение, если необходимо)
            if df[target].isna().any():
                print(f"Пропуски обнаружены в целевой переменной '{target}'. Заполнение нулями.")
                df[target] = df[target].fillna(0)

            # Проверка уникальных значений
            unique = df[target].unique()
            if set(unique) - {0, 1}:
                print(f"Внимание: целевая переменная '{target}' содержит неожиданные метки: {unique}")
                # Преобразование всех меток >0 в 1, остальные в 0
                df[target] = df[target].apply(lambda x: 1 if x > 0 else 0)
                unique = df[target].unique()
                print(f"После преобразования: {unique}")
        # 2. Загрузка языковых моделей
        print("Загрузка языковых моделей Spacy...")
        nlp_en, nlp_ru = load_spacy_models()

        # 3. Загрузка модели эмбеддингов
        print("Загрузка модели SentenceTransformer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('sergeyzh/rubert-tiny-turbo', device=device)

        # 4. Генерация эмбеддингов для текстовых столбцов
        embedding_feature_names = []
        for col, lang in TEXT_COLUMNS.items():
            print(f"Генерация эмбеддингов для столбца '{col}'...")
            embeddings = generate_embeddings(df[col], model, nlp_en, nlp_ru, lang=lang)
            # Добавляем эмбеддинги как одну колонку с векторами
            df[col + '_emb'] = embeddings.tolist()
            embedding_feature_names.append(col + '_emb')
            print(f"Эмбеддинги для '{col}' добавлены как новая колонка '{col}_emb'.")

        # 5. Подготовка признак
    else:
        df = pd.read_csv("data_with_embeds.csv")

    # Удаляем исходные текстовые столбцы
    df = df.drop(columns=list(TEXT_COLUMNS.keys()))

    # Определяем числовые и категориальные столбцы
    embedding_feature_names = ['groups_names_emb', 'groups_descriptions_emb', 'posts_text_emb']
    embedding_features = ['groups_names_emb', 'groups_descriptions_emb', 'posts_text_emb']
    # embedding_feature_names = []
    # for col, lang in TEXT_COLUMNS.items():
    #     emb_dim = 312  # Предполагаем размер эмбеддинга, проверьте фактический размер
    #     embedding_feature_names.extend([f"{col}_emb_{i}" for i in range(emb_dim)])

    # Остальные числовые столбцы (предполагаем, что они числовые)
    exclude_columns = ['vk_id', 'user_id', 'first_name', 'last_name', 'birth_date'] + TARGETS
    # Определяем числовые колонки (исключая эмбеддинговые)
    numerical_columns = [col for col in df.columns if col not in exclude_columns and col not in embedding_features]

    # Проверка на наличие числовых колонок
    if numerical_columns:
        # Нормализация числовых признаков
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    else:
        scaler = None
        print("Нет числовых колонок для нормализации.")

    df.to_pickle(PROCESSED_DATA_PATH)
    print(f"Обработанный DataFrame сохранен в '{PROCESSED_DATA_PATH}'")

    # 6. Разделение данных на обучающую и тестовую выборки с стратификацией по первому целевому
    print("Разделение данных на обучающую и тестовую выборки...")
    # Читать из pickle, чтобы убедиться, что список сохранен корректно
    df = pd.read_pickle(PROCESSED_DATA_PATH)

    X = df.drop(columns=TARGETS + ['vk_id', 'user_id', 'first_name', 'last_name', 'birth_date'])
    y_dict = {target: df[target] for target in TARGETS}

    # Чтобы обеспечить одинаковое разбиение для всех целей, используем общий индекс разбиения
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=None  # Можно выбрать одну цель для стратификации, если они сбалансированы
    )

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]

    y_train_dict = {target: y_dict[target].loc[train_indices] for target in TARGETS}
    y_test_dict = {target: y_dict[target].loc[test_indices] for target in TARGETS}
    for target in TARGETS:
        print(f"\nПроверка уникальных меток для целевой переменной '{target}'...")
        print(f"Уникальные метки в обучающей выборке: {y_train_dict[target].unique()}")
        print(f"Уникальные метки в тестовой выборке: {y_test_dict[target].unique()}")

        # Убедимся, что обе выборки содержат оба класса
        if set(y_train_dict[target].unique()) != {0, 1}:
            print(f"Ошибка: Обучающая выборка для '{target}' не содержит оба класса. Пропуск модели.")
            continue

        if set(y_test_dict[target].unique()) != {0, 1}:
            print(f"Ошибка: Тестовая выборка для '{target}' не содержит оба класса. Пропуск модели.")
            continue

    # 8. Обучение и оценка моделей CatBoost
    for target in TARGETS:
        print(f"\nОбучение модели для целевой переменной '{target}'...")

        # Проверка уникальных меток
        print(f"\nПроверка уникальных меток для целевой переменной '{target}'...")
        print(f"Уникальные метки в обучающей выборке: {y_train_dict[target].unique()}")
        print(f"Уникальные метки в тестовой выборке: {y_test_dict[target].unique()}")

        # Убедимся, что обе выборки содержат оба класса
        if set(y_train_dict[target].unique()) != {0, 1}:
            print(f"Ошибка: Обучающая выборка для '{target}' не содержит оба класса. Пропуск модели.")
            continue

        if set(y_test_dict[target].unique()) != {0, 1}:
            print(f"Ошибка: Тестовая выборка для '{target}' не содержит оба класса. Пропуск модели.")
            continue

        # Подготовка Pool с указанием эмбеддинг признаков
        train_pool = Pool(
            X_train,
            y_train_dict[target],
            embedding_features=embedding_feature_names
        )

        test_pool = Pool(
            X_test,
            y_test_dict[target],
            embedding_features=embedding_feature_names
        )

        # Обработка дисбаланса классов (опционально)
        class_counts = y_train_dict[target].value_counts().to_dict()
        total = sum(class_counts.values())
        class_weights = {cls: total / count for cls, count in class_counts.items()}
        print(f"Классовые веса для '{target}': {class_weights}")

        # Инициализация модели
        model_cb = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            eval_metric='F1',
            random_seed=42,
            task_type='GPU' if device == "cuda" else 'CPU',
            class_weights=class_weights,  # Добавление классовых весов
            verbose=100
        )

        # Обучение модели
        model_cb.fit(
            train_pool,
            eval_set=test_pool,
            early_stopping_rounds=50,
            use_best_model=True
        )

        # Предсказания и оценка
        y_pred = model_cb.predict(X_test)

        # Проверка уникальных предсказанных меток
        unique_preds = np.unique(y_pred)
        print(f"Уникальные предсказанные метки для '{target}': {unique_preds}")

        # Убедимся, что y_pred содержит только 0 и 1
        if set(unique_preds) - {0, 1}:
            print(f"Ошибка: Предсказания модели '{target}' содержат неожиданные метки: {unique_preds}. Пропуск оценки.")
            continue

        accuracy = accuracy_score(y_test_dict[target], y_pred)
        f1 = f1_score(y_test_dict[target], y_pred, average='binary')
        print(f"Модель для '{target}': Accuracy = {accuracy:.4f}, F1-score = {f1:.4f}")
        print("Отчет классификации:")
        print(classification_report(y_test_dict[target], y_pred))

        # Сохранение модели
        model_path = os.path.join(MODEL_SAVE_PATH, f"catboost_{target}.cbm")
        model_cb.save_model(model_path)
        print(f"Модель сохранена в '{model_path}'")

    # 9. Сохранение scaler для последующей предобработки новых данных
    if scaler is not None:
        scaler_path = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Скалер сохранен в '{scaler_path}'")

    # 10. Сохранение embedding_feature_names для последующей загрузки
    embedding_features_path = os.path.join(MODEL_SAVE_PATH, "embedding_features.pkl")
    with open(embedding_features_path, 'wb') as f:
        pickle.dump(embedding_feature_names, f)
    print(f"Список эмбеддинг признаков сохранен в '{embedding_features_path}'")

    print("\nПайплайн завершен успешно.")


# 10. Функция для загрузки и использования модели
def load_and_predict(new_data, target, models_path='catboost_models'):
    """
    Загружает модель CatBoost и использует её для предсказания.

    :param new_data: pandas DataFrame с новыми данными.
    :param target: Целевая переменная для предсказания.
    :param models_path: Путь к сохраненным моделям.
    :return: Предсказания модели.
    """
    # Загрузка необходимых файлов
    scaler_path = os.path.join(models_path, "scaler.pkl")
    embedding_features_path = os.path.join(models_path, "embedding_features.pkl")
    model_path = os.path.join(models_path, f"catboost_{target}.cbm")

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = None
        print("Скалер не найден. Пропуск нормализации числовых признаков.")

    with open(embedding_features_path, 'rb') as f:
        embedding_feature_names = pickle.load(f)

    # Загрузка модели
    model_cb = CatBoostClassifier()
    model_cb.load_model(model_path)

    # Предобработка новых данных
    # Здесь необходимо повторить шаги предобработки: очистка текста, генерация эмбеддингов и нормализация
    # Для примера предполагаем, что new_data уже содержит эмбеддинги и числовые признаки

    # Например, если нужно повторить обработку:
    # cleaned_texts = new_data['groups_names'].apply(lambda x: clean_and_lemmatize(x, nlp_en, nlp_ru, lang='ru'))
    # embeddings = model.encode(cleaned_texts.tolist())
    # new_data['groups_names_emb'] = embeddings.tolist()
    # и т.д. для других текстовых колонок

    # Нормализация числовых признаков
    numerical_columns = [col for col in new_data.columns if
                         col not in embedding_feature_names and col not in ['vk_id', 'user_id', 'first_name',
                                                                            'last_name', 'birth_date']]
    if scaler is not None and numerical_columns:
        new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

    # Создание Pool для предсказания
    prediction_pool = Pool(
        new_data,
        embedding_features=embedding_feature_names
    )

    # Предсказание
    predictions = model_cb.predict(prediction_pool)
    return predictions


if __name__ == 'main':
    main()
