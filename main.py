from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Определяем список товарных категорий
categories = []
with open("categories.txt", 'r', encoding='utf-8') as f:
    lines = set(f.readlines())
    for line in lines:
        categories.append(line)
training_data = []
# Определяем обучающий набор данных (название товара - категория)
with open("learning_data.csv", 'r', encoding='utf-8') as f:
    for line in f:
        out_str = tuple(line.split(';'))
        training_data.append(out_str)

# Разделяем данные на признаки и целевую переменную
training_labels, training_features = zip(*training_data)

# Создаем векторизатор и преобразуем признаки в числовой формат
vectorizer = CountVectorizer()
training_features_vectorized = vectorizer.fit_transform(training_features)

# Инициализируем мультиномиальный наивный Байесовский классификатор и обучаем его на обучающем наборе данных
classifier = MultinomialNB()
classifier.fit(training_features_vectorized, training_labels)
with  open("check_ data.txt", 'r', encoding='utf-8') as f:
    for line in f:
    # Используем обученную модель для классификации новых товаров
        new_product_name = line.strip()
        new_product_name_vectorized = vectorizer.transform([new_product_name])
        predicted_category = classifier.predict(new_product_name_vectorized)[0]

        print(f"{new_product_name}    ---      {predicted_category}")
