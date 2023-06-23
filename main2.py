import pandas as pd
import nltk
from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import Pipeline


def preprocess_text(text):
    text = str(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords \
              and token != " " \
              and len(token) >= 3 \
              and token.strip() not in punctuation \
              and token.isdigit() == False]
    text = " ".join(tokens)
    return text


sku = pd.read_csv('learning_data.csv', sep=';')
sku.head()
sku.groupby('Category').agg(['count'])
mystem = Mystem()
russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(
    ['лента', 'ассорт', 'разм', 'арт', 'что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', 'к', 'на'])
print("-4addada")
sku['processed'] = sku["SKU"].apply(preprocess_text)
print("-3addada")
sku.head()
print("-2addada")
x = sku.processed
print("-1addada")
y = sku.Category
print("0addada")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print("1sdadsdsdada")
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
print("2dsadsaad")
text_clf = text_clf.fit(X_train, y_train)
y_pred = text_clf.predict(X_test)
print('Score:', text_clf.score(X_test, y_test))
