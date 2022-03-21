# Databricks notebook source
import os
import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph.
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# COMMAND ----------

import string
import re
import nltk
from nltk.stem import WordNetLemmatizer


# defining the function to remove punctuation

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree


# defining function for tokenization



def tokenization(text):
    tokens = re.split(' ',text)
    return tokens


nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

# data should be the text column from the dataset
def preprocessing(data):
    data= data.apply(lambda x:remove_punctuation(x))
    data= data.apply(lambda x: x.lower())
    data= data.apply(lambda x: tokenization(x))
    data= data.apply(lambda x:remove_stopwords(x))
    data= data.apply(lambda x:lemmatizer(x))
    
    return data

# COMMAND ----------

labeled_df = pd.read_parquet("/dbfs/FileStore/shared_uploads/yl817@duke.edu/full_raw_data_parquet-1.gzip")
labeled_df.head()

# COMMAND ----------

labeled_df = labeled_df.sample(100000)


cleaned_df = preprocessing(labeled_df.text)
cleaned_df.head()

# COMMAND ----------

labeled_df["text_cleaned"] = cleaned_df.str[0]

labeled_df['sentiment_id'] = labeled_df['sentiment']
labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("negative", 0)
labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("neutral", 1)
labeled_df['sentiment_id'] = labeled_df['sentiment_id'].replace("positive", 2)
labeled_df['sentiment_id'].value_counts()


# COMMAND ----------

labeled_df['sentiment'].value_counts().sort_values().plot(kind = 'barh')


# COMMAND ----------

count_class_2, count_class_1, count_class_0 = labeled_df.sentiment_id.value_counts()

df_class_0 = labeled_df[labeled_df['sentiment_id'] == 0]
df_class_1 = labeled_df[labeled_df['sentiment_id'] == 1]
df_class_2 = labeled_df[labeled_df['sentiment_id'] == 2]

df_class_1_under = df_class_1.sample(count_class_0)
df_class_2_under = df_class_2.sample(count_class_0)
undersample_df_cleaned = pd.concat([df_class_0, df_class_1_under, df_class_2_under], axis=0)

print('Random under-sampling:')
print(undersample_df_cleaned.sentiment.value_counts())
undersample_df_cleaned = undersample_df_cleaned.dropna(axis=0,how = 'any')
undersample_df_cleaned.sentiment.value_counts().plot(kind='bar', title='Count (target)')
undersample_df_cleaned.head()

# COMMAND ----------

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')
# We transform each complaint into a vector
features = tfidf.fit_transform(undersample_df_cleaned.text_cleaned.dropna()).toarray()
labels = undersample_df_cleaned.sentiment_id
print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

# COMMAND ----------

# Create a new column 'category_id' with encoded categories
category_id_df = undersample_df_cleaned[['sentiment', 'sentiment_id']].drop_duplicates()

category_id_df.values

# COMMAND ----------

# Dictionaries for future use

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['sentiment_id', 'sentiment']].values)


# COMMAND ----------

#Finding the three most correlated terms with each of the product categories
N = 3
for Product, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("n==> %s:" %(Product))
    print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
    print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))


# COMMAND ----------

X = undersample_df_cleaned['text_cleaned'] # Collection of documents
y = undersample_df_cleaned['sentiment_id'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state = 0)

# COMMAND ----------

# importing the relevant modules
from sklearn.feature_extraction.text import CountVectorizer

# vectorizing the sentences
cv = CountVectorizer(binary = True) # implies that it indicates whether the word is present or not.

cv.fit(X) # find all the unique words from the training set

X_train_vec = cv.transform(X_train)
X_test_vec = cv.transform(X_test)

# importing the relevant modules
import xgboost as xgb

# creating a variable for the new train and test sets
xgb_train = xgb.DMatrix(X_train_vec, y_train)
xgb_test = xgb.DMatrix(X_test_vec, y_test)

from sklearn.metrics import accuracy_score, f1_score

# Setting the Parameters of the Model
param = {'eta': 1,
         'max_depth': 50,
         'num_class': 3,
         'objective': 'multi:softmax'}

# Training the Model
xgb_model = xgb.train(param, xgb_train, num_boost_round=30)
# Predicting using the Model
y_pred = xgb_model.predict(xgb_test)
y_pred = np.where(np.array(y_pred) > 0.5, 1, 0)  # converting them to 1/0’s
# Evaluation of Model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'F1 Weighted: {f1_score(y_test, y_pred, average="micro")}')

# COMMAND ----------

models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])



mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1,
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc

# COMMAND ----------

# random search logistic regression model on the sonar dataset
from scipy.stats import loguniform
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfTransformer

# build model

X = undersample_df_cleaned['text_cleaned'] # Collection of documents
y = undersample_df_cleaned['sentiment_id'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

X_train_vec = tfidf.fit_transform(X_train).toarray()
X_test_vec = tfidf.fit_transform(X_test).toarray()

model = LogisticRegression()
# define evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1, )
# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)
# define search
search = RandomizedSearchCV(model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1, verbose=10)
# execute search
result = search.fit(X_train_vec, y_train)
# summarize result
print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

