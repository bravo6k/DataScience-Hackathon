import warnings
warnings.filterwarnings("ignore")
import numpy as np
import json
import pandas as pd
import tensorflow
from nltk.corpus import stopwords
import re
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from collections import Counter
from pipe import transform_text_func,FeatureExtractor, ImputeNA, CategoricalEncoding,text
from scipy.sparse import hstack
from sklearn.linear_model import RidgeClassifier,Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, make_scorer,mean_squared_error
from sklearn.preprocessing import StandardScaler
from nltk.stem import PorterStemmer
from scipy.sparse import csr_matrix
import logging
from sklearn.pipeline import make_pipeline, make_union
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation, BatchNormalization, PReLU
from keras.initializers import he_uniform
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam, SGD
from keras.models import Model
import gc

def generate_interaction(feature_list):
    total = len(feature_list)*(len(feature_list)-1)/2
    step = 0
    for i,ai in enumerate(feature_list):
        for j,bj in enumerate(feature_list):
            if i<j:
                x = total_data[ai]
                y = total_data[bj]
                t = []
                for l in range(total_data.shape[0]):
                    t.append(str(x[l])+' '+ str(y[l]))
                total_data[ai+'_'+bj] = t
                step +=1
                bar.drawProgressBar(step/total)


def upper_prob(data):
    uppercase = []
    total = len(data)
    step = 0
    for i in data:
        length = len(i.split())
        tmp = []
        for j in i:
            if j.isupper():
                tmp.append(j)
        uppercase.append(len(tmp)/length)
    return(uppercase)

def scale(data):
    data = np.array(data).reshape(-1,1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return(data)

def tokenize_stop(text):
    text = text.replace('.',' ')
    text = text.split()
    return(text)

def stop_and_max_feature(data, top_frequent_num, word_least_frequency_num,scale_v,stop_v):
    x_lower = [sublist.lower() for sublist in data]
    x_lower = [tokenize_stop(i) for i in x_lower]
    x_unlist = []
    for i in x_lower:
        x_unlist += i
    vocab_dic = Counter(x_unlist)
    stopwords_num = top_frequent_num

    print('      total vocab: ',len(vocab_dic.most_common()))
    maxfeature = len([i[0] for i in vocab_dic.most_common() if i[1]>word_least_frequency_num])
    print('      vocab size frequency >', word_least_frequency_num, ': ', maxfeature)

    stop = [i[0] for i in vocab_dic.most_common(stopwords_num)]

    x_n_level = [list(compress(x_lower, list(np.array(y_total)==i))) for i in np.unique(y_total)]

    x_n_level_unlist = [[] for i in range(len(np.unique(y_total)))]
    for i in range(len(x_n_level)):
        for j in x_n_level[i]:
            x_n_level_unlist[i] += j

    multilevel_vocab = []
    for i in range(len(np.unique(y_total))):
        multilevel_vocab.append(Counter(x_n_level_unlist[i]))

    multilevel_stop = defaultdict(list)
    for i in range(len(np.unique(y_total))):
        tt = len(x_n_level_unlist[i])
        for j in stop:
            multilevel_stop[j].append(multilevel_vocab[i][j]/tt)

    stop_var = [(key,np.std(value)*scale_v) for key,value in multilevel_stop.items() ]
    stop = [i[0] for i in stop_var if i[1]<stop_v]
    return(stop,maxfeature)

def tokenize(text):
    try:
        punctuation = string.punctuation.replace('#','')
        regex = re.compile('[' +re.escape(punctuation) +']')
        text = regex.sub(" ", text) # remove punctuation
        text = text.replace('#1','')
        text = text.replace('#2','')
        text = text.replace('#3','')
        text = text.replace('#4','')
        text = text.replace('#5','')
        text = text.replace('#6','')
        text = text.replace('#7','')
        text = text.replace('#8','')
        text = text.replace('#9','')
        ps = PorterStemmer()
        tokens = []
        tokens_ = [s.split() for s in sent_tokenize(text)]
        for token_by_sent in tokens_:
            tokens += token_by_sent
        filtered_tokens = [ps.stem(w.lower()) for w in tokens]
        return filtered_tokens
    except TypeError as e: print(text,e)



print('Read Data......')
train = pd.read_csv('training_data.csv',header= 0 ,delimiter='\t|\n')

print('Generate Interaction Features Between Categorical Features......')
s = time.time()
cate_list = ['age_cat','sex','stay_cat','lang','er','category']
generate_interaction(cate_list)
print('\ntime elapsed: ', time.time()-s,'\n')

x_total = list(train.comment)
y_total = list(train.score)

print('Calculate Uppercase Probability to Features......')
s = time.time()
upper_p = upper_prob(x_total)
new_up = scale(upper_p)
print('time elapsed: ', time.time()-s,'\n')

print('Choose Specific Stop Words and Max Text Feature Number......')
s=time.time()
stop, maxfeature = stop_and_max_feature(x_total,250,0,1000,0.1)
print('time elapsed: ', time.time()-s,'\n')

# Cate pipeline
onehot_list = ['age_cat', 'sex', 'stay_cat', 'lang', 'er','age_cat_sex', 'age_cat_stay_cat',
       'age_cat_lang', 'age_cat_er', 'age_cat_category', 'sex_stay_cat',
       'sex_lang', 'sex_er', 'sex_category', 'stay_cat_lang', 'stay_cat_er',
       'stay_cat_category', 'lang_er', 'lang_category', 'er_category']
onehot_pipeline = make_pipeline(FeatureExtractor(onehot_list),
                                CategoricalEncoding('OneHot'),
                                )

print('-------Transform to Features(word tf-idf level)-------')
s=time.time()
comment_word_tfidf_pipeline = make_pipeline(FeatureExtractor('comment'),
                                text(method='tfidf', ngram = 3, max_f = maxfeature,
                                     binary = False, stopwords=stop,tokenizer=tokenize,analyzer ='word'))

feature_union_word_tfidf = make_union(
    onehot_pipeline,
    comment_word_tfidf_pipeline
)
X_word_tfidf = feature_union_word_tfidf.fit_transform(train)
print('time elapsed: ', time.time()-s)

print('      Add Length and Upper Prob to Features......')
s = time.time()
length = [(X_word_tfidf[i,]!=0).sum() for i in range(X_word_tfidf.shape[0])]
new_l = scale(length)
X_word_tfidf = hstack([X_word_tfidf,new_l],format='csr')
X_word_tfidf = hstack([X_word_tfidf,new_up],format='csr')

print('      X_word_tfidf shape: ',X_word_tfidf.shape)
print('      time elapsed: ', time.time()-s)

print('-------Word TF-IDF Data Complete-------')



x_train, x_test, y_train, y_test = train_test_split(X_word_tfidf, y_total, test_size=0.33, random_state=109)


print('-------Ridge-------')
s = time.time()
# rkf = RepeatedKFold(n_splits=5, n_repeats=2)
# parameters = {'alpha':[1,3,5]}
# mse_score = make_scorer(mean_squared_error)
# ridge_model = Ridge()
# ridge_cv = GridSearchCV(ridge_model, parameters,cv=rkf, pre_dispatch=4, return_train_score = True,scoring=mse_score)
# ridge_cv.fit(x_train, y_train)
# print(ridge_cv.cv_results_)
# ridge_train_res = ridge_cv.predict(x_train)
# ridge_test_res = ridge_cv.predict(x_test)
ridge_model = Ridge(alpha = 1.5)
ridge_model = ridge_model.fit(x_train, y_train)
ridge_train_res = ridge_model.predict(x_train)
ridge_test_res = ridge_model.predict(x_test)

ridge_train_res = [10 if i >10 else round(i) for i in ridge_train_res]
ridge_train_res = np.array([0 if i<0 else i for i in ridge_train_res])
ridge_test_res = [10 if i >10 else round(i) for i in ridge_test_res]
ridge_test_res = np.array([0 if i<0 else i for i in ridge_test_res])

print("train accuracy:", mean_squared_error(ridge_train_res, y_train))
print("test accuracy:", mean_squared_error(ridge_test_res, y_test))
print('time elapsed: ', time.time()-s)


print('-------SparseNN-------')
s = time.time()

def sparseNN():
    sparse_data = Input(shape=[x_train.shape[1]], dtype = 'float32', sparse = True, name='sparse_data')
    x = Dense(200 , kernel_initializer=he_uniform(seed=0) )(sparse_data)
    x = PReLU()(x)
    x = Dense(200 , kernel_initializer=he_uniform(seed=0) )(x)
    x = PReLU()(x)
    x = Dense(100, kernel_initializer=he_uniform(seed=0) )(x)
    x = PReLU()(x)
    x= Dense(1)(x)
    model = Model([sparse_data],x)
    optimizer = Adam(.001)
    model.compile(loss="mse", optimizer=optimizer)
    return model

BATCH_SIZE = 1000
epochs = 20

sparse_nn = sparseNN()

print("Fitting SPARSE NN model ...")

for ep in range(epochs):
    BATCH_SIZE = int(BATCH_SIZE*2)
    sparse_nn.fit(x_train, np.array(y_train).reshape(-1,1),
                      batch_size=BATCH_SIZE, epochs=1, verbose=10 )

gc.collect
nn_test_res = sparse_nn.predict(x_test, batch_size=len(y_test))
nn_train_res = sparse_nn.predict(x_train, batch_size=len(y_train))

nn_train_res = [i[0] for i in nn_train_res]
nn_train_res = [10 if i >10 else round(i) for i in nn_train_res]
nn_train_res = [0 if i<0 else round(i) for i in nn_train_res]
print("train accuracy:", mean_squared_error(nn_train_res, y_train))
nn_test_res = [i[0] for i in nn_test_res]
nn_test_res = [10 if i >10 else round(i) for i in nn_test_res]
nn_test_res = [0 if i<0 else round(i) for i in nn_test_res]
print("test accuracy:", mean_squared_error(nn_test_res, y_test))
print('time elapsed: ', time.time()-s)


print('-------SGD-------')
s = time.time()

from sklearn.linear_model import SGDRegressor
import sklearn
model_sgd=SGDRegressor(loss = "squared_epsilon_insensitive",                         n_iter = 600,                           penalty="l2",                           alpha=0.00000001,#                            penalty : str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
                         epsilon=0.0001,\
                         random_state = 30,\
                         shuffle = True)
model_sgd = model_sgd.fit(x_train,y_train)
sgd_train_res = model_sgd.predict(x_train)
sgd_train_res = [10 if i >10 else round(i) for i in sgd_train_res]
sgd_train_res = [0 if i<0 else round(i) for i in sgd_train_res]
print("train accuracy:", mean_squared_error(sgd_train_res, y_train))
sgd_test_res = model_sgd.predict(x_test)
sgd_test_res = [10 if i >10 else round(i) for i in sgd_test_res]
sgd_test_res = [0 if i<0 else round(i) for i in sgd_test_res]
print("test accuracy:", mean_squared_error(sgd_test_res, y_test))
print('time elapsed: ', time.time()-s)

print('-------SVM-------')
s = time.time()
model_svm=sklearn.svm.LinearSVR(epsilon=0,                                    tol=0.000001,                                    C=0.1,                                     loss="squared_epsilon_insensitive",                                     fit_intercept=True,                                     intercept_scaling=1.0,                                     dual=True, verbose=0,                                     random_state=30,                                     max_iter=10000)

model_svm = model_svm.fit(x_train,y_train)
svm_train_res = model_svm.predict(x_train)
svm_train_res = [10 if i >10 else round(i) for i in svm_train_res]
svm_train_res = [0 if i<0 else round(i) for i in svm_train_res]
print("train accuracy:", mean_squared_error(svm_train_res, y_train))

svm_test_res = model_svm.predict(x_test)
svm_test_res = [10 if i >10 else round(i) for i in svm_test_res]
svm_test_res = [0 if i<0 else round(i) for i in svm_test_res]
print("test accuracy:", mean_squared_error(svm_test_res, y_test))
print('time elapsed: ', time.time()-s)



import lightgbm as lgb
num_train, num_feature = x_train.shape

lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train, free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'mean_squared_error',
    'num_leaves': 30,
    'learning_rate': 0.05,
}
print('Start training...')

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_test,
               early_stopping_rounds=5)



gbm_res = gbm.predict(x_test)
gbm_res = [10 if i >10 else round(i) for i in gbm_res]
gbm_res = [0 if i<0 else round(i) for i in gbm_res]
print("test accuracy:", mean_squared_error(gbm_res, y_test))
