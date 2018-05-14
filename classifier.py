from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
import global_var as gv
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

train=pd.read_csv(gv.sentenceTrainPath)
Y_train = list(train[train.columns[2]])
X_train = list(train[train.columns[1]])

X_train=[k.translate(gv.table) for k in X_train]

def linear_train_SVC(X_train,Y_train):
        vectorizer = TfidfVectorizer(stop_words='english')
        train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size=0.04,random_state=42 )
        train_vectors = vectorizer.fit_transform(train_x)
        #print(train_vectors[0])
        test_vectors = vectorizer.transform(test_x)
        classifier_linear = svm.SVC(kernel='linear')
        classifier_linear.fit(train_vectors, train_y)
        prediction_linear = classifier_linear.predict(test_vectors)
        #print(prediction_linear)
        #print(classification_report(test_y, prediction_linear))
        return classifier_linear,vectorizer
    
#linear_train_SVC(X_train,Y_train)
def linear_test_SVC(testdoc):
        global X_train
        global Y_train
        classifier,vectorizer=linear_train_SVC(X_train,Y_train)
        test_vectors=vectorizer.transform(testdoc)
        prediction_linear = classifier.predict(test_vectors)
        return prediction_linear    
    

def clusterModel(X):
    vectorizer=TfidfVectorizer(stop_words='english')
    train_vectors = vectorizer.fit_transform(X)
    model = KMeans(n_clusters=gv.k_cluster, init='k-means++', max_iter=gv.maxiter, n_init=1)
    res=model.fit_predict(train_vectors)
    dist=model.transform(train_vectors)
    #print(model.get_params())
    final=[]
    for i in dist:
        res1=[]
        for j  in i:
            res1.append(float(j))
        final.append(res1)
    final_out=list(zip(res,final,X))
    #print(final_out)
    final_dict={}
    for i in final_out:
        if i[0] not in final_dict:
            final_dict[i[0]]=[]
            final_dict[i[0]].append((i[2],i[1][i[0]]))
        else:
            final_dict[i[0]].append((i[2],i[1][i[0]]))
    final_res=[]
    for i in final_dict:
            min_dist=min(final_dict[i],key=lambda x:x[1])
            final_res.append(min_dist[0])
    return final_res

