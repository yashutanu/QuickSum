from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stopWords=set(stopwords.words('english'))
charReplaceMapping={'.':' ','-':'',"'":' ','\n':''}
table=str.maketrans(charReplaceMapping)
lemmatizer = WordNetLemmatizer()
sentenceTrainPath="/home/yashu/8TH_PRO/Testing/testing/neural_final_test/sentence_label_23_4.csv"
k_cluster=4
maxiter=100

""" acc=76.6
TrainingParameter={
    "embedding_size":120,
    "kernel_size":4,
    "filters":3,
    "pool_size":4,
    "lstm_output_size":70,
    "batch_size":60,
    "epochs":2
}
"""

TrainingParameter={
    "embedding_size":256,
    "kernel_size":4,
    "filters":4,
    "pool_size":4,
    "lstm_output_size":120,
    "batch_size":120,
    "epochs":5,
    "maxlength":126
}
