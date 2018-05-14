import os
from flask import Flask, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
import os
import sys
import global_var as gv
from nltk.tokenize import sent_tokenize
import lstm_keras as dn
from keras.models import model_from_json
import classifier as cs
import sentence as sc
from documentPreprocessing_v2 import document_Processing
from lexChain_v2 import LexicalChain

def getSentenceList(content):
    result=[]
    sentences=sent_tokenize(content)
    for i in sentences:
        result.append(i.translate(gv.table))
    return result

def clusterWrapper(content):
    result=getSentenceList(content)
    final_res=cs.clusterModel(result)
    return final_res
    
def SVCWrapper(content):
    sentencelist=getSentenceList(content)
    test=cs.linear_test_SVC(sentencelist)
    result=list(zip(test,sentencelist))
    finalres=[]
    for i in result:
            if i[0]==1:
                    finalres.append(i[1])
    return finalres


def createSummary(result):
    finalres=""     
    if len(result)>0:     
            for i in result:
                    finalres+=(i+"\n")
            finalres+="\n\n#####sentences in shorter form:#####\n"
            for i in result:
                    sc.result_store=""
                    lt=list(sc.english_parser.raw_parse(i+"."))
                    short=sc.traverse(lt[0])
                    finalres+=(short+"\n")   
    return finalres
   
def lexicalChainWrapper(content):
    percentage=0.4
    wordList = []
    chains = []
    sentences = []
    doc = document_Processing()
    sentences=doc.preprocessing(content)
    wordList = doc.pickNounAndLemmatize(sentences)
    lex = LexicalChain()
    chains=lex.assignChain(wordList,chains,sentences,percentage) 
    return chains  
        
    
def neuralNetworkWrapper(content):
        result=getSentenceList(content)
        X,Y=dn.prepareData(gv.sentenceTrainPath)
        tokenizer=dn.getTokenizer(X)
        json_file = open('model_final.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_final.h5")
        print("Loaded model from disk")
        maxlength=gv.TrainingParameter["maxlength"]
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        testdata=tokenizer.texts_to_sequences(result)
        test =dn.pad_sequences(testdata, maxlen=maxlength,padding="post")
        predicted_output=loaded_model.predict(test)
        rounded = [int(round(x[0])) for x in predicted_output]
        result1=list(zip(rounded,result))
        finalres=[]
        #print(result1)
        for i in result1:
            if i[0]==1:
                finalres.append(i[1])
        return finalres

                 
app = Flask(__name__)

APP_ROOT=os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)

@app.route("/",methods = ['GET','POST'])
def hello():
     return render_template(
        'front.html')
 
@app.route('/uploader', methods = ['GET','POST'])
def upload_file():
     if request.method == 'POST':  
            #file = request.files['file']
            target=os.path.join(APP_ROOT,'files/')
            #print(file.filename)
            print(target)
            if not os.path.isdir(target):
                os.mkdir(target)
            f1= request.files.getlist("file")
            if len(f1)>0:
                f=f1[0]
                destination='/'.join([target,f.filename])
                print(destination)
                f.save(destination)
                with open(destination) as fileobj:
                    data=fileobj.read()
            else:
                data=request.form["data"]
            print(data)
            s4=clusterWrapper(data)
            s3=SVCWrapper(data)
            s2=lexicalChainWrapper(data)
            s1=neuralNetworkWrapper(data)
            ss1=createSummary(s1)
            ss2=createSummary(s2)
            ss3=createSummary(s3)
            ss4=createSummary(s4)
            os.remove(destination)
            return render_template('front.html',content=data,summary1=ss1,summary2=ss2,summary3=ss3,summary4=ss4)
      
if __name__ == "__main__":
    app.run()
