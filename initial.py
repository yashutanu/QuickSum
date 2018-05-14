import os
import sys
import global_var as gv
from nltk.tokenize import sent_tokenize
import lstm_keras as dn
from keras.models import model_from_json
import classifier as cs
import sentence as sc
from documentPreprocessing import document_Processing
from lexChain import LexicalChain

def getFileSentenceList(pathname):
    result=[]
    filelist=[]
    for file in os.listdir(pathname):
            if file.endswith(".txt"):
                with open(pathname+"/"+file,'r') as f:
                        sentences=sent_tokenize(f.read())
                        for i in sentences:
                            result.append(i.translate(gv.table))
                            filelist.append(file)
    return result,filelist

def clusterWrapper(pathname):
    for file in os.listdir(pathname):
            if file.endswith(".txt"):
                result=[]
                with open(pathname+"/"+file,'r') as f:
                        sentences=sent_tokenize(f.read())
                        for i in sentences:
                            result.append(i.translate(gv.table))
                final_res=cs.clusterModel(result)
                clusterSummary(final_res,file)
    
def SVCWrapper(sentencelist,filelist):
    test=cs.linear_test_SVC(sentencelist)
    result=list(zip(test,filelist,sentencelist))
    finalres={}
    for i in result:
            if i[0]==1:
                if i[1] not in finalres:
                    finalres[i[1]]=[]
                    finalres[i[1]].append(i[2])
                else:
                    finalres[i[1]].append(i[2])
    return finalres

def classifierSummary(result):
    os.chdir(sys.argv[1])
    if not os.path.exists("classifier"):
        os.mkdir("classifier")
    os.chdir("classifier")            
    for i,j in result.items():
        with open("classifier_"+i,'w') as f:
                for k in j:
                    f.write(k+"\n")
    for i,j in result.items():
        with open("classifier_"+i,'a') as f:
                f.write("\n\n#####sentences in shorter form:#####\n")
                for k in j:
                    sc.result_store=""
                    lt=list(sc.english_parser.raw_parse(k+"."))
                    short=sc.traverse(lt[0])
                    f.write(short+"\n")
    os.chdir("..")
    os.chdir("..")        
     
def clusterSummary(result,filename):
        os.chdir(sys.argv[1])
        if not os.path.exists("cluster"):
            os.mkdir("cluster")
        os.chdir("cluster")
        with open("cluster_"+filename,'w') as f:
                for k in result:
                    f.write(k+"\n")
        with open("cluster_"+filename,'a') as f:
                f.write("\n\n#####sentences in shorter form:#####\n")
                for k in result:
                    #convert sentence into short form
                    sc.result_store=""
                    lt=list(sc.english_parser.raw_parse(k+"."))
                    short=sc.traverse(lt[0])
                    f.write(short+"\n")
        os.chdir("..")
        os.chdir("..")

def neuralSummary(result):
    os.chdir(sys.argv[1])
    if not os.path.exists("neural"):
        os.mkdir("neural")
    os.chdir("neural")            
    for i,j in result.items():
        with open("neural_"+i,'w') as f:
                for k in j:
                    f.write(k+"\n")
    for i,j in result.items():
        with open("neural_"+i,'a') as f:
                f.write("\n\n#####sentences in shorter form:#####\n")
                for k in j:
                    sc.result_store=""
                    lt=list(sc.english_parser.raw_parse(k+"."))
                    short=sc.traverse(lt[0])
                    f.write(short+"\n")
    os.chdir("..")
    os.chdir("..")      
    
def getFiles(root):
    files = [filename for filename in os.listdir(root) if filename.endswith(".txt")]
    return files
    
def lexicalChainWrapper(files,dirname):
    os.chdir(dirname)
    percentage=0.4
    for file in files:
        wordList = []
        chains = []
        sentences = []
        doc = document_Processing()
        sentences=doc.preprocessing(file,dirname)
        wordList = doc.pickNounAndLemmatize(sentences)
        lex = LexicalChain()
        chains=lex.assignChain(wordList,chains,sentences,percentage,dirname,file)   
        
    
def neuralNetworkWrapper(result,filelist):
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
        result1=list(zip(rounded,filelist,result))
        finalres={}
        #print(result1)
        for i in result1:
            if i[0]==1:
                if i[1] not in finalres:
                    finalres[i[1]]=[]
                    finalres[i[1]].append(i[2])
                else:
                    finalres[i[1]].append(i[2])
        return finalres

                 
if len(sys.argv)>2:
    # SVC method
    
    if sys.argv[2] == 'classifier':
            result,filelist=getFileSentenceList(sys.argv[1])
            final_result=SVCWrapper(result,filelist)
            classifierSummary(final_result)
            
    elif sys.argv[2] == 'clustering':
            clusterWrapper(sys.argv[1])
           
    elif sys.argv[2] == 'lexchain':
            files=getFiles(sys.argv[1])
            lexicalChainWrapper(files,sys.argv[1])
            
    elif sys.argv[2] == 'neuralnetwork':                          
            # load json and create neural network model
            result,filelist=getFileSentenceList(sys.argv[1])
            final=neuralNetworkWrapper(result,filelist)
            neuralSummary(final)
    
else:
    print("Please provide pathname of a directory")

