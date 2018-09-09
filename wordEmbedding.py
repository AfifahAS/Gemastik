import pandas as pd
import numpy as np
import mysql.connector
from tqdm import tqdm
import json
import re

def WordEmbedding(word,cursor):
    sql = "select vec from term where term like %s;"
    cursor.execute(sql, (str(word),))
    lst = cursor.fetchall()
#    print (word)
    if len(lst) > 0 :
#        decoded_vec = json.JSONDecoder().decode(data[0][0])
#        vec = np.asarray(lst, dtype=np.float32)
        lst = lst[0][0].replace('"','')
        lst = lst.replace(',','')
        lst = lst.replace('[','')
        lst = lst.replace(']','')
        lst = re.split(' ',lst)
#       print (lst)
        vec = []
        for i in lst:
            vec.append(float(i))
#       print ("wordembedding")
        return True, vec
    else:
        return False, lst
    
def Tokenizing(sentence, lower=True):
    term = sentence.split(" ")
    remover = re.compile("[^a-zA-Z-]")
    token = []
    for i in term:
        word = remover.sub("",i)
        if lower == True:
            word = word.lower()
        token.append(word)
#    tokenized = filter(None, token)
#    print (len(tokenized))
    return token

def SentenceToVec(sentence):
    if type(sentence) is not str:

        return np.zeros((300,),dtype=float)
    sentence = sentence.replace("\n","")
    word_array = np.array(Tokenizing(sentence))
    begin = True
#    print ("sentencetovec")
    for word in word_array:
        stat, vec = WordEmbedding(word, cursor)
#        print (str(stat) + "\n" + str(vec))
        if not stat:
#            print ("\continue")
            continue
        if begin:
            begin = False
            feature = vec
#            print ("begin=false")
        else:
#            print ("feature+=vec")
            feature+=vec
    if feature is not None:
        feature = feature/np.linalg.norm(feature)
    else:
        feature = np.zeros((300,),dtype=float)
    return feature

data_path = "Data/"
filename = "stream.csv"

train_df = pd.read_csv(data_path + filename, delimiter = ",")
train_df.head()

instansi_list = train_df["DisposisiInstansi"].value_counts()
instansi = train_df["DisposisiInstansi"].values

#with pd.option_context('display.max_rows',None,'display.max_columns',None):
#    print (instansi_list)
    
data = train_df["IsiLaporan"].values

db = mysql.connector.connect(user="root",password="",database="glove-300")
cursor = db.cursor(buffered=True)

pbar = tqdm(total=len(data))
ftr = []
for i in data:
    try:
#        print("\nbisa")
        ftr.append(SentenceToVec(i))
    except Exception as err:
        print ("\nError : " + str(err))
        print (i)
        break
    pbar.update(1)
pbar.close()
ftr = np.array(ftr)

nan_idx = []
for i in range(len(instansi)):
    if instansi[i] is np.nan:
        nan_idx.append(i)
instansi_c = np.delete(instansi, nan_idx)
ftr_c = np.delete(ftr, nan_idx, 0)

cls = []
instansi_u = pd.unique(instansi_c)
instansi_u = instansi_u.tolist()
for i in range(len(instansi_c)):
    one_hot = np.zeros((len(instansi_u),),dtype=int)
    idx = instansi_u.index(instansi_c[i])
    one_hot[idx] = 1
    cls.append(one_hot)
cls = np.array(cls)

print ("Cek size array fitur dan array kelas")
print ("Size fitur : " + str(ftr_c.shape))
print ("Size class : " + str(cls.shape))

np.save(data_path + "ftr-data-source.npy", ftr_c)
np.save(data_path + "cls-data-source.npy", cls)
