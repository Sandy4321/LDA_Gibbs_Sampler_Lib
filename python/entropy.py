import nltk
import math
import string
import argparse
import numpy as np
import parallelLib as par
from multiprocessing import Queue
import multiprocessing as mp
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

#################add help document and argument here#################
parser = argparse.ArgumentParser(description='I really want to write a clear discription of this program to help users, but you know what, this is a HELP DOCUMENT!!!! If anyone can understand the documentation of software without going to any forums, then that must be bullshit. So, lets talk about how to use this command: #$%^&*(*&^%$#$%^&*(&^%$#$%^&*(&^%$ ')
parser.add_argument('dir',help=' Directory of training texts',type=str)
parser.add_argument('dirT',help=' Directroy of testing texts',type=str)
parser.add_argument('train',help='Number of documents in training sets',type=int)
parser.add_argument('test',help='Number of documents in testing sets',type=int)
args = parser.parse_args()


####################################################################
print args.dir, args.dirT, args.train, args.test

numOfTrain =args.train
numOfTest = args.test

def get_tokens(i,dirc):
    with open(dirc + str(i),'r') as shakes:
         text = shakes.read()
         tokens = nltk.word_tokenize(text)
         return tokens
def get_corpus(i,dirc):
    with open(dirc + str(i),'r') as corpus:
         text = corpus.read()
         return text

def tf_idf(word,docidx):
    length = np.zeros((1,numOfTrain+20))
    occur = np.zeros((1,numOfTrain+20))        
    P = np.zeros((1,numOfTrain+20))
    PLOGP = np.zeros((1,numOfTrain+20))
    for i in range(docidx-1,docidx +15):
        tokens = get_tokens(i+1,str(args.dir)+'/')
        length[0,i] = len(tokens)
        occur[0,i] = tokens.count(word)
        P[0,i] = (occur[0,i]+1)/(length[0,i] + numOfTrain)
        PLOGP[0,i] = P[0,i]*np.log2(P[0,i])
    entropy = -np.tanh(PLOGP.sum()*occur.sum())
    return entropy
    print word
    print entropy
        
        
        

filted = []
tokenTupleList = ()
print('Gathering terms, please wait......')



def parFilter(taskList,q,id,upper):
    print('Worker %d starting.....' %id )
    counter = 0
    tokenList = []
    while counter < upper:
      for i in taskList:
          if counter >= upper:
              break
          
          tokens = get_tokens(i+1,str(args.dir)+'/')
          listTemp = list(enumerate(tokens))
          for term in listTemp:
              if counter >= upper:
                  break
              if term[1] not in tokenList and len(term[1])>2:
                 
                 entropy = tf_idf(term[1],i+1)
                 tokenList.append(term[1])
                 if 0.618 < entropy<1 and counter < upper:
                    counter += 1
                    filted.append(term[1])

    q.put(filted)
'''

for i in range(1,numOfTrain + 1):
    tokens = get_tokens(i,str(args.dir)+'/')
    listTemp = list(enumerate(tokens))
    for term in listTemp:
        if term[1] not in tokenList and len(term[1])>2:
           entropy = tf_idf(term[1],i)
           tokenList.append(term[1])
           if 0.04 < entropy < 0.618:
              filted.append(term[1])
              with open('keys','a') as fKey:
                   fKey.write(str(term[1])+'\t')

print('Writing keywords to file')
#with open('keys','w') as fKey:
 #    for i in range(1,len(filted)):
  #       fKey.write(str(filted[i])+'\t')

'''

if __name__=='__main__':
    num_core = 4
    taskList = []
    queueList = []
    for i in range(num_core):
        queueList.append(Queue())
    taskList = par.splitTask(range(4000),4)    
    p1 = mp.Process(target = parFilter, args = (taskList[0],queueList[0],1,1200000))
    p2 = mp.Process(target = parFilter, args = (taskList[1],queueList[1],2,1200000))
    p3 = mp.Process(target = parFilter, args = (taskList[2],queueList[2],3,1200000))
    p4 = mp.Process(target = parFilter, args = (taskList[3],queueList[3],4,1200000))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    s1 = set(queueList[0].get())
    s2 = set(queueList[1].get())
    s3 = set(queueList[2].get())
    s4 = set(queueList[3].get())
    keySet = s1|s2|s3|s4
    print len(s1)
    print len(s2)
    print len(keySet)
    with open('keys_big','w') as keyWritter:
        for i in keySet:
            keyWritter.write(str(i)+'\t')








