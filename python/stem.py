import nltk
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import argparse

parser = argparse.ArgumentParser(description='stem some documents')
parser.add_argument('fromdir',help='from direct',type=str)
parser.add_argument('todir',help='to direct',type=str)
parser.add_argument('number',help='number of documents',type=int)
args = parser.parse_args()

fromDir = args.fromdir + '/'
toDir = args.todir + '/'
number = args.number
def get_tokens(i):
    with open(fromDir + str(i),'r') as shakes:
         text = shakes.read()
         try:
             tokens = nltk.word_tokenize(text)
         except UnicodeError:
             print('Decode Error')
         else:
             pass
    return tokens

stem = PorterStemmer()
wnl = WordNetLemmatizer()
for i in range(1,number+1):
    print 'processing document :',i
    tokens = get_tokens(i)
    listTemp = list(enumerate(tokens))
    with open(toDir + str(i),'w') as writer:
         for term in listTemp:
             try:
                 wordStemmed = stem.stem(term[1])
             except IndexError:
                 print 'Error'
                 print term[1]
             else:
                 pass
             wordStemmed = re.sub(r'[^a-zA-Z\s\n]',' ',wordStemmed)
             writer.write(str(wordStemmed).lower()+'\t')
