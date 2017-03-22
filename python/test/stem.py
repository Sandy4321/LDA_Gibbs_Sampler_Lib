import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def get_tokens(i):
    with open(str(i),'r') as shakes:
         text = shakes.read()
         tokens = nltk.word_tokenize(text)
    return tokens

stem = PorterStemmer()
wnl = WordNetLemmatizer()
for i in range(1,4):
    tokens = get_tokens(i)
    listTemp = list(enumerate(tokens))
    for term in listTemp:
        wordLemmatized = wnl.lemmatize(term[1])
        wordStemmed = stem.stem(wordLemmatized)
        with open(str(i),'w') as writer:
             writer.write(str(wordStemmed)+'\t')
