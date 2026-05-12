import spacy
import re
import markovify
import nltk
from nltk.corpus import gutenberg
import warnings

warnings.filterwarnings('ignore')

nltk.download('gutenberg')

# print(gutenberg.fileids())


#import novels as text objects
hamlet = gutenberg.raw('shakespeare-hamlet.txt')
macbeth = gutenberg.raw('shakespeare-macbeth.txt')
caesar = gutenberg.raw('shakespeare-caesar.txt')
# #print first 100 characters of each
# print('nRaw:n', hamlet[:100])
# print('nRaw:n', macbeth[:100])
# print('nRaw:n', caesar[:100])


#utility function for text cleaning
def text_cleaner(text):
    text = re.sub(r'--',' ',text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\b\d+(\.\d+)?\b', '', text)
    text = re.sub(r'\b[A-Z][a-zA-Z]{1,10}\.\s', '', text)
    text = re.sub(r'Enter .*?\.', '', text)
    text = re.sub(r'Exit .*?(?=[.!?])', '', text)
    text = re.sub(r'\s+([?.!,;:])', r'\1', text)
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r'\s+', ' ', text)
    return text

#remove chapter indicator
hamlet = re.sub(r'Chapter \d+', '', hamlet)
macbeth = re.sub(r'Chapter \d+', '', macbeth)
caesar = re.sub(r'Chapter \d+', '', caesar)

#apply cleaning function to corpus
hamlet = text_cleaner(hamlet)
caesar = text_cleaner(caesar)
macbeth = text_cleaner(macbeth)

#parse cleaned novels
nlp = spacy.load("en_core_web_sm")
print("Processing text...")
hamlet_doc = nlp(hamlet)
macbeth_doc = nlp(macbeth)
caesar_doc = nlp(caesar)

hamlet_sents = ' '.join([sent.text for sent in hamlet_doc.sents if len(sent.text) > 1])
macbeth_sents = ' '.join([sent.text for sent in macbeth_doc.sents if len(sent.text) > 1])
caesar_sents = ' '.join([sent.text for sent in caesar_doc.sents if len(sent.text) > 1])

shakespeare_sents = (
    hamlet_sents + " " +
    macbeth_sents + " " +
    caesar_sents
)

# print(shakespeare_sents)

#create text generator using markovify
generator_1 = markovify.Text(shakespeare_sents, state_size=2)

#We will randomly genrate five sentences
print("\nGenerated Sentences:\n")
for i in range(5):
    sentence = generator_1.make_sentence()

    if sentence:
        print(sentence)

# #We will randomly genrate three sentences but no more than 100 chars
# print("\nShort Sentences:\n")
# for i in range(3):
#     print(generator_1.make_short_sentence(max_chars=100))

#spacy's part of english
class POSifiedText(markovify.Text):

    def word_split(self, sentence):
       return ['::'.join((word.orth_, word.pos_)) for word in nlp(sentence)]
   
    def word_join(self, words):
      sentence = ' '.join(word.split('::')[0] for word in words)
      return sentence 
    
#call the class on our text
generator_2 = POSifiedText(shakespeare_sents, state_size=2)

#now we will use the above generator to generate sentences
print("\nPOS-based Generated Sentences:\n")
for i in range(5):
  sentence = generator_2.make_sentence()

  if sentence:
    print(sentence)

#print 100 characters or less sentences
print("\nPOS-based Short Sentences:\n")
for i in range(5):
  sentence = generator_2.make_short_sentence(max_chars=100)

  if sentence:
    print(sentence)
