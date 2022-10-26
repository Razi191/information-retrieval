import sys
import os
import xml.etree.ElementTree as ET
import nltk
import math
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

dictionary = {} # This dictionary holds all the words, the tf-idf for each file for every word.
words_per_file = {} # This dictionary holds the number of words after tokenization for every file. (key = file_name : value = integer)
query_dictionary = {}
document_reference = {} # This dictionary holds the length of all document vectors
corpus = {}# This dictionary holds the words dictionary ("dictionary") and the document_reference (dictionary of all files and their lengths)
katsha = []
max_word_per_record = {}

            ### PART 1: Inverted Index and TF-IDF score. ###

# Addin a document to document_reference dict
def insert_to_document(record_id):
    if record_id not in document_reference:
        document_reference[record_id] = 0


# Extract the needed text from file
def extract_words(filename):

    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    xml_tree = ET.parse(filename)
    root = xml_tree.getroot()
    for child in root.findall("./RECORD"):  #extracts all the text from file.

        #Initialize...
        text = ""
        filtered_text = []

        #Calculating...
        for entry in child:
            if entry.tag =="RECORDNUM":
                record_id = int(entry.text)
                insert_to_document(record_id)
            if entry.tag == "TITLE":
                text += str(entry.text)+" "
            if entry.tag == "ABSTRACT":
                text += str(entry.text)+" "
            if entry.tag == "EXTRACT":
                text += str(entry.text)+" "

        text = text.lower()
        text = tokenizer.tokenize(text)  #tokenize and filter punctuation.
        filtered_text = [word for word in text if not word in stop_words]  #remove stop words.

        for i in range(len(filtered_text)):  #stemming
            filtered_text[i] = ps.stem(filtered_text[i])

        update_dictionary(filtered_text, record_id)
        words_per_file[record_id] = len(text)
        update_record_words(filtered_text, record_id)



def update_record_words(filtered, record_id):
    d = {}
    max_val = 1
    max_word = ""
    for word in filtered:
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
            cnt = d[word]
            if cnt >= max_val:
                max_word = word
                max_val = cnt

    max_word_per_record[record_id] = max_val,max_word


# Insert to dictionary all the words that appear in the file: 'file_name'
def update_dictionary(text, file_name):
    for word in text:
        if word in dictionary:
            if dictionary.get(word).get(file_name):
                dictionary[word][file_name]["count"] += 1
            else:
                dictionary[word].update({file_name : {"count" : 1 , "tf_idf" : 0 , "bm25" : 0, "file_len" : 0}})
        else:
            dictionary[word] = {file_name : {"count" : 1 , "tf_idf" : 0, "bm25" : 0, "file_len" : 0}}


# tfidf calcultaing for each word for each file....
def tf_idf_calculate(amount_of_docs):
    for word in dictionary:
        for file in dictionary[word]:
            tf = dictionary[word][file].get('count')/max_word_per_record[file][0]
            idf = math.log2(amount_of_docs / len(dictionary[word]))
            dictionary[word][file]["tf_idf"] = tf*idf
            dictionary[word][file]["file_len"] = words_per_file.get(file)
            document_reference[file] += (tf*idf*tf*idf)

# Sqaure root of document vectors lengths
def lenof_rootdoc():
    for file in document_reference:
        document_reference[file] = math.sqrt(document_reference[file])

# Build inverted index with tf-idf score to all the words from the given files.
def create_index():
    input_dir = sys.argv[2]
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".xml"):
            file=input_dir+"/"+file_name
            extract_words(file)

    docs_num = len(document_reference)
    # adding tfidf weights and calculating vectors lengths
    tf_idf_calculate(docs_num)

    # Sqaure root of lengths in document_reference
    lenof_rootdoc()

    # Add dictionary and document_reference to corpus
    corpus["dictionary"] = dictionary
    corpus["document_reference"] = document_reference

    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent = 8)
    inverted_index_file.close()


            ### PART 2: Information Retrieval given a query. ###



def query():
    index_path = sys.argv[3]

    try:
        json_file = open(index_path,"r") # Open json file to read from it.
    except:
        print("Could not open given path of index: ",index_path)
        return

    vsm_file = json.load(json_file)
    inverted_index_words = vsm_file["dictionary"]
    document_vectors = vsm_file["document_reference"]
    amount_of_docs = len(document_vectors)
    json_file.close()

    query = new_simplified_query() # extracting stopwords etc

    if query == False:
        print("there is no query question in the input")
        return

    if (sys.argv[2] == "bm25"):
        query_bm25(query, inverted_index_words, document_vectors)
        relevant_docs = results2222(query, inverted_index_words, document_vectors)
        f = open("ranked_query_docs.txt", "w")
        for i in range(0, len(relevant_docs)):
            if (relevant_docs[i][1] >= 7.99):  ### <<<< not so sure
                f.write(relevant_docs[i][0] + "\n")

        f.close()
    if (sys.argv[2] == "tfidf"):

        query_tf_idf(query, inverted_index_words,amount_of_docs) # Calculate the tf_idf of the words in query and insert to global variable 'query_dictionary'.
        relevant_docs = results(query_dictionary, inverted_index_words, document_vectors)
        f = open("ranked_query_docs.txt", "w")
        for i in range(0, len(relevant_docs)):
            if (relevant_docs[i][1] >= 0.075):  ### <<<< not so sure
                f.write(relevant_docs[i][0] + "\n")

        f.close()






def query_max_word(query):
    d = {}
    max_val = 1
    for word in query:
        if word not in d:
            d[word] = 1
        else:
            d[word] += 1
            cnt = d[word]
            if cnt >= max_val:
                max_val = cnt
    return max_val


def new_simplified_query():
    try:
        query = sys.argv[4].lower()
    except:
        return False
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    query = tokenizer.tokenize(query) #tokenize and filter punctuation.
    new_query = [word for word in query if not word in stop_words]  #removing stop words.
    for i in range(len(new_query)): #stemming
        new_query[i] = ps.stem(new_query[i])
    return new_query



# calculate query's tfidf
def query_tf_idf(query, dictionary, amount_of_docs):

    for word in query:
        count = 0
        if query_dictionary.get(word) == None:
            for word22 in query:
                if word == word22:
                    count =count+ 1
            a = query_max_word(query)
            tf = (count / a)
            if dictionary.get(word) != None:
                idf = math.log2(amount_of_docs / len(dictionary.get(word)))
            else:
                idf = 0
            query_dictionary.update({str(word) : tf*idf})



def results(query_map, inverted_index, document_reference):
    results_return = []

    query_length = 0
    for token in query_map:
        query_length += (query_map[token]*query_map[token])
    query_length = math.sqrt(query_length)

    documents_vectors = cossim_calc(query_map, inverted_index)
    for doc in documents_vectors:
        doc_query_product = documents_vectors[doc]
        doc_length = document_reference[doc]
        cosSim = doc_query_product / (doc_length * query_length)
        results_return.append((doc, cosSim))

    # sorting list by cosSim
    results_return.sort(key = lambda x: x[1], reverse=1)
    return results_return


# Create hashmap of dj * q for all documents that include words from query
def cossim_calc(query_map, inverted_index):
    d = {}
    for word in query_map:
        c = query_map[word] # w = token's tf-idf (in query)
        if inverted_index.get(word) != None:
            for record in inverted_index[word]:
                if record not in d:
                    d[record] = 0

                d[record] += (inverted_index[word][record]["tf_idf"] * c)

    return d













def count2(word, inverted_index):
    c = 0
    for doc in inverted_index[word]:
        c = c +1
    return c

def files_length(inverted_index):
    d = {}
    c = 0
    for word in inverted_index:
        for record in inverted_index[word]:
            if record not in d:
                d[record] = 1
                c = c + inverted_index[word][record].get('file_len')
    return c




def query_bm25(query, inverted_index,document_reference):

    N = len(document_reference.keys())
    fileslen = files_length(inverted_index)


    for word in query:

        if inverted_index.get(word) != None:


            for doc in inverted_index[word]:

                c = count2(word, inverted_index)
                idfq = math.log((N+ 0.5 -c)/(c+0.5) +1)


                fqi = inverted_index[word][doc].get('count')


                d = inverted_index[word][doc].get('file_len')



                avg = fileslen/N

                M = idfq*(fqi*2.4)/(fqi + 1.4*(1-0.75+0.75*(d/avg)))




                inverted_index[word][doc]["bm25"] = M








def results2222(query, inverted_index,document_reference):

    results = []

    for record in document_reference:

        c = 0
        for word in query:
            if inverted_index.get(word):
                if inverted_index.get(word).get(record):
                    c = c + inverted_index[word][record]["bm25"]

        results.append((record, c))

    results.sort(key=lambda x: x[1], reverse=1)
    return results


# Call method 'create_index' or 'query' depends on command line arguments input.

def main():
    if (sys.argv[1] == "create_index"):
        create_index()

    elif (sys.argv[2] == "tfidf" or sys.argv[2] == "bm25"):
        query()




    else:
        print("Illegal Input! \n please insert correct instruction  :)")


main()
