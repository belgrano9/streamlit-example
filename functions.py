#*********************************************************************


# This archive could be a potential first stone of the project.
# Now contains only functions used throughout the files, but 
# in the future could contain more complex structures.  


#*********************************************************************


import pdfplumber
import docx2txt
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, models,util





def reading_word(string):
    text = docx2txt.process("var.docx")
    return text

def reading_pdf(string):
    all_text=""
    with pdfplumber.open(string) as pdf:
        for pdf_page in pdf.pages:
            bold=pdf_page.filter(lambda obj: not(obj["object_type"] == "char" and obj["size"]>=10 ))
            single_page_text = bold.extract_text(x_tolerance=2)   
        #print( single_page_text )
        # separate each page's text with newline
            all_text = all_text + '\n' + single_page_text
    return all_text


def reading_file(string):
    """"
    -----------------------------------------------------------------------------
    
    This function takes as arguments the file that we want to analyze. Depending the file type we use some python library. 
    For the moment we detect only: PDF and Words.

    Returns: Long string with all the sentences in the document

    -----------------------------------------------------------------------------
    
    Input:

    string: path of the file we want to analyze

    """

    ext = os.path.splitext(string)[-1].lower()
    if ext == ".pdf":
        text=reading_pdf(string)
    elif ext == ".docx":
        text=reading_word(string)
    else:
        print ("Unknown file format.")
    return text

def filtering(text):
    """"
    -----------------------------------------------------------------------------
    
    This function takes as arguments the string obtained in the reading step and filters out undesired characters. 

    Potential things to filter: Index of contents, titles, formulas, references, tables (?) 
    
    
    Returns: Long string with all the sentences in the document.

    -----------------------------------------------------------------------------
    
    Input:

    string: string obtained in the previous reading step.

    """    
    clean1=re.sub("\d{1,}.\d{1,}.+","", text) #removing number of the table of contents
    clean1=re.sub("\w{1,} \w{1,} \.{4,} \d{1,}\d{1,}\n|\w{1,} \.{4,} \d{1,}\d{1,}\n|\w{1,} \w{1,} \w{1,} \.{4,} \d{1,}\d{1,}\n","",clean1) #removing number of the table of contents
    clean1=re.sub(" \n\d{1,} \n | \n\d{1,} \n \n |\d{1,}\. \w{1,} \w{1,}", "", clean1)
    clean1=re.sub("\.{4,} \d{1,}|\.{4,} Error! Bookmark not defined.", " ",clean1) #filtering the index
    clean1=re.sub("\n\n\n\n\n+|\n \n+", " ",clean1)#filtering long page jumps
    clean1=re.sub("\no |\n\uf0b7","",clean1)
    #clean1=re.sub(" \n"," ",clean1)
    return clean1

def everything_vs_word(query, corpus, model_name, number=5, score_function=util.cos_sim, ax=None):
    """"
    -----------------------------------------------------------------------------
    
    This function takes as arguments the text that we want to compare, the query with respect to we want to 
    compare, and then the number of comparisons we wanna show (by defect 5), the model used, and the metric used
    to compute the similarity (by defect cosine similarity).

    Returns: Histogram plot

    -----------------------------------------------------------------------------
    
    Input:

    query: String
    corpus: String or list of strings (usually the latter for a document --> list of sentences)
    number: Int
    model_name: String
    score_function: Function
    ax: Axis object

    """

    # model info retrieval
    model = SentenceTransformer(model_name)
    n=len(query)

    # tokenize according to the model 
    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)   

    # semantic search gives a list of lists composed of dictionaries
    hits = util.semantic_search(query_embedding, corpus_embedding,top_k=number,score_function=score_function)
    hits = hits[0]
    #print("Comparing ", query, " VS:")
    
    scoring=[]
    corp=[]
    for hit in hits:  
        #print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        scoring.append(hit['score'])
        corp.append(corpus[hit['corpus_id']])
    
    # defining dataframe for easiness in plotting
    data = pd.DataFrame(np.column_stack([corp, scoring]), 
                               columns=['Expression', 'Score'])
    data.sort_values(by=['Score'], ascending=False)
    data = data.explode('Score')
    data['Score'] = data['Score'].astype('float')

    return sns.barplot(data=data.reset_index(), ax=ax, x='Score', y='Expression')

# now we have a series of functions that do the same as the previous one but with some
# small differences, so they are potentially redundant.

def def_vs_syn(query, corpus, model_name, score_function, ax=None):

    #query = synonyms[10].split(", ")
    n=len(query)
    #corpus=definitions[10]

    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    scoring=[]
    for i in range(n):
        hits = util.semantic_search(query_embedding[i], corpus_embedding,top_k=30,score_function=score_function)
    #hits = hits[0]      #Get the hits for the first query
    #print("Comparing ", query[i], " VS:")
    #print(corpus, "(Score: {:.4f})".format(hits[0][0]["score"]))
        scoring.append(hits[0][0]["score"])

    query=np.array(query)
    data = pd.DataFrame(np.column_stack([query, scoring]),columns=['Query', 'Score'])
    data = data.explode('Score')
    data['Score'] = data['Score'].astype('float')
    data=data.sort_values(by=['Score'], ascending=False)

    return sns.barplot(data=data.reset_index(), ax=ax, x="Score", y="Query")

def sent_vs_def(query,corpus, model_name, score_function, ax=None):
    
    model=SentenceTransformer(model_name)

    n=len(query)

    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    scoring=[]
    hits = util.semantic_search(query_embedding, corpus_embedding,top_k=30,score_function=score_function)
    hits = hits[0]      #Get the hits for the first query
#print("Comparing ", query[i], " VS:")
    for hit in hits:
        #print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit["score"]))
        scoring.append(hit["score"])


    query=np.array(query)
    data = pd.DataFrame(np.column_stack([corpus, scoring]),columns=['Expression', 'Score'])
    sns.set(rc={'figure.figsize':(20, 10)})
    data = data.explode('Score')
    data['Score'] = data['Score'].astype('float')

    data=data.sort_values(by=['Score'], ascending=False)

    return sns.barplot(data=data.reset_index(), ax=ax, x="Score", y="Expression")

def sim(query, corpus, model_name, number=5, score_function=util.cos_sim):
    # model info retrieval
    model = SentenceTransformer(model_name)
    n=len(query)

    # tokenize according to the model 
    corpus_embedding = model.encode(corpus, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)   

    # semantic search gives a list of lists composed of dictionaries
    hits = util.semantic_search(query_embedding, corpus_embedding,top_k=number,score_function=score_function)
    hits = hits[0]
    #print("Comparing ", query, " VS:")
    
    scoring=[]
    corp=[]
    for hit in hits:  
        #print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
        scoring.append(hit['score'])
        corp.append(corpus[hit['corpus_id']])
    
    # defining dataframe for easiness in plotting
    data = pd.DataFrame(np.column_stack([corp, scoring]), 
                               columns=['Expression', 'Score'])
    data.sort_values(by=['Score'], ascending=False)
    data = data.explode('Score')
    data['Score'] = data['Score'].astype('float')
    return data


def sim_2(query, corpus, model_name, threshold,number=5, score_function=util.cos_sim):
    # with this function we will be able to perform the sentence similarity  and also modulate how to plot the results
    data=sim(query, corpus, model_name, number, score_function)
    # with the threshold number we will filter less important sentences
    return data


############ EXTRA BALL ################
# detecting the conclusion and getting all the sentences of that paragraph for future use.
def conclusion():
    return 
