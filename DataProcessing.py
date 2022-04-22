import pandas as pd # for dataframe and csv

stopwords = {"ourselves", "hers", "between", "yourself", "again", "there", "about", "once", "during",
             "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours",
             "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from",
             "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through",
            "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while",
             "above", "both", "up", "to", "ours", "had", "she", "all", "when", "at", "any", "before", "them",
             "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what",
             "over", "why", "so", "can", "did", "now", "under", "he", "you", "herself", "has", "just", "where",
             "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "being", "if", "theirs","my",
             "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than", "ve", "ll"}

def clean_text(text):
    text = text.lower() # change to lowercase
    text = text.replace("<br />",'') # remove the balise
    non_alph = set(text) - set(" abcdefghijklmnopqrstuvwxyz") # set of all non alph in text
    for char in non_alph:
        text = text.replace(char, ' ') # remove all non alphabetical character in the string
    
    text = text.split()
    text = " ".join([word for word in text if word not in stopwords])
    return text

def clean_data(df):
    """
    This function will remove all non alphabetical character in the dataframe
    """
    df_copy = df.copy() 
    df_copy.iloc[:,0] = df.iloc[:,0].apply(clean_text) 
    return df_copy

def get_unique_words( data_frame ) :
    """
    This function will return all words from the input dataframe
    """
    unique_word = set()
    for doc in data_frame.iloc[:,0]:
        doc_split = doc.split()
        unique_word_doc = set(doc_split)
        for word in unique_word_doc:
            unique_word.add(word)
    return list(unique_word)

  
def one_hot_encoding(text_data_frame, list_of_all_words):
    """
    This function take imput a text dataframe and return a binary trasformation 
    """
    mat = []
    for doc in text_data_frame.review:
        doc_vec = []
        for word in list_of_all_words:
            value = 1 if word in doc else 0
            doc_vec.append(value)
        mat.append(doc_vec)
    df = pd.DataFrame(data = mat, columns = list_of_all_words)
    return df
