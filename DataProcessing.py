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

