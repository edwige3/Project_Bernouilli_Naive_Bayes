from DataProcessing import *
import numpy as np
import pandas as pd

class Bernoulli_Naive_Bayes_Classifier:
    def __init__(self, PATH_proba_from_csv_to_load = None):
        if PATH_proba_from_csv_to_load != None:
            self.__probabilities = pd.read_csv(PATH_proba_from_csv_to_load)
            self.__list_of_words = list(self.__probabilities.columns)
            self.__list_of_words.remove("_C_")

        else:
            self.__list_of_words = None
            self.__probabilities = None


    @property
    def list_of_words(self):
        return self.__list_of_words
    @property
    def probabilities(self):
        return self.__probabilities 
    

    def fit(self, train_DataFrame, use_laplacian = False):
        # removing the non alphabetical charcters in the data
        df_cleaned_data = clean_data(train_DataFrame.copy())
        # recuperation of all words in the training data
        self.__list_of_words = get_unique_words(df_cleaned_data)
        # recupering the encoding function
        # encoding the data using the one hot encoding function
        encoded_data = one_hot_encoding(df_cleaned_data, self.__list_of_words)

        # recupering the data of each classes
        df_positive_class = encoded_data[df_cleaned_data.iloc[:,-1] == "positive"]
        df_negative_class = encoded_data[df_cleaned_data.iloc[:,-1] == "negative"]
        
        # counting the number of datapoints we have for each class (1 for positive and 0 for negative)
        N_C1 = df_positive_class.shape[0]
        N_C0 = df_negative_class.shape[0]

        # alpha is a variable will switch on/off the laplacian smoothing
        alpha = 1 if use_laplacian else 0

        # creating a dataframe to store the probability of predictors xi's given classes C_k
        p_xi_Ck = pd.DataFrame()

        # computing P(xi | C_k) for each predictor xi and for each class Ck = 0 (negative) and Ck = 1 (positive)
        for xi in self.__list_of_words:
            p_xi_C0 = (alpha + df_negative_class[xi].sum())/(2*alpha + N_C0)
            p_xi_C1 = (alpha + df_positive_class[xi].sum())/(2*alpha + N_C1)
            p_xi_Ck[ xi ] = [p_xi_C0, p_xi_C1] # the first row will be for C0 and the second for C1
        
        # computing the probability for each class
        P_C0 = (alpha + N_C0)/(2*alpha + N_C0 + N_C1)
        P_C1 = (alpha + N_C1)/(2*alpha + N_C0 + N_C1)
        p_xi_Ck["_C_"] = [P_C0, P_C1] # saving the proba of the classes on the column named "_C_"

        # updating the probabilities
        self.__probabilities = p_xi_Ck
    
    def predict(self, test_DataFrame, inplace = False):
        # preprocessing the data
        cleaned_data = clean_data(test_DataFrame)
        # encoding the cleaned data
        encoded_data = one_hot_encoding(cleaned_data, self.__list_of_words)

        # counting the number of documents to predict
        n_docs = encoded_data.shape[0]

        # creating the list to store the predicted classes for all documents
        result = [0]*n_docs 

        for idx_doc in range(n_docs):
            p0, p1 =  self.__probabilities._C_ # recupering the proba of classes
            # we will use log-proba to avoid underflowing
            log_p_doc_C0 = np.log(p0)
            log_p_doc_C1 = np.log(p1)
            for word in self.__list_of_words:
                xi = encoded_data[word].iloc[idx_doc] # xi = 1 if word appears in document of index idx_doc, 0 otherwise
                pi0 = self.__probabilities[word].iloc[0] # pi0 is the proba of word appear given class 0
                pi1 = self.__probabilities[word].iloc[1] # pi1 is the proba of word appear given class 1
                log_p_doc_C0 += np.log( pi0 if xi == 1 else 1 - pi0) # add to the log proba of the doc to be class 0
                log_p_doc_C1 += np.log( pi1 if xi == 1 else 1 - pi1) # add to the log proba of the doc to be class 1
            # we make the prediction based on the log_likelihood of the doc to be in the given class
            pred = "positive" if log_p_doc_C1 > log_p_doc_C0 else "negative"
            result[idx_doc] = pred # storing the prediction following the order of docmuments

        # if inplace is true then create a column named prediction and store predictions there
        # otherwise return predictions
        if inplace:
            test_DataFrame["prediction"] = result
        else:
            return result

    
    def confusion_matrix(self, DataFrame_to_test):
        C_true = DataFrame_to_test.iloc[:, -1].values
        C_pred = self.predict(DataFrame_to_test, inplace = False)
        TP, FN, FP, TN = 0, 0, 0, 0
        num = len(C_pred)
        for i in range(num):
            if C_pred[i] == "positive":
                if C_true[i] == "positive":
                    TP += 1
                else:
                    FP += 1
            else:
                if C_true[i] == "positive":
                    FN += 1
                else:
                    TN += 1
        return {"TP":TP, "FN":FN, "FP":FP, "TN":TN}
