import os
import time
import shutil
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from gensim.models import LsiModel
from gensim.similarities import Similarity
from sklearn.metrics.pairwise import cosine_similarity
from azure.storage.blob import BlobClient, ContainerClient
from utils.coo_311_utils import lsi_model_req, status_message
from preprocess_text import clean_text, create_trigram_corpus
from LSI_training_functions import train_lsi_cluster_model

pd.options.mode.chained_assignment = None

def import_lsi_model(model_folder_name, blob_keys):
    #####################
    # LOAD IN LSI MODEL
    #####################
    try:
        start_time_model = time.time()
        lsi_model_container = 'lsi-models'
        lsi_model_container_client = ContainerClient.from_connection_string(conn_str=blob_keys['storage_connection_string'], container_name=lsi_model_container)
        for blob in lsi_model_container_client.list_blobs():
            if model_folder_name in blob['name']:
                in_blob_client =  BlobClient.from_connection_string(conn_str=blob_keys['storage_connection_string'], container_name=lsi_model_container, blob_name=blob['name'])
                for folder in blob['name'].split('/')[:-1]:
                    if folder not in os.listdir("./lsi_model"):
                        os.mkdir("./lsi_model/" + folder)
                with open(os.path.join("./lsi_model/", blob['name']), "wb") as data:
                    data.write(in_blob_client.download_blob().readall())
        lsi = LsiModel.load("./lsi_model/" +  model_folder_name + "/lsi_model.model")
        status_message("Imported {} Model in {:.2f} seconds".format(model_folder_name, time.time() - start_time_model))
        start_time_req = time.time()
        with open("./lsi_model/" + model_folder_name + "/lsi_model_req.pk1", "rb") as req_pic:
            model_requisites = pickle.load(req_pic)
        dictionary = model_requisites.dictionary
        tfidf = model_requisites.tfidf_matrix
        status_message("Imported {} Model Requisites in {:.2f} seconds".format(model_folder_name, time.time() - start_time_req))
        shutil.rmtree(os.path.join("./lsi_model", model_folder_name))
        return lsi, dictionary, tfidf
    except Exception as e:
        status_message("Could not Import LSI Model: {}".format(model_folder_name))
        print(e)
    return None, None, None


def import_cluster_lsi_model(model_folder_name, blob_keys):
    #####################
    # LOAD IN CLUSTER LSI MODEL
    #####################
    try:
        start_time_model = time.time()
        lsi_model_container = 'lsi-models'
        lsi_model_container_client = ContainerClient.from_connection_string(conn_str=blob_keys['storage_connection_string'], container_name=lsi_model_container)
        for blob in lsi_model_container_client.list_blobs():
            if model_folder_name in blob['name']:
                in_blob_client =  BlobClient.from_connection_string(conn_str=blob_keys['storage_connection_string'], container_name=lsi_model_container, blob_name=blob['name'])
                for folder in blob['name'].split('/')[1:-1]:
                    if folder not in os.listdir("./lsi_model/LSI_Clustering_Approach"):
                        os.mkdir("./lsi_model/LSI_Clustering_Approach/" + folder)
                with open(os.path.join("./lsi_model/", blob['name']), "wb") as data:
                    data.write(in_blob_client.download_blob().readall())
        lsi = LsiModel.load("./lsi_model/" +  model_folder_name + "/lsi_model.model")
        status_message("Imported {} Model in {:.2f} seconds".format(model_folder_name, time.time() - start_time_model))
        start_time_req = time.time()
        with open("./lsi_model/" + model_folder_name + "/lsi_model_req.pk1", "rb") as req_pic:
            model_requisites = pickle.load(req_pic)
        dictionary = model_requisites.dictionary
        tfidf = model_requisites.tfidf_matrix
        status_message("Imported {} Model Requisites in {:.2f} seconds".format(model_folder_name, time.time() - start_time_req))
        cluster_vectors_df = pd.read_csv(os.path.join("./lsi_model", model_folder_name, "cluster_vectors_df.csv"), index_col = 'Unnamed: 0')
        weights_df = pd.read_csv(os.path.join("./lsi_model", model_folder_name, 'cluster_weights_df.csv'), header=None)[0]
        shutil.rmtree(os.path.join("./lsi_model", model_folder_name))
        return lsi, dictionary, tfidf, cluster_vectors_df, weights_df
    except Exception as e:
        status_message("Could not Import LSI Model: {}".format(model_folder_name))
        print(e)
    return None, None, None, None, None


def trim_test_set(df, trim_min_perc=0, trim_max_perc=1):
    if trim_min_perc != 0 or trim_max_perc != 1:    
        # status_message("Trimming Testing Sentence Set")
        # Index sentences by Blob
        df['phraseIndexByBlob'] = df.groupby(['blobName']).cumcount() + 1
        # trim_by_perc is True if want to trim by percentage
        status_message("Keeping from {}% and {}% of sentences per Blob".format(trim_min_perc*100, trim_max_perc*100))
        ## Percentage of Calls
        blob_sentence_count = df.groupby('blobName').count()['lexical']
        df['percent_cut'] = df[['blobName', 'phraseIndexByBlob']].apply(lambda x: 1 if (x[1] / blob_sentence_count[x[0]] >= trim_min_perc) and (x[1] / blob_sentence_count[x[0]] <= trim_max_perc) else 0, axis=1)
        return df[df['percent_cut'] == 1]
    else:
        # status_message("No Request to Trim Sentence Set")
        return df


def test_LSI_cluster_model(test_df, model_folder_name, test_text_column = '', test_conf_thresh = 0, train_df = pd.DataFrame(), train_text_column = '', write_new_model = True, num_topics = 0):
    # IMPORT MODEL
    cluster_models_path = "LSI_Clustering_Approach"
    model_base_path = os.path.join("./lsi_model", cluster_models_path)
    if model_folder_name in os.listdir(model_base_path):
        # status_message("Model Found. Importing Model from Folder '{}'".format(model_folder_name))
        lsi, dictionary, tfidf, cluster_vectors_df, weights_df = import_cluster_lsi_model(os.path.join(model_base_path, model_folder_name), blob_keys)
    else:
        status_message("No Model Found in '{}'." .format(model_folder_name))
        if train_df.empty or train_text_column == '':
            status_message("Please input proper training parameters: train_df (pd.DataFrame), train_text_column (string), write_model (boolean) and num_topics (int).")
            return None
        elif num_topics <= 0:
            status_message("num_topics must be greater than 0")
            return None
        else:
            status_message("Training Custom Model with Defined Parameters...")
            cluster_vectors_df, weights_df, dictionary, tfidf, lsi = train_lsi_cluster_model(train_df, train_text_column, num_topics, blob_keys = config['BLOB_KEYS'], write_model = write_new_model, custom_folder_name = model_folder_name)
    status_message("Imported Model")
    #########################
    # Calculate Distance (Cosine Similarity) Between New Vector and Each of those Found Major Clusters
    #########################
    weights_df = weights_df / sum(weights_df)
    result = {}
    if test_text_column == '' or test_df.empty:
        status_message("Improper Test DataFrame or Text Column when Running Against Model: {}. Please Check and Retry.".format(model_folder_name))
    for test_sentence_index in test_df[test_text_column].index:
        test_sentence = test_df[test_text_column].loc[test_sentence_index]
        # Gather Weights:
        text_df = clean_text(pd.DataFrame([test_sentence], columns = ["test_sentence"]), "test_sentence")
        ## CREATING CORPUS
        test_initial_corpus = text_df['clean_text'].apply(lambda x: x.split(' ')).values[0]
        if len(test_initial_corpus) < 3:
            continue
        test_trigram_corpus = create_trigram_corpus(test_initial_corpus)
        test_corpus = dictionary.doc2bow(test_trigram_corpus)
        test_corpus_tfidf = tfidf[test_corpus]
        test_corpus_lsi = lsi[test_corpus_tfidf]
        test_lsi_comparison = np.array([i[1] for i in test_corpus_lsi])
        test_lsi_comparison_df = pd.DataFrame([test_lsi_comparison], index = ["test"])
        if test_lsi_comparison_df.empty:
            continue
        test_lsi_comparison_df.columns = cluster_vectors_df.columns
        test_clusters_df = cluster_vectors_df.append(test_lsi_comparison_df)
        similarity_mat = cosine_similarity(test_clusters_df)
        similarity_row = pd.DataFrame(similarity_mat, columns = test_clusters_df.index, index = test_clusters_df.index).loc["test"]
        # print(test_sentence)
        # print(similarity_row)
        # print(weights_df)
        #########################
        # Aggregate Distance Metrics using Weights (Based on Size of Clusters). Use as "Correlation Confidence Measure"
        #########################
        final_confidence_level = 0
        weights_df.index = similarity_row.iloc[:-1].index
        for group in weights_df.index:
            final_confidence_level += weights_df.loc[group] * similarity_row.loc[group]
        if final_confidence_level > test_conf_thresh:
            result[test_sentence_index] = final_confidence_level
    # status_message("Returning LSI Results with Confidence Level Over Defined Threshold: {}".format(test_conf_thresh))
    return result



def test_blob_subset(dictionary, tfidf, lsi, test_df, text_col):
    # Choose Documents to Run Against
    if test_df is None:
        status_message("Unable to process. Please specific DataFrame with test blob transcription using the 'test_df' parameter")
        return None
    ## CLEAN TEXT
    text_df = clean_text(test_df, text_col)
    ## CREATING CORPUS
    initial_corpus = text_df['clean_text'].apply(lambda x: x.split(' ')).values
    trigram_corpus = create_trigram_corpus(initial_corpus)
    corpus = [dictionary.doc2bow(text) for text in trigram_corpus]
    corpus_modelled = tfidf[corpus]
    lsi_corpus = lsi[corpus_modelled]
    return lsi_corpus


def remove_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            status_message('Failed to delete %s. Reason: %s' % (file_path, e))

def filter_on_q_LSI(q_list, lsi, tfidf, dictionary, test_df, text_col, conf_thresh = 0, num_results=10, test_type = ''):
    if type(q_list) is not list:
        status_message("Test Failed. Please Pass Query as Type 'list'.")
        return []
    #####################
    # GATHER TEST CORPUS
    #####################
    ## Options for test_type: all_trained_blobs, blob_subset, raw_text 
    corpus_lsi = test_blob_subset(dictionary, tfidf, lsi, test_df, text_col)
    if corpus_lsi is None:
        return []
    #####################
    # PREPARE TEST INPUT
    #####################
    # status_message("Running LSI model...".format(len(q_list)))
    results = []
    for q in q_list:
        # "Folding in new documents"
        # status_message("Creating Test Corpus for Query: '{}'".format(q))
        # start_time_query = time.time()
        text_df = clean_text(pd.DataFrame([q], columns = ["test_sentence"]), "test_sentence")
        ## CREATING CORPUS
        test_initial_corpus = text_df['clean_text'].apply(lambda x: x.split(' ')).values[0]
        test_trigram_corpus = create_trigram_corpus(test_initial_corpus)
        test_corpus = dictionary.doc2bow(test_trigram_corpus)
        test_corpus_tfidf = tfidf[test_corpus]
        # Map new documents into the latent semantic space
        test_corpus_lsi = lsi[test_corpus_tfidf]
        # Run a query to check simiilarity - scalable version
        # status_message("Creating Similarity Matrix")
        index = Similarity('./similarities/sims', corpus_lsi, num_features=200)
        #####################
        # TEST SIMILARITY AND PRINT RESULTS
        #####################
        if conf_thresh == 0:
            # status_message("Printing Top {} Results".format(num_results))
            # status_message([i[0] for i in sorted(enumerate(index[test_corpus_lsi][0]), key=lambda item: -item[1])[:num_results]])
            # status_message("Returned List with {} Results".format(num_results))
            results.append(sorted(enumerate(index[test_corpus_lsi][0]), key=lambda item: -item[1])[:num_results])
        else:
            similarity_results = index[test_corpus_lsi]
            if len(similarity_results) == 1 and similarity_results[0] > 1:
                similarity_results = similarity_results[0]
            threshold_list = [i for i in enumerate(similarity_results) if i[1] > conf_thresh]
            # status_message("Number of Results with Confidence over {}%: {}".format(conf_thresh*100, len(threshold_list)))
            # status_message("Printing Top 10 Results such that the Correlation is OVER {}%".format(conf_thresh*100))
            # status_message(sorted(threshold_list, key = lambda item: -item[1])[:10])
            # status_message("Returned Threshold List")
            results.append(sorted(threshold_list, key = lambda item: -item[1]))
        remove_folder_contents('./similarities')
    # status_message("Query took {:.2f} seconds".format(time.time() - start_time_query))
    return results



# def write_threshold_csv(df, topic):
#     df = df['phrase']
#     write_folder =  "train_" + topic
#     write_file = "luis_df_" + topic + ".csv"
#     num_identify = 1
#     while write_file in os.listdir('./training_sentences/' + write_folder):
#         num_identify += 1
#         write_file = "luis_df_" + topic + str(num_identify) + ".csv"
#     if write_folder not in os.listdir("./training_sentences/"):
#         os.mkdir("./training_sentences/" + write_folder)
#     df.to_csv("./training_sentences/" + write_folder + "/" + write_file, index=False)


