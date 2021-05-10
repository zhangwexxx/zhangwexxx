import os
import time
import shutil
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel
from azure.storage.blob import BlobClient, ContainerClient
from convert_transcriptions_to_df import generate_dataframes
from utils.coo_311_utils import lsi_model_req, status_message
from preprocess_text import clean_text, create_trigram_corpus
from sklearn.cluster import AgglomerativeClustering as aggcluster


def write_general_model(model, dictionary, tfidf, lsi_corpus, blob_keys):
    write_folder =  "LSImodel_1"
    num_identify = 1
    while write_folder in os.listdir('./lsi_model'):
        num_identify += 1
        write_folder_split = write_folder.split("_")
        write_folder = write_folder_split[0] + "_" + str(num_identify)
    os.mkdir("./lsi_model/" + write_folder)
    model.save("./lsi_model/" + write_folder + "/lsi_model.model")
    model_requisites = lsi_model_req(dictionary, tfidf, lsi_corpus)
    with open("./lsi_model/" + write_folder + "/lsi_model_req.pk1", "wb") as output:
        pickle.dump(model_requisites, output, pickle.HIGHEST_PROTOCOL)
    connection_string = blob_keys['storage_connection_string']
    container_name = "lsi-models"
    # Make sure folder exists in blob before this!
    try:
        for file_name in os.listdir('./lsi_model/' + write_folder):
            in_blob_client =  BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name=os.path.join(write_folder, file_name))
            with open(os.path.join("./lsi_model", write_folder, file_name), "rb") as data:
                    in_blob_client.upload_blob(data,overwrite=True)
        shutil.rmtree(os.path.join("./lsi_model", write_folder))
    except:
        status_message("Could Not Export Training Model")
    # status_message("Saved model and requisites in folder '{}'".format(write_folder))



def train_lsi_model(df, blob_keys, text_column=None):
    if isinstance(df, pd.DataFrame) and text_column is None:
        status_message("Failed. Please specify Column with Text")
        return None
    total_start_time = time.time()
    
    ## CLEAN TEXT
    status_message("Cleaning Text...")
    text_df = clean_text(df, text_column)
    status_message("Text Cleaned")
    ## CREATING CORPUS
    start_time_corpus = time.time()
    initial_corpus = text_df['clean_text'].apply(lambda x: x.split(' ')).values
    trigram_corpus = create_trigram_corpus(initial_corpus)

    dictionary = Dictionary(trigram_corpus)
    corpus = [dictionary.doc2bow(text) for text in trigram_corpus]
    status_message("Created Corpus in: {:.2f} seconds".format(time.time() - start_time_corpus))

    ## Generate TFIDF Matrix
    start_time_tfidf = time.time()
    tfidf = TfidfModel(corpus)
    corpus_modelled = tfidf[corpus]
    status_message("Created TFIDF in: {:.2f} seconds".format(time.time() - start_time_tfidf))


    ## LSI Model
    start_time_train = time.time()
    lsi = LsiModel(corpus_modelled, id2word=dictionary, num_topics=200)
    lsi_corpus = lsi[corpus_modelled]
    status_message("Trained LSI Model in: {:.2f} seconds". format(time.time() - start_time_train))
    
    write_general_model(lsi, dictionary, tfidf, lsi_corpus, blob_keys)
    status_message("Total Time Taken: {:.2f} seconds".format(time.time() - total_start_time))

############################################################################
######### TRAINING THE DOCUMENT AND SENTENCE GENERAL LOOKUP MODELS #########

# # Prepare Quotes
# # test = pd.read_csv('../Connor.Wilkinson/transcription_csv/phrase_transcription_df.csv')
# status_message("Reading in Transcriptions")
# df = pd.read_csv('../../Connor.Wilkinson/transcription_csv/phrase_options_df.csv')
# # df['lexical'] = df['lexical'].fillna('')


# ## TRAIN ##
# # TRAIN DOCUMENT TOPIC MODEL
# # beginning_cutoff_perc = 0.1
# # end_cutoff_perc = 0.8
# # regrouping_df = trim_transcriptions(df, beginning_cutoff_perc, end_cutoff_perc)
# # regrouping_df = pd.read_csv('./transcription_csv/full_10_80_lex.csv')
# # train_lsi_model(regrouping_df, 'lexical')

# # TRAIN SENTENCE TOPIC MODEL
# sentence_df = df[df['phraseIndex'] == 0]
# sentence_df['lexical'] = sentence_df['lexical'].fillna('')
# train_lsi_model(sentence_df, 'lexical')
############################################################################


# Trimming Transcriptions
def trim_transcriptions(df, beginning_cutoff_perc, end_cutoff_perc):
    #
    status_message("Trimming Transcriptions")
    df_best = df[df['phraseIndex'] == 0].reset_index(drop=True)
    df_best['offset_float'] = df_best['offset'].apply(lambda x: float(x[2:][:-1]) if 'M' not in x or 'S' not in x else float(x[2:].split('M')[0]) * 60 + float(x[2:].split('M')[1][:-1]))
    # Assuming people dont leave the conversation with an incredible run-on sentence...
    approx_call_len = df_best.groupby('blobName')['offset_float'].max()
    trimmed_df = pd.DataFrame()
    blob_num = 0
    for blob_name in df_best['blobName'].unique():
        blob_num += 1
        if blob_num % 1000 == 0:
            status_message("Trimmed {} / {} Transcriptions".format(blob_num, len(approx_call_len)))
        trimmed_blob_df = df_best[df_best['blobName'] == blob_name][df_best['offset_float'] > approx_call_len[blob_name] * beginning_cutoff_perc][df_best['offset_float'] < approx_call_len[blob_name] * end_cutoff_perc]
        trimmed_df = trimmed_df.append(trimmed_blob_df)
    #
    regrouping_df = trimmed_df.groupby('blobName')['lexical'].transform(lambda x: ' '.join(x))
    regrouping_df = regrouping_df.drop_duplicates()
    # regrouping_df.to_frame().to_csv('./transcription_csv/full_{}_{}_lex.csv'.format(beginning_cutoff_perc*100, end_cutoff_perc*100), index=False)


# def determine_LSI_dimension(x_train, x_test):
#     # Apply dimenstionality reduction - check below for explained variance to capture some relative amount of variance
#     pca = PCA(n_components=500)
#     pca_df_train = pd.DataFrame(pca.fit_transform(x_train))
#     pca_df_test = pd.DataFrame(pca.transform(x_test))
#     # Show the plot of explained variance relative to number of components
#     exp_var = pca.explained_variance_ratio_
#     exp_var_cumsum = np.cumsum(exp_var)



def train_lsi_cluster_model(df, text_column, num_topics, blob_keys, write_model = True, custom_folder_name = None):
    #########################
    # Train LSI Model on Topic
    #########################
    status_message("---- Training LSI Model")
    text_df = clean_text(df, text_column)
    ## CREATING CORPUS
    initial_corpus = text_df['clean_text'].apply(lambda x: x.split(' ')).values
    trigram_corpus = create_trigram_corpus(initial_corpus)

    dictionary = Dictionary(trigram_corpus)
    corpus = [dictionary.doc2bow(text) for text in trigram_corpus]

    ## Generate TFIDF Matrix
    tfidf = TfidfModel(corpus)
    corpus_modelled = tfidf[corpus]

    ## LSI Model
    lsi = LsiModel(corpus_modelled, id2word=dictionary, num_topics=num_topics)
    lsi_corpus = lsi[corpus_modelled]
    status_message("---- Trained LSI Model")

    #########################
    # Map them to LSI Space
    #########################
    status_message("---- Mapping Text to LSI Space")
    test_text_df = text_df
    test_text_df['test_initial_corpus'] = test_text_df['clean_text'].apply(lambda x: x.split(' ')).values
    test_text_df['test_trigram_corpus'] = test_text_df['test_initial_corpus'].apply(lambda x: create_trigram_corpus(x))
    test_text_df['test_final_corpus'] = test_text_df['test_trigram_corpus'].apply(lambda x: dictionary.doc2bow(x))
    test_text_df['lsi_transform'] = test_text_df['test_final_corpus'].apply(lambda x: lsi[tfidf[x]])
    test_text_df = test_text_df[test_text_df['lsi_transform'].apply(lambda x: len(x) == num_topics)]

    status_message("---- LSI Transform Complete")

    #########################
    # Perform Hierarchical Clustering such that 95% of Vectors are within a "Major" Cluster. Discard Leaf Vectors (ones outside of vectors)
    #########################
    status_message("---- Clustering Vectors")

    # Number of Vectors in Cluster such that it is considered a "Major Group"
    major_group_total = len(test_text_df) * 0.05
    test_text_df['lsi_transform_values'] = test_text_df['lsi_transform'].apply(lambda x: np.array([i[1] for i in x]))

    distance_threshold = 0
    leaf_count = len(test_text_df)
    old_leaf_count = 0
    old_num_clusters = 0
    same_iteration_count = 0

    while leaf_count > 0.05*len(test_text_df) and same_iteration_count <= 5:
        if leaf_count / len(test_text_df) > 0.1:
            distance_threshold += 1
        elif leaf_count / len(test_text_df) > 0.08:
            distance_threshold += 0.5
        else:
            distance_threshold += 0.1
        status_message("Testing Distance Threshold: {:.1f}".format(distance_threshold))
        aggtest = aggcluster(distance_threshold = distance_threshold, n_clusters = None)
        test_text_df['agg_result'] = aggtest.fit_predict(test_text_df['lsi_transform_values'].tolist())
        status_message("Number of Clusters: {}".format(test_text_df['agg_result'].max() + 1))
        agg_result_count = test_text_df.groupby('agg_result').count()['lsi_transform_values']
        leaf_count = agg_result_count[agg_result_count < major_group_total].sum()
        if old_leaf_count == leaf_count and old_num_clusters == test_text_df['agg_result'].max() + 1:
            same_iteration_count += 1
        else:
            same_iteration_count = 0
        old_leaf_count = leaf_count
        old_num_clusters = test_text_df['agg_result'].max() + 1
        status_message("Percentage of Leafs: {}".format(leaf_count / len(test_text_df)))
        status_message("Same Iteration Count: {}".format(same_iteration_count))
        print("")

    status_message("Using Last Result. Number of Clusters: {}".format(len(agg_result_count)))
    status_message("Cluster Distribution:")
    
    final_cluster_vectors = []
    for group in test_text_df['agg_result'].unique():
        final_cluster_vectors.append(np.mean(test_text_df[test_text_df['agg_result'] == group]['lsi_transform_values'], axis=0))

    cluster_vectors_df = pd.DataFrame(final_cluster_vectors, index = ["train_" + str(i) for i in pd.DataFrame(final_cluster_vectors).index])
    if write_model:
        cluster_model_folder = "LSI_Clustering_Approach"
        path = os.path.join("./lsi_model", cluster_model_folder)
        count = 0
        if custom_folder_name == None:
            folder_name = "cluster_model"
            while folder_name in os.listdir(path):
                count += 1
                folder_name = folder_name + str(count)
        else:
            while custom_folder_name in os.listdir(path):
                count += 1
                custom_folder_name = custom_folder_name + str(count)
            if count > 0:
                status_message("Folder Name Already Exists! Printing Folder as with name: {}".format(custom_folder_name))
            else:
                status_message("Printing Model using the following Path: {}".format(os.path.join(path, custom_folder_name)))
            folder_name = custom_folder_name
        os.mkdir(os.path.join(path, folder_name))
        cluster_vectors_df.to_csv(os.path.join(path, folder_name, "cluster_vectors_df.csv"))
        weights_df = test_text_df.groupby('agg_result').count()['clean_text']
        weights_df.columns = ['groupCounts']
        weights_df.to_csv(os.path.join(path, folder_name, "cluster_weights_df.csv"), index=False)
        lsi.save(os.path.join(path, folder_name, "lsi_model.model"))
        model_requisites = lsi_model_req(dictionary, tfidf, lsi_corpus)
        with open(os.path.join(path, folder_name, "lsi_model_req.pk1"), "wb") as output:
            pickle.dump(model_requisites, output, pickle.HIGHEST_PROTOCOL)
        # Write to Blob and Clear Folder
        connection_string = blob_keys['storage_connection_string']
        container_name = "lsi-models"
        try:
            for file_name in os.listdir(os.path.join(path, folder_name)):
                in_blob_client =  BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name=os.path.join(cluster_model_folder, folder_name, file_name))
                with open(os.path.join(path, folder_name, file_name), "rb") as data:
                        in_blob_client.upload_blob(data,overwrite=True)
            shutil.rmtree(os.path.join(path, folder_name))
        except:
            status_message("Could Not Export Training Model")
    return cluster_vectors_df, weights_df, dictionary, tfidf, lsi


def train_general_lsi_model_on_processed_transcriptions(model_name, blob_keys,  model_type = ''):
    # blob_keys = config['BLOB_KEYS']
    if model_type == '':
        status_message("Please specify if this model is training on the 'sentences' or 'documents'")
        status_message("model_type takes 's' and 'd' as arguments")
        return None
    transcription_container_name = 'training-transcriptions'
    model_container_name = 'lsi-models'
    connection_string = blob_keys['storage_connection_string']
    # Check if model_name already exists
    model_container_client = ContainerClient.from_connection_string(conn_str=connection_string, container_name=model_container_name)
    if model_name in [i['name'] for i in model_container_client.list_blobs()]:
        status_message("Model Name already exists. Please choose another")
        return None
    # Read in Transcriptions
    all_blob_full_df, all_blob_sentence_df = generate_dataframes(transcription_container_name, connection_string)
    all_blob_full_df['fullPhraseLex'] = all_blob_full_df['fullPhraseLex'].fillna('')
    all_blob_full_df['fullPhraseDisplay'] = all_blob_full_df['fullPhraseDisplay'].fillna('')
    all_blob_sentence_df = all_blob_sentence_df[all_blob_sentence_df['phraseIndex'] == 0]
    all_blob_sentence_df['lexical'] = all_blob_sentence_df['lexical'].fillna('')
    all_blob_sentence_df['display'] = all_blob_sentence_df['display'].fillna('')
    lsi = None

    # Train LSI Model
    if model_type == 's':
        train_lsi_model(all_blob_sentence_df, blob_keys, 'lexical')
    if model_type == 'd':
        train_lsi_model(all_blob_full_df, blob_keys, 'fullPhraseLex')



def train_cluster_model_on_text_file(model_name, container_name, training_blob_name, blob_keys):
    # Assuming Input Blob is CSV with Proper Format. See samples in "extra-training-material" container for examples
    # Ingest Blob
    connection_string = blob_keys['storage_connection_string']
    num_dim = 3
    in_blob_client =  BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name=training_blob_name)
    with open(os.path.join("./temp_cluster_training_data", "cluster_train.csv"), "wb") as data:
        data.write(in_blob_client.download_blob().readall())
    train_df = pd.read_csv("./temp_cluster_training_data/cluster_train.csv", headers=None)
    shutil.rmtree(os.path.join("./temp_cluster_training_data/cluster_train.csv"))
    cluster_vectors_df, weights_df, dictionary, tfidf, lsi = train_lsi_cluster_model(train_df, 0, num_dim, blob_keys = blob_keys, write_model = True, custom_folder_name = model_name)





