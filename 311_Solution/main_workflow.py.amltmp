#!/bin/env python
# Import Dependencies for Pipeline Container

from utils.import_utils import import_required_libraries
import_required_libraries()
import sys
import nltk
import time
import json
import pandas as pd
from datetime import datetime
from configparser import ConfigParser
from input_preparation import gather_input
from fill_response import fill_response_fields
from LSI_testing_functions import import_lsi_model
from filter_using_LSI import recognize_garbage_call_lsi_response
from utils.coo_311_utils import lsi_model_req, status_message, qa_response_structure, convert_response_to_df, write_text_blob, write_csv_blob
nltk.download('stopwords')

pd.options.mode.chained_assignment = None

# def gather_test_input():
#     ##############################
#     #### GATHER INPUT
#     ##############################
#     # Read in Sentences
#     status_message("Reading in Transcriptions")
#     sentence_df = pd.read_csv('../Training_Material/transcription_csv/phrase_options_df.csv')
#     sentence_df = sentence_df[sentence_df['phraseIndex'] == 0]
#     sentence_df['lexical'] = sentence_df['lexical'].fillna('')
#     sentence_df['display'] = sentence_df['display'].fillna('')
#     full_df = pd.read_csv('../Training_Material/transcription_csv/full_transcription_df.csv')
#     full_df['fullPhraseLex'] = full_df['fullPhraseLex'].fillna('')
#     full_df['fullPhraseDisplay'] = full_df['fullPhraseDisplay'].fillna('')
#     ######################
#     # Garbage Related Call
#     ######################
#     blob_name1 = '6792fcb1-5d30-4709-b79e-945ecb5417ef/audio-adastra-processing/2436462_UCMAVoice_2020100916440575.mp3.json'
#     blob_name2 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506363_UCMAVoice_2020102817165384.mp3.json'
#     blob_name3 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506404_UCMAVoice_2020102817311671.mp3.json'
#     ######################
#     # Non Garbage Related Calls
#     ######################
#     blob_name4 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506321_UCMAVoice_2020102817050870.mp3.json'
#     # blob_name = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506795_UCMAVoice_2020102823190387.mp3.json'
#     # Test on List of Blobs
#     blob_names = [blob_name2] #, blob_name1, blob_name2]
#     all_blob_sentence_df = sentence_df[sentence_df['blobName'].apply(lambda x: x in blob_names)]
#     all_blob_full_df = full_df[full_df['blobName'].apply(lambda x: x in blob_names)]
#     status_message("Transcriptions Ingested For Test: {}".format(blob_names))
#     print(all_blob_full_df['fullPhraseDisplay'].iloc[0])
#     print(all_blob_sentence_df['lexical'])
#     return all_blob_full_df, all_blob_sentence_df

def azureml_main():
    ##############################
    #### GATHER INPUT
    ##############################
    final_response_df = pd.DataFrame()
    time_import_start = time.time()
    # Read in Config File
    config = ConfigParser()
    config.read("./custom_config/config.ini")
    input_sas_uri = "https://msa311stgz7qrruujufqzq.blob.core.windows.net/input-audio?sp=rl&st=2021-04-29T01:57:46Z&se=2021-05-01T03:57:46Z&spr=https&sv=2020-02-10&sr=c&sig=ROW5xnJIDsZsx3Dt3NMLMhc59qBcw8si3X5GfqYb0Kw%3D"
    output_sas_uri = "https://msa311stgz7qrruujufqzq.blob.core.windows.net/transcribed-audio?sp=racwdl&st=2021-04-29T02:01:06Z&se=2021-04-30T10:01:06Z&spr=https&sv=2020-02-10&sr=c&sig=IzRipsq52vpkQf2DnVAWEYpBzWyr5QSMxtIUdMGlnbA%3D"
    all_blob_full_df, all_blob_sentence_df = gather_input(config['BLOB_KEYS'], config['STT_KEYS'], input_sas_uri, output_sas_uri)
    # all_blob_full_df, all_blob_sentence_df = gather_input(config['BLOB_KEYS'], config['STT_KEYS'], sys.argv[1], sys.argv[2])
    if len(all_blob_full_df) == 0 and len(all_blob_sentence_df) == 0:
        status_message("Killing Process.")
        return None
    all_blob_full_df = all_blob_full_df.reset_index(drop=True)
    all_blob_sentence_df = all_blob_sentence_df.reset_index(drop=True)
    # all_blob_full_df, all_blob_sentence_df = gather_test_input()
    # Import LSI Models
    document_filter_lsi, document_filter_dict, document_filter_tfidf = import_lsi_model(config['MODELS']['document_topic_model'], config['BLOB_KEYS'])
    sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf = import_lsi_model(config['MODELS']['sentence_topic_model'], config['BLOB_KEYS'])
    status_message("Time Taken Importing Data and Models: {:.2f} Seconds".format(time.time() - time_import_start))
    time_complete_processing = time.time()
    for blob_name in all_blob_full_df.blobName.unique():
        print()
        status_message("Processing Blob: {}".format(blob_name))
        time_processing_blob = time.time()
        test_sentence_df = all_blob_sentence_df[all_blob_sentence_df['blobName'] == blob_name]
        test_full_df = all_blob_full_df[all_blob_full_df['blobName'] == blob_name]
        response = qa_response_structure()
        ##############################
        #### CHECK IF GARBAGE SENTENCE
        ##############################
        # status_message("Checking for Garbage Related Call")
        text_full_col = 'fullPhraseDisplay'
        text_sentence_col = 'lexical'
        response.blob_name = blob_name
        garbage_response = recognize_garbage_call_lsi_response(
            test_full_df, 
            text_full_col, 
            document_filter_lsi, 
            document_filter_dict, 
            document_filter_tfidf, 
            conf_thresh = 0.2)
        if len(garbage_response) == 0:
            # status_message("No Garbage Relation Found in Call. Halting Process")
            write_text_blob(test_sentence_df, config['BLOB_KEYS'])
            response_df = convert_response_to_df(response)
            final_response_df = final_response_df.append(response_df, ignore_index = True)
            status_message("    Audio Blob Not a Garbage Related Call")
            continue
        # status_message("Garbage Relation Found in Call. Continuing to Extract Information")
        response.garbage_call_ind = 1
        ##############################
        #### FILL response FIELDS
        ##############################
        response = fill_response_fields(
            test_sentence_df = test_sentence_df,
            test_full_df = test_full_df,
            text_sentence_col = text_sentence_col,
            text_full_col = text_full_col,
            document_filter_lsi = document_filter_lsi,
            document_filter_dict = document_filter_dict,
            document_filter_tfidf = document_filter_tfidf,
            sentence_filter_lsi = sentence_filter_lsi,
            sentence_filter_dict = sentence_filter_dict,
            sentence_filter_tfidf = sentence_filter_tfidf,
            config = config,
            response = response)
        status_message("    Blob Processing Time: {:.2f} Seconds".format(time.time() - time_processing_blob))
        write_text_blob(test_sentence_df, config['BLOB_KEYS'])
        # Append to Final Response CSV
        response_df = convert_response_to_df(response)
        final_response_df = final_response_df.append(response_df, ignore_index = True)
    write_csv_blob(final_response_df, "response_21_03_17_14_33.csv", config['BLOB_KEYS'])
    print()
    status_message("Complete Processing Time: {:.2f} Seconds".format(time.time() - time_complete_processing))
    return final_response_df


if __name__ == '__main__':
    json_response = azureml_main()
    # print("\nResponse DataFrame:")
    # print(json.dumps(str(json_response), indent=3))
