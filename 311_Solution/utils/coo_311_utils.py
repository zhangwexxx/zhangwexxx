import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from azure.storage.blob import BlobClient

class lsi_model_req(object):
    def __init__(self, dictionary, tfidf_matrix, lsi_base_corpus):
        self.dictionary = dictionary
        self.tfidf_matrix = tfidf_matrix
        self.lsi_base_corpus = lsi_base_corpus

class qa_response_structure(object):
    def __init__(self):
        self.blob_name = ""
        self.garbage_call_ind = 0
        self.open_greeting = {"value": False, "conf": 0}
        self.open_language =  {"value": False, "conf": 0}
        self.probing_questions =  {"value": False, "conf": 0}
        self.protocol_hold =  {"value": False, "conf": 0}
        self.protocol_transfer =  {"value": False, "conf": 0}
        self.protocol_mfippa =  {"value": False, "conf": 0}
        self.protocol_councellor_checkbox =  {"value": False, "conf": 0}
        self.resolve_accurate_info =  {"value": False, "conf": 0}
        self.resolve_all_options =  {"value": False, "conf": 0}
        self.serv_request_open =  {"value": False, "conf": 0}
        self.serv_request_all_info =  {"value": False, "conf": 0}
        self.serv_request_website =  {"value": False, "conf": 0}
        self.serv_request_number =  {"value": False, "conf": 0}
        self.confirm_outcome_service_level =  {"value": False, "conf": 0}
        self.confirm_outcome_communicate =  {"value": False, "conf": 0}
        self.close_final_assistance =  {"value": False, "conf": 0}
        self.close_friendly =  {"value": False, "conf": 0}
        self.prof_respect_willing_to_help =  {"value": False, "conf": 0}
        self.prof_respect_client_respect =  {"value": False, "conf": 0}
        self.prof_respect_positive_city =  {"value": False, "conf": 0}
        self.prof_respect_polite =  {"value": False, "conf": 0}
        self.prof_respect_active_listening =  {"value": False, "conf": 0}

def status_message(imsg):
    s = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    s = s[:-4]
    timestamp = '[' + s + ']'
    print(timestamp + ' ' + imsg)

def convert_response_to_df(response):
    # print(response.open_greeting)
    # print(response.open_language)
    # print(response.probing_questions)
    # print(response.protocol_hold)
    # print(response.protocol_transfer)
    # print(response.protocol_mfippa)
    # print(response.protocol_councellor_checkbox)
    # print(response.resolve_accurate_info)
    # print(response.resolve_all_options)
    # print(response.serv_request_open)
    # print(response.serv_request_all_info)
    # print(response.serv_request_website)
    # print(response.serv_request_number)
    # print(response.confirm_outcome_service_level)
    # print(response.confirm_outcome_communicate)
    # print(response.close_final_assistance)
    # print(response.close_friendly)
    # print(response.prof_respect_willing_to_help)
    # print(response.prof_respect_client_respect)
    # print(response.prof_respect_positive_city)
    # print(response.prof_respect_polite)
    # print(response.prof_respect_active_listening)
    return pd.DataFrame([[
        response.blob_name,
        response.garbage_call_ind,
        response.open_greeting['value'],
        response.open_greeting['conf'],
        response.open_language['value'],
        response.open_language['conf'],
        response.probing_questions['value'],
        response.probing_questions['conf'],
        response.protocol_hold['value'],
        response.protocol_hold['conf'],
        response.protocol_transfer['value'],
        response.protocol_transfer['conf'],
        response.protocol_mfippa['value'],
        response.protocol_mfippa['conf'],
        response.protocol_councellor_checkbox['value'],
        response.protocol_councellor_checkbox['conf'],
        response.resolve_accurate_info['value'],
        response.resolve_accurate_info['conf'],
        response.resolve_all_options['value'],
        response.resolve_all_options['conf'],
        response.serv_request_open['value'],
        response.serv_request_open['conf'],
        response.serv_request_all_info['value'],
        response.serv_request_all_info['conf'],
        response.serv_request_website['value'],
        response.serv_request_website['conf'],
        response.serv_request_number['value'],
        response.serv_request_number['conf'],
        response.confirm_outcome_service_level['value'],
        response.confirm_outcome_service_level['conf'],
        response.confirm_outcome_communicate['value'],
        response.confirm_outcome_communicate['conf'],
        response.close_final_assistance['value'],
        response.close_final_assistance['conf'],
        response.close_friendly['value'],
        response.close_friendly['conf'],
        response.prof_respect_willing_to_help['value'],
        response.prof_respect_willing_to_help['conf'],
        response.prof_respect_client_respect['value'],
        response.prof_respect_client_respect['conf'],
        response.prof_respect_positive_city['value'],
        response.prof_respect_positive_city['conf'],
        response.prof_respect_polite['value'],
        response.prof_respect_polite['conf'],
        response.prof_respect_active_listening['value'],
        response.prof_respect_active_listening['conf']
    ]], columns = [
        'blob_name',
        'garbage_ind',
        '1_1_open_greeting_value',
        '1_1_open_greeting_conf',
        '1_2_open_language_value',
        '1_2_open_language_conf',
        '2_1_probing_questions_value',
        '2_1_probing_questions_conf',
        '3_1_protocol_hold_value',
        '3_1_protocol_hold_conf',
        '3_2_protocol_transfer_value',
        '3_2_protocol_transfer_conf',
        '3_3_protocol_mfippa_value',
        '3_3_protocol_mfippa_conf',
        '3_4_protocol_councellor_checkbox_value',
        '3_4_protocol_councellor_checkbox_conf',
        '4_1_resolve_accurate_info_value',
        '4_1_resolve_accurate_info_conf',
        '4_2_resolve_all_options_value',
        '4_2_resolve_all_options_conf',
        '5_1_serv_request_open_value',
        '5_1_serv_request_open_conf',
        '5_2_serv_request_all_info_value',
        '5_2_serv_request_all_info_conf',
        '5_3_serv_request_website_value',
        '5_3_serv_request_website_conf',
        '5_4_serv_request_number_value',
        '5_4_serv_request_number_conf',
        '6_1_confirm_outcome_service_level_value',
        '6_1_confirm_outcome_service_level_conf',
        '6_2_confirm_outcome_communicate_value',
        '6_2_confirm_outcome_communicate_conf',
        '7_1_close_final_assistance_value',
        '7_1_close_final_assistance_conf',
        '7_2_close_friendly_value',
        '7_2_close_friendly_conf',
        '8_1_prof_respect_willing_to_help_value',
        '8_1_prof_respect_willing_to_help_conf',
        '8_2_prof_respect_client_respect_value',
        '8_2_prof_respect_client_respect_conf',
        '8_3_prof_respect_positive_city_value',
        '8_3_prof_respect_positive_city_conf',
        '8_4_prof_respect_polite_value',
        '8_4_prof_respect_polite_conf',
        '8_5_prof_respect_active_listening_value',
        '8_5_prof_respect_active_listening_conf'
    ])


def find_qna_pair_from_display(full_display_text):
    question_mark_split = full_display_text.split("?")
    question_list = []
    answer_list = []
    for sentence_ind in range(len(question_mark_split[:-1])):
        question_list.append(re.split('\.|\!', question_mark_split[sentence_ind])[-1])
        answer_list.append('. '.join(re.split('\.|\!', question_mark_split[sentence_ind+1])[:-1]))
    return question_list, answer_list


def acquire_final_qna(full_display_text):
    questions_list, answers_list = find_qna_pair_from_display(full_display_text)
    if len(answers_list) <= 2:
        return questions_list, answers_list
    if len(answers_list[-1]) < 25:
        return questions_list[-2:], answers_list[-2:]
    return [questions_list[-1]], [answers_list[-1]]


def write_text_locally(sentence_df, text_blob_name):
    # Grab Sentences Associated with Just That Blob
    transcription_sentences = sentence_df[sentence_df['blobName'].apply(lambda x: text_blob_name.split('.')[0] in x)]['display']
    with open("./temp_output_dir/" + text_blob_name, "w") as write_file:
        for transcription_sentence in transcription_sentences:
            write_sentences_df = write_file.write(transcription_sentence + "\n\n")
    

def write_text_blob(sentence_df, blob_keys):
    # Obtain Connection to Storage Account
    connection_string = blob_keys['storage_connection_string']
    output_container_name ='311-automated-output'
    text_blob_name = sentence_df.iloc[0]['blobName'].split('/')[-1].split('.')[0] + ".txt"  
    try:
        # Write Text Locally
        write_text_locally(sentence_df, text_blob_name)
        # Write Text to Blob
        in_blob_client =  BlobClient.from_connection_string(conn_str=connection_string, container_name=output_container_name, blob_name="transcription_text/" + text_blob_name)
        with open("./temp_output_dir/" + text_blob_name, "rb") as data:
            in_blob_client.upload_blob(data,overwrite=True)
        os.remove("./temp_output_dir/" + text_blob_name)
        status_message("Wrote Audio File Transcript to Blob:  {}".format(text_blob_name))
    except Exception as e:
        status_message("Could Not Write Text File {} to Blob Storage".format(text_blob_name))
        print(e)

def write_csv_blob(final_response_df, csv_blob_name, blob_keys):
    # Obtain Connection to Storage Account
    connection_string = blob_keys['storage_connection_string']
    output_container_name ='311-automated-output'
    try:
        # Read CSV Locally
        in_blob_client =  BlobClient.from_connection_string(conn_str=connection_string, container_name=output_container_name, blob_name=csv_blob_name)
        with open("./temp_output_dir/" + csv_blob_name, "wb") as data:
            data.write(in_blob_client.download_blob().readall())
        existing_df = pd.read_csv("./temp_output_dir/" + csv_blob_name)
        # existing_df['temp_index'] = existing_df.index + 1
        # final_response_df['temp_index'] = 0
        final_response_df = existing_df.append(final_response_df, sort=True).reset_index(drop=True)
        # final_response_df = final_response_df.drop('temp_index', axis=1)
        final_response_df.to_csv("./temp_output_dir/" + csv_blob_name, index=False)
        # Write Updated CSV to Blob
        with open("./temp_output_dir/" + csv_blob_name, "rb") as data:
            in_blob_client.upload_blob(data,overwrite=True)
        os.remove("./temp_output_dir/" + csv_blob_name)
        status_message("Wrote Final Result CSV:  {}".format(csv_blob_name))
    except Exception as e:
        status_message("Could Not Write CSV File {} to Blob Storage".format(csv_blob_name))
        print(e)



