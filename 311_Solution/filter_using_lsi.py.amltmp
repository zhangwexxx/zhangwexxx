import numpy as np
import pandas as pd
from LSI_testing_functions import filter_on_q_LSI
from utils.coo_311_utils import lsi_model_req, status_message
from call_LUIS import get_luis_response, handle_multiple_luis_responses



def recognize_garbage_call_lsi_response(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    #############################
    # SET PARAMETERS
    #############################
    # Get Sentences that match query
    query = ['garbage',
        'green bin',
        'blue bin',
        'garbage bags',
        'recycle'
        ]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    if sum([len(i) > 0 for i in thresh_list]):
        # status_message("Garbage Relation Found with LSI Model.")
        corr_sentences = sum([[j[0] for j in i] for i in thresh_list], [])
        corr_perc = sum([[j[1] for j in i] for i in thresh_list], [])
        lsi_result = {}
        for i in range(len(corr_sentences)):
            if corr_sentences[i] in lsi_result.keys():
                if corr_perc[i] > lsi_result[corr_sentences[i]]:
                    lsi_result[corr_sentences[i]] = corr_perc[i]
            else:
                lsi_result[corr_sentences[i]] = corr_perc[i]
        return lsi_result
    else:
        # status_message("No Garbage Intent Found with LSI Model.")
        return {}


def recognize_transfer_request_lsi(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ['i will transfer you', 'would you like me to transfer you', 'transfer you']
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    transfer_result_dict = {}
    for phrase_results in thresh_list:
        for sentence_matches in phrase_results:
            if sentence_matches[0] in transfer_result_dict.keys():
                transfer_result_dict[sentence_matches[0]] = max(sentence_matches[1], transfer_result_dict[sentence_matches[0]])
            else:
                transfer_result_dict[sentence_matches[0]] = sentence_matches[1]
    return transfer_result_dict

def recognize_service_request_number(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["request number", "can I provide you with service request number", "number for this request"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    service_num_result_dict = {}
    for phrase_results in thresh_list:
        for sentence_matches in phrase_results:
            if sentence_matches[0] in service_num_result_dict.keys():
                service_num_result_dict[sentence_matches[0]] = max(sentence_matches[1], service_num_result_dict[sentence_matches[0]])
            else:
                service_num_result_dict[sentence_matches[0]] = sentence_matches[1]
    return service_num_result_dict

def recognize_final_assistance(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["can I help you with anything else", "is there anything else I can assist you with", "help you", "assistance"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh
        )
    conf = 0
    conf_list = []
    result_count = 0
    for result in thresh_list:
        if len(result) > 0:
            result_count += 1
            # Its shape has to be [(,)]
            temp_conf = result[0][1]
            if temp_conf > 0.2:
                conf_list.append(min(0.95, temp_conf / 0.4))
    if len(conf_list) > 0:
        conf += 0.8 * np.mean(conf_list)
    conf += 0.2 * result_count / len(query)
    return conf


def recognize_positive_close(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["have a great day", "thank you for calling", "have a good evening", "have a good night"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    luis_sent_score = 0
    # The filtering on the next line ensures we get only ONE response from LUIS
    luis_input = test_df[text_col].iloc[0][-500:]
    luis_response = get_luis_response(raw_utterance = luis_input, specific_topic = 'greeting')
    luis_response = luis_response[0]
    if len(luis_response) > 0:
        luis_sentiment_response = luis_response[2]
        if luis_sentiment_response['label'] == 'positive':
            luis_sent_score = luis_sentiment_response['score']
    pos_end_result_dict = {}
    for phrase_results in thresh_list:
        for sentence_matches in phrase_results:
            if sentence_matches[0] in pos_end_result_dict.keys():
                pos_end_result_dict[sentence_matches[0]] = max(sentence_matches[1], pos_end_result_dict[sentence_matches[0]])
            else:
                pos_end_result_dict[sentence_matches[0]] = sentence_matches[1]
    if len(pos_end_result_dict) > 0:
        lsi_result_conf = list(pos_end_result_dict.values())[0]
        pos_end_score = lsi_result_conf * 0.2 + luis_sent_score * 0.8
    else:
        pos_end_score = luis_sent_score * 0.8
    return pos_end_score


def recognize_hold_intro_lsi(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["please hold", "hold the line"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    conf = 0
    ind = 0
    ind_list = []
    conf_list = []
    luis_conf = 0
    for q_result in thresh_list:
        if len(q_result) > 0:
            for sentence_result in q_result:
                # Get LUIS Result
                luis_response = get_luis_response(raw_utterance = test_df[text_col].loc[sentence_result[0]], specific_topic = 'hold_out')
                luis_conf = luis_response[0][0][0]
                # Its shape has to be [(,)]
                temp_conf = sentence_result[1]
                if temp_conf > 0.2:
                    ind_list.append(sentence_result[0])
                    conf_list.append(min(0.95, temp_conf))
    if len(conf_list) > 0:
        conf += 0.4 * np.mean(conf_list)
        ind = min(ind_list)
        conf += 0.6 * luis_conf
    return_dict = {}
    return_dict[ind] = conf
    return return_dict


def recognize_hold_outro_lsi(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["thank you for holding", "thanks for holding"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    conf = 0
    ind = 0
    ind_list = []
    conf_list = []
    luis_conf = 0
    for q_result in thresh_list:
        if len(q_result) > 0:
            for sentence_result in q_result:
                # Get LUIS Result
                luis_response = get_luis_response(raw_utterance = test_df[text_col].loc[sentence_result[0]], specific_topic = 'hold_out')
                luis_conf = luis_response[0][0][0]
                # Its shape has to be [(,)]
                temp_conf = sentence_result[1]
                if temp_conf > 0.2:
                    ind_list.append(sentence_result[0])
                    conf_list.append(min(0.95, temp_conf))
    if len(conf_list) > 0:
        conf += 0.4 * conf_list[0]
        ind = ind_list[0]
        conf += 0.6 * luis_conf
    return_dict = {}
    return_dict[ind] = conf
    return return_dict


def recognize_city_speech(test_df, text_col, lsi, dictionary, tfidf, conf_thresh = 0.5):
    query = ["city of ottawa"]
    # Get LSI Result
    thresh_list = filter_on_q_LSI(
        query, 
        lsi = lsi,
        tfidf = tfidf,
        dictionary = dictionary,
        test_df = test_df,
        text_col = text_col,
        conf_thresh = conf_thresh, 
        )
    conf = 0
    ind = 0
    ind_list = []
    conf_list = []
    luis_conf = 0
    for q_result in thresh_list:
        if len(q_result) > 0:
            for sentence_result in q_result:
                temp_conf = sentence_result[1]
                if temp_conf > 0.2:
                    ind_list.append(sentence_result[0])
                    conf_list.append(min(0.95, temp_conf))
    return dict(zip(ind_list, conf_list))

