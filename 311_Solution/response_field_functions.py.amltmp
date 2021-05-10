import re
import random
import numpy as np
import pandas as pd
import Levenshtein as lev
from configparser import ConfigParser
from call_LUIS import get_luis_response, handle_multiple_luis_responses
from LSI_testing_functions import filter_on_q_LSI, trim_test_set, test_LSI_cluster_model
from utils.coo_311_utils import lsi_model_req, status_message, find_qna_pair_from_display, acquire_final_qna
from filter_using_LSI import recognize_transfer_request_lsi, recognize_service_request_number, recognize_positive_close, recognize_final_assistance, recognize_hold_intro_lsi, recognize_hold_outro_lsi, recognize_city_speech

def find_topic_cluster_LSI_LUIS(test_df, test_text_column, model_type, model_config, response, lsi_conf_thresh, specific_topic = '', trim_min_perc = 0, trim_max_perc = 1):
    model_name = model_type.split("_")[1]
    # status_message("Testing LSI and LUIS Model to Satisfy {}".format(model_name))
    ##########################
    # RUN AGAINST LSI
    ##########################
    # # Obtain Sentence DF
    # test_df = trim_test_set(test_df, trim_min_perc=trim_min_perc, trim_max_perc=trim_max_perc)
    # # test_df = test_df.groupby('blobName').agg({'lexical': lambda x: ' '.join(x)})
    # lsi_result = test_LSI_cluster_model(
    #     test_df, 
    #     model_config[model_type],
    #     test_text_column = test_text_column, 
    #     test_conf_thresh = lsi_conf_thresh
    #     )
    lsi_result = {}
    lsi_result[test_df.index[0]] = 0.6
    ##########################
    # RUN AGAINST LUIS
    ##########################
    # if len(lsi_result) == 0:
        # status_message("No Results Found when running against model: {}".format(model_config[model_type]))
    # status_message("Running LUIS Against {} Sentence(s)".format(len(lsi_result)))
    for index in lsi_result.keys():
        luis_responses = get_luis_response(test_df = test_df, test_text_column = test_text_column, index = index, specific_topic = specific_topic)
        found_prediction, found_entities, found_sentiment = handle_multiple_luis_responses(luis_responses)
    ##########################
    # GATHER RESULTS
    ##########################
    if len(lsi_result) > 0 and len(found_prediction) > 0:
        for result_index in found_prediction.keys():
            response_conf = 100*(found_prediction[result_index]*0.8 + lsi_result[result_index]*0.2)
            if response_conf > lsi_conf_thresh*100:
                response_dict = {
                    'value': True,
                    'conf': response_conf
                    }
                # status_message("LSI and LUIS Model Satisfied {}".format(model_name))
                return response_dict
            # else:
                # status_message("Found Match with Confidence ({:.1f}%) Below Threshold ({:.1f}%)".format(response_conf, lsi_conf_thresh*100))
                # status_message("No Match found for {} in LSI and LUIS Model".format(model_name))
    # else:
        # status_message("No Match found for {} in LSI and LUIS Model".format(model_name))
    return {}


def get_back_half_luis_response(test_full_df, test_text_column):
    cut_start_perc = 0.6
    cut_end_perc = 1
    test_full_df['sfd'] = test_full_df[test_text_column].apply(lambda x: re.split('\.|\!|\?', x))
    test_full_df['sfd_cut'] = test_full_df['sfd'].apply(lambda x: x[int(len(x)*cut_start_perc):int(len(x)*0.8)])
    test_full_df['back_half'] = test_full_df['sfd_cut'].apply(lambda x: '.'.join(x))
    # Run in Batches through LUIS
    # -- Why run in batches? LUIS only takes utterances of size 500, so we would likely need to batch anyways. Doing this allows us to analyze sentiment of sections of the text, where certain entities are found.
    batch_size = 300
    utterance = test_full_df['back_half'].iloc[0]
    utterance_list = [utterance[:batch_size]]
    for i in range(int(np.round(len(utterance)/batch_size, 0))):
        utterance_list.append(utterance[batch_size*(i+1):batch_size*(i+2)])
    back_half_luis_predictions = []
    back_half_luis_entities = []
    back_half_luis_sentiments = []
    if len(utterance_list) >= 10:
        utterance_list = random.sample(utterance_list, 10)
    # status_message("Calling LUIS {} times to obtain Entities and Sentiment of Call.".format(len(utterance_list)))
    for sub_utterance in utterance_list:
        luis_response = get_luis_response(raw_utterance = sub_utterance, specific_topic = 'greeting')
        luis_response = luis_response[0]
        if len(luis_response) > 0:
            back_half_luis_predictions.append(luis_response[0])
            back_half_luis_entities.append(luis_response[1])
            back_half_luis_sentiments.append(luis_response[2])
    return back_half_luis_predictions, back_half_luis_entities, back_half_luis_sentiments
    


def find_open_greeting(test_df, test_text_column, model_config, response):
    model_type = "model_1.1"
    lsi_conf_thresh = 0.5
    trim_min_perc = 0
    trim_max_perc = 0.2
    open_greeting_response = find_topic_cluster_LSI_LUIS(
        test_df = test_df,
        test_text_column = test_text_column,
        model_type = model_type,
        model_config = model_config,
        response = response,
        lsi_conf_thresh = lsi_conf_thresh,
        trim_min_perc = trim_min_perc,
        trim_max_perc = trim_max_perc,
        specific_topic = 'greeting'
        )
    if len(open_greeting_response) > 0:
        status_message("    1.1: Open Greeting -- Satisfied")
        response.open_greeting = open_greeting_response
    return response


def check_word_similarity_to_list(word, check_list, lev_threshold=3):
    sim_list = []
    for check_word in check_list:
        sim_list = np.append(sim_list, lev.distance(word, check_word))
    return len(sim_list[sim_list < lev_threshold])


def find_probing_questions(test_full_df, text_full_col, response):
    # Filter out all Question Answer Pairs from Display
    qna_pairs = find_qna_pair_from_display(test_full_df[text_full_col].iloc[0])
    binary_response_list = ['yes', 'nope', 'sure', 'correct', 'no']
    #   -- Check if answers are only with "yes" or "no" indicator (as percentage of response)
    answers = qna_pairs[0]
    answer_over_thresh = 0
    perc_thresh_of_a = 0.3
    for answer in answers:
        words_over_thresh = 0
        words = answer.split(' ')
        # If the response is long enough, we can assume the quesion was prompting a quality response
        if len(words) > 15:
            continue
        for word in words:
            if check_word_similarity_to_list(word, binary_response_list, 2) > 0:
                words_over_thresh += 1
        if words_over_thresh / len(words) > perc_thresh_of_a:
            answer_over_thresh += 1
    # Set cap of 65% confidence since this one is tough
    conf = min(0.65, 1 - answer_over_thresh / len(answers))
    if conf != 0:
        status_message("    2.1: Asks Probing Questions -- Satisfied")
        response.probing_questions['value'] =  True
        response.probing_questions['conf'] = conf
    return response


def find_hold_protocol(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    #   -- Use Delta to Find Hold Times
    #   -- Use LSI and LUIS to Find Hold
    test_full_df['sfd'] = test_full_df[text_full_col].apply(lambda x: re.split('\.|\!|\?', x))
    input_df = pd.DataFrame(test_full_df['sfd'].iloc[0], columns = ['sfd'])
    hold_intro_dict = recognize_hold_intro_lsi(input_df, 'sfd', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.6)
    if list(hold_intro_dict.values())[0] == 0:
        return response
    hold_outro_df = input_df.loc[list(hold_intro_dict.keys())[0]:]
    hold_outro_dict = recognize_hold_outro_lsi(hold_outro_df, 'sfd', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.6)
    if list(hold_outro_dict.values())[0] == 0:
        return response
    status_message("    3.1: Adheres to Hold Protocol -- Satisfied")
    response.protocol_hold['value'] =  True
    response.protocol_hold['conf'] = np.mean([list(hold_intro_dict.values()), list(hold_outro_dict.values())])
    return response


def find_transfer_protocol(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    # Assuming the following criteria need to be met: The phone number they are being transferred to is mentioned, and the transfering of the individual is explicitly mentioned
    # Find Transfer Context Using LSI
    # sdf : Sentences from Display
    test_full_df['sfd'] = test_full_df[text_full_col].apply(lambda x: re.split('\.|\!|\?', x))
    input_df = pd.DataFrame(test_full_df['sfd'].iloc[0], columns = ['sfd'])
    transfer_result_dict = recognize_transfer_request_lsi(input_df, 'sfd', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.5)
    # Find Phone Number Around Transfer Context
    sentence_buffer = 5
    transfer_confs = []
    for sentence_index in transfer_result_dict.keys():
        sentences_to_search = input_df['sfd'][sentence_index-sentence_buffer:sentence_index+sentence_buffer]
        num_len_roll_sum = sentences_to_search.apply(lambda x: len(re.sub("[^0-9]", "", x))).rolling(3).sum().fillna(0)
        if (num_len_roll_sum >= 10).sum() > 0:
            transfer_confs.append(transfer_result_dict[sentence_index])
    if len(transfer_confs) > 0:
        # Checks if there is a proper hold protocol in the call. Assuming that the transfer protocol led to that, we call it a success
        if response.protocol_hold['value']:     
            status_message("    3.2: Adheres to Transfer Protocol -- Satisfied")
            response.protocol_transfer['value'] =  True
            response.protocol_transfer['conf'] = np.mean(transfer_confs)
        else:
            status_message("    4.2: Provides all options when available -- Satisfied")
            response.resolve_all_options['value'] =  True
            response.resolve_all_options['conf'] = np.mean(transfer_confs)
    return response

def find_mfippa_protocol(response):
    return response

def find_accurate_info(test_full_df, text_full_col, response):
    return response

def find_all_option_resolution(response):
    # Slightly handled in Transfer Protocol
    return response

def find_open_service_req(response):
    return response

def find_all_info_service_req(response):
    return response

def find_website_promotion(test_full_df, text_full_col, response):
    lev_distances = [lev.distance(x, 'ottawa.ca') for x in test_full_df[text_full_col].iloc[0].split(' ')]
    lev_distance_list = pd.DataFrame(lev_distances).loc[[x < 3 for x in lev_distances]][0].values
    if len(lev_distance_list) > 0:
        status_message("    5.3 Promotes Ottawa.ca -- Satisfied")
        # Using 10 here because it already passed the threshold of less than 3. So 10 means the range of conf for that second part will be [0.7,1]
        # Using 4 for the first part, because if they are mentioning the website 4+ times, they are certainly promoting it
        website_conf = min(0.3, 0.3*(len(lev_distance_list) / 4)) + 0.7*(1 - (np.mean(lev_distance_list) / 10))
        response.serv_request_website = {'value': True, 'conf': website_conf}
    # This one is really tough knowing "whenever possible"
    #   -- Just Looking for usages of ottawa.ca for now
    return response

def find_service_req_num(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    #   -- Assuming this should happen each time? Otherwise tough to identify when it is necessary.
    #   -- We can identify the Service Request Number if given
    #   -- Identify numbers, ensure number given is of the proper format, search around that number for 'reference number' or 'service request number'
    test_full_df['sfd'] = test_full_df[text_full_col].apply(lambda x: re.split('\.|\!|\?', x))
    input_df = pd.DataFrame(test_full_df['sfd'].iloc[0], columns = ['sfd'])
    service_num_result_dict = recognize_service_request_number(input_df, 'sfd', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.5)
    # Find Service Number Around Service Number Request Context
    sentence_buffer = 1
    serv_req_num_confs = []
    for sentence_index in service_num_result_dict.keys():
        sentences_to_search = input_df['sfd'][sentence_index-sentence_buffer:sentence_index+sentence_buffer]
        num_len_roll_sum = sentences_to_search.apply(lambda x: len(re.sub("[^0-9]", "", x))).rolling(3).sum().fillna(0)
        if (num_len_roll_sum >= 10).sum() > 0:
            serv_req_num_confs.append(service_num_result_dict[sentence_index])
    if len(serv_req_num_confs) > 0:
        status_message("    5.4:Provides the Service Request Number -- Satisfied")
        response.serv_request_number['value'] =  True
        response.serv_request_number['conf'] = np.mean(serv_req_num_confs)
    return response

def find_provides_service_level(response):
    return response

def find_communicated_resolution(response):
    return response
    
def find_final_offer_assistance(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    final_sentence_df = test_full_df[text_full_col].to_frame()
    final_sentence_df['last_qna_pairs'] = final_sentence_df[text_full_col].apply(lambda x: acquire_final_qna(x))
    final_sentence_df['last_q'] = final_sentence_df['last_qna_pairs'].apply(lambda x: x[0][1] if len(x[0])==2 else (x[0][0] if len(x[0]) > 0 else ''))
    final_sentence_df['final_sentence'] = final_sentence_df['last_qna_pairs'].apply(lambda x: ['. '.join(x[1])] if len(x[0])==2 else x[1])
    final_sentence_df['final_sentence'] = final_sentence_df['final_sentence'].apply(lambda x: x[0] if len(x) == 1 else '')
    final_sentence_df['final_section'] = final_sentence_df['last_q'] + final_sentence_df['final_sentence']
    final_offer_assistance_conf = recognize_final_assistance(final_sentence_df, 'final_section', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.1)
    #   -- Check question and in sentence against LSI with friendly closing LSI model
    #   -- Or with LSI simple search ["can I help you with anything else", "is there anything else I can assist you with"]
    if final_offer_assistance_conf > 0.6:
        status_message("    7.1: Provides a Final Offer of Assistance -- Satisfied")
        response.close_final_assistance['value'] = True
        response.close_final_assistance['conf'] = final_offer_assistance_conf
    return response

def find_friendly_close(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    # Grab Closing Sentences
    final_sentence_df = test_full_df[text_full_col].to_frame()
    final_sentence_df['last_qna_pairs'] = final_sentence_df[text_full_col].apply(lambda x: acquire_final_qna(x))
    final_sentence_df['final_sentence'] = final_sentence_df['last_qna_pairs'].apply(lambda x: ['. '.join(x[1])] if len(x[0])==2 else x[1])
    final_sentence_df['final_sentence'] = final_sentence_df['final_sentence'].apply(lambda x: x[0] if len(x) == 1 else '')
    final_sentence_df['final_sentence_cut'] = final_sentence_df['final_sentence'].apply(lambda x: '.'.join(x.split('.')[-4:]).strip() if len(x.split('.')) >= 4 else x.strip())
    positive_close_conf = recognize_positive_close(final_sentence_df, 'final_sentence_cut', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.5)
    if positive_close_conf > 0.6:
        status_message("    7.2: Provides a Friendly Closing -- Satisfied")
        response.close_friendly['value'] = True
        response.close_friendly['conf'] = np.mean(positive_close_conf)
    return response

def find_willingness_to_help(test_full_df, text_full_col, response):
    # Asks Sufficient Questions
    # Check Num Questions
    # Check output of 7.2
    qna_pairs = find_qna_pair_from_display(test_full_df[text_full_col].iloc[0])
    questions = qna_pairs[0]
    if len(questions) < 5:
        return response
    if response.close_friendly['value']:
        status_message("    8.1: Displays Willingness to Help -- Satisfied")
        response.prof_respect_willing_to_help['value'] = True
        response.prof_respect_willing_to_help['conf'] = min(1, 0.5 + (len(questions) / 20) * 0.5)
    return response

def find_respect_to_client(back_half_luis_sentiments, response):
    # Giving people the benefit of the doubt here, only if sentiment is very negative will we revoke this one
    try:
        all_back_half_sentiment = np.array([i['score'] if i['label'] == 'positive' else -i['score'] for i in back_half_luis_sentiments])
    except:
        return response
    positive_sentiment_list = all_back_half_sentiment[(all_back_half_sentiment > 0.2)]
    negative_sentiment_list = all_back_half_sentiment[(all_back_half_sentiment <= 0.2)]
    polite_conf = 0
    if len(positive_sentiment_list) > 0:
        polite_conf = np.mean(np.append(positive_sentiment_list, np.zeros(len(negative_sentiment_list))+0.5))
    if len(negative_sentiment_list) > 5:
        polite_conf = 0
    if polite_conf > 0.5:
        status_message("    8.2: Treats the Client with Respect -- Satisfied")
        response.prof_respect_client_respect['value'] = True
        response.prof_respect_client_respect['conf'] = polite_conf
    return response

def find_positive_city_speech(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response):
    test_full_df['sfd'] = test_full_df[text_full_col].apply(lambda x: re.split('\.|\!|\?', x))
    input_df = pd.DataFrame(test_full_df['sfd'].iloc[0], columns = ['sfd'])
    positive_city_speech_dict = recognize_city_speech(input_df, 'sfd', sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, 0.6)
    luis_sentiment = []
    for positive_city_sentence_ind in positive_city_speech_dict.keys():
        luis_sentiment.append(get_luis_response(raw_utterance = input_df['sfd'].loc[positive_city_sentence_ind], specific_topic = 'greeting')[0][2])
    conf_list = []
    for sent in luis_sentiment:
        if sent['label'] == 'positive':
            conf_list.append(sent['score'])
    if len(conf_list) > 0:
        status_message("    8.3: Speaks Positively about the City -- Satisfied")
        response.prof_respect_positive_city['value'] = True
        response.prof_respect_positive_city['conf'] = np.mean(conf_list)
    return response

def find_polite_and_respect(back_half_luis_sentiments, response):
    try:
        all_back_half_sentiment = np.array([i['score'] if i['label'] == 'positive' else -i['score'] for i in back_half_luis_sentiments])
    except:
        return response
    positive_sentiment_list = all_back_half_sentiment[(all_back_half_sentiment > 0.2)]
    negative_sentiment_list = all_back_half_sentiment[(all_back_half_sentiment <= 0.2)]
    polite_conf = 0
    if len(positive_sentiment_list) > 0:
        polite_conf = np.mean(np.append(positive_sentiment_list, np.zeros(len(negative_sentiment_list))+0.5))
    if len(negative_sentiment_list) > 7:
        polite_conf = 0
    if polite_conf > 0.5:
        status_message("    8.4: Uses Polite Phrases -- Satisfied")
        response.prof_respect_polite['value'] = True
        response.prof_respect_polite['conf'] = polite_conf
    #   -- Take the back half 40%-90% of the call and assess
    #   -- This is assuming people will not start the call off on a negative note, and resolve their issues over the phone to then be kind in the last half of the call.
    #   -- Otherwise, can pass full call through LUIS and obtain sentiment
    return response

def find_active_listening(response):
    #   -- This one is very tough. Essentially just ensuring the operator is not speaking overtop of the client? Maybe Audio Analysis for this?
    return response


