from configparser import ConfigParser
from response_field_functions import *
from utils.coo_311_utils import lsi_model_req, status_message, find_qna_pair_from_display

def fill_response_fields(test_sentence_df, test_full_df, text_sentence_col, text_full_col, document_filter_lsi, document_filter_dict, document_filter_tfidf, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, config, response):
    model_config = config["MODELS"]

    # Gather QnA Responses from Display Text
    found_questions, found_answers = find_qna_pair_from_display(test_full_df[text_full_col].iloc[0])
    # Pass Back Half of Call to LUIS
    # I'm preferring the Display Transcription to the Lex one, so I will split based off Display
    # Alternate approach: Split based on Lex and Based on Timing of the Call (Take % Cut based of timing)
    back_half_luis_predictions, back_half_luis_entities, back_half_luis_sentiments = get_back_half_luis_response(test_full_df, text_full_col)
    # back_half_luis_predictions, back_half_luis_entities, back_half_luis_sentiments = [[{None: 0.9888562}, {None: 0.9888562}, {None: 0.9888562}], [[{'entity_type': '311_operator', 'values': [{'value': 'natasha', 'score': 0.991716862}]}], [{'entity_type': '311_operator', 'values': [{'value': 'natasha', 'score': 0.991716862}]}], [{'entity_type': '311_operator', 'values': [{'value': 'natasha', 'score': 0.991716862}]}]], [{'label': 'positive', 'score': 0.9831325}, {'label': 'positive', 'score': 0.9831325}, {'label': 'positive', 'score': 0.9831325}]]

    # Fill 1.1 #
    # Proper Greeting
    response = find_open_greeting(
        test_df = test_sentence_df,
        test_text_column = text_sentence_col,
        model_config = model_config,
        response = response)

    # Fill 1.2 #
    # Proper Language
    status_message("    Ignoring 1.2")
    
    # Fill 2.1 #
    # Probing Questions
    response = find_probing_questions(
        test_full_df, 
        text_full_col, 
        response)
   
    # Fill 3.1 #
    # On Hold Protocol
    response = find_hold_protocol(
        test_full_df, 
        text_full_col, 
        sentence_filter_lsi, 
        sentence_filter_dict, 
        sentence_filter_tfidf, 
        response)
    
    # Fill 3.2 #
    # Transfer Protocol
    response = find_transfer_protocol(
        test_full_df, 
        text_full_col,
        sentence_filter_lsi,
        sentence_filter_dict,
        sentence_filter_tfidf,
        response)
    
    # Fill 3.3
    # MFIPPA Protocol
    # Check whether or not binary response was given to personal question
    response = find_mfippa_protocol(response)
    
    # Fill 3.4 #
    # Councellor Checkbox Protocol
    status_message("    Ignoring 3.4")
    
    # Fill 4.1 ---------------------------
    # Provides Accurate Information
    response = find_accurate_info(test_full_df, text_full_col, response)
    
    # Fill 4.2
    # Resolves All Options
    response = find_all_option_resolution(response)
    
    # Fill 5.1
    # Open Service Request
    # Check for Open Request phrases
    response = find_open_service_req(response)
    
    # Fill 5.2
    # Service Request Provides All Info
    response = find_all_info_service_req(response)

    # Fill 5.3 #
    # Promotes Ottawa.ca Whenever Possible
    response = find_website_promotion(test_full_df, text_full_col, response)

    # Fill 5.4 #
    # Provides the Service Request Number
    response = find_service_req_num(
        test_full_df, 
        text_full_col, 
        sentence_filter_lsi, 
        sentence_filter_dict, 
        sentence_filter_tfidf, 
        response)
    
    # Fill 6.1
    # Provides Service Level Client, when applicable
    response = find_provides_service_level(response)

    # Fill 6.2
    # Communicates the Resolution to the Client
    response = find_communicated_resolution(response)

    # Fill 7.1 #
    # Provides Final Offer of Assistance
    response = find_final_offer_assistance(
        test_full_df, 
        text_full_col, 
        sentence_filter_lsi, 
        sentence_filter_dict, 
        sentence_filter_tfidf, 
        response)
    
    # Fill 7.2 #
    # Uses a Friendly Closing
    response = find_friendly_close(
        test_full_df, 
        text_full_col, 
        sentence_filter_lsi, 
        sentence_filter_dict, 
        sentence_filter_tfidf, 
        response)
    
    # Fill 8.1
    # Displays Willingness to Help
    response = find_willingness_to_help(test_full_df, text_full_col, response)
    
    # Fill 8.2 #
    # Treats the Client with Respect
    response = find_respect_to_client(back_half_luis_sentiments, response)
    
    # Fill 8.3
    # Speaks about the City and its Employees in a Positive Manner
    response = find_positive_city_speech(test_full_df, text_full_col, sentence_filter_lsi, sentence_filter_dict, sentence_filter_tfidf, response)
    
    # Fill 8.4
    # Respectful and Polite Phrases
    response = find_polite_and_respect(back_half_luis_sentiments, response)

    # Fill 8.5
    # Active Listener
    response = find_active_listening(response)
    
    return response
