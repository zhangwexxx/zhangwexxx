import requests
import itertools
import numpy as np
from utils.coo_311_utils import status_message

"""
# Sample Response
{'query': '<TEXT>', 'prediction': {'topIntent': 'None', 'intents': {'None': {'score': 0.6538619}, 'greeting': {'score': 0.29723984}}, 'entities': {'url': ['ottawa.ca'], '$instance': {'url': [{'type': 'builtin.url', 'text': 'ottawa.ca', 'startIndex': 63, 'length': 9, 'modelTypeId': 2, 'modelType': 'Prebuilt Entity Extractor', 'recognitionSources': ['model']}]}}, 'sentiment': {'label': 'positive', 'score': 0.7989545}}}
"""

def handle_multiple_luis_responses(luis_responses):
    found_predictions = []
    found_entities = []
    found_sentiments = []
    for luis_response in luis_responses:
        found_predictions.append(luis_response[0])
        found_entities.append(luis_response[1])
        found_sentiments.append(luis_response[2])
    # Gather LUIS Prediction
    final_prediction = {}
    for found_prediction in found_predictions:
        for prediction_index in found_prediction.keys():
            if prediction_index in final_prediction.keys():
                final_prediction[prediction_index] = max(found_prediction[prediction_index], final_prediction[prediction_index])
            else:
                final_prediction[prediction_index] = found_prediction[prediction_index]
    # Flatten LUIS Entities List
    found_entities = list(itertools.chain(*found_entities))
    # Gather LUIS Sentiment
    final_sentiment = {'sentiment_score': 0}
    for found_sentiment in found_sentiments:
        if len(found_sentiment) > 0:
            if found_sentiment['label'] == 'positive':
                final_sentiment['sentiment_score'] += found_sentiment['score']
            else:
                final_sentiment['sentiment_score'] -= found_sentiment['score']
    return final_prediction, found_entities, final_sentiment

def get_luis_response(raw_utterance = '', test_df=None, test_text_column=None, index=None, specific_topic = ''):
    # print(test_df)
    # print(test_text_column)
    # print(index)
    if (test_df is None or test_text_column is None or index is None) and raw_utterance == '':
        return [{}, [], {}]
    if raw_utterance == '':
        utterance = test_df[test_text_column].loc[index]
    else:
        utterance = raw_utterance
        index = 0
    # Handling Utterances Over the 500 Character LUIS Limit
    utterance_list = [utterance[:500]]
    for i in range(int(np.round(len(utterance)/500, 0))):
        utterance_list.append(utterance[500*(i+1):500*(i+2)])
    all_LUIS_responses = []
    for sub_utterance in utterance_list:
        if sub_utterance == '':
            continue
        try:
            # status_message("    Calling LUIS with sentence: {}".format(sub_utterance))
            appId = 'd82c95af-1e89-4516-9457-11a641fba835'
            prediction_key = '3f25a21593a1470eac4b70b270fb682a'
            prediction_endpoint = 'https://canadacentral.api.cognitive.microsoft.com/'
            headers = {
            }
            params ={
                'query': sub_utterance,
                'timezoneOffset': '0',
                'verbose': 'true',
                'show-all-intents': 'true',
                'spellCheck': 'false',
                'staging': 'false',
                'subscription-key': prediction_key
            }
            # response = requests.get(f'{prediction_endpoint}luis/prediction/v3.0/apps/{appId}/slots/production/predict', headers=headers, params=params)
            # luis_response = response.json()
            luis_response = {'query': 'thank you for holding', 'prediction': {'topIntent': 'hold_out', 'intents': {'hold_out': {'score': 0.99862957}, 'greeting': {'score': 0.00432524551}, 'hold_in': {'score': 0.00259633036}, 'final_assistance': {'score': 0.00248763757}, 'transfer_request': {'score': 0.00142020651}, 'None': {'score': 0.000343273481}}, 'entities': {}, 'sentiment': {'label': 'positive', 'score': 0.9720607}}}
            luis_greeting_score = {}
            found_entities = []
            found_sentiment = {}
            if specific_topic in luis_response['prediction']['intents'].keys():
                # Gather Specific Intent Prediction
                luis_greeting_score[index] = luis_response['prediction']['intents'][specific_topic]['score']
            else:
                # Gather Top Intent Prediction
                luis_greeting_score[luis_response['prediction']['topIndent']] = luis_response['prediction']['intents'][luis_response['prediction']['topIndent']]['score']
            # Gather Entities
            if len(luis_response['prediction']['entities']) > 0:
                entity_response = luis_response['prediction']['entities']['$instance']
                for entity_type in entity_response.keys():
                    new_entity = {}
                    new_entity['entity_type'] = entity_type
                    for entity in entity_response[entity_type]:
                        entity_values = []
                        entity_values.append({'value': entity['text'], 'score': entity['score']})
                        new_entity['values'] = entity_values
                    found_entities.append(new_entity)
            # Gather Sentiment
            if 'sentiment' in luis_response['prediction'].keys():
                found_sentiment = luis_response['prediction']['sentiment']
            all_LUIS_responses.append([luis_greeting_score, found_entities, found_sentiment])
        except Exception as e:
            print(f'{e}')
            return [{}, [], {}]
    return all_LUIS_responses
