import pandas as pd
from datetime import datetime
from call_LUIS import get_luis_response
from message_utils import lsi_model_req, status_message
from LSI_testing_functions import trim_test_set, test_LSI_cluster_model

pd.options.mode.chained_assignment = None

# Read in Sentences
status_message("Reading in Sentences")
df = pd.read_csv('./transcription_csv/phrase_options_df.csv')
df = df[df['phraseIndex'] == 0]
df['lexical'] = df['lexical'].fillna('')

# blob_name = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506321_UCMAVoice_2020102817050870.mp3.json'
blob_name = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506795_UCMAVoice_2020102823190387.mp3.json'
# blob_name = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506336_UCMAVoice_2020102817070432.mp3.json'
# blob_name = 'd647a564-6497-45ae-9bd4-444fe5b0fca7/audio-adastra-processing/2449909_UCMAVoice_2020101414163493.mp3.json'
# blob_name = '3d67ef8c-3f66-4059-b138-79a7a10d6fb2/audio-adastra-processing/2489298_UCMAVoice_2020102409415730.mp3.json'

# Test on Single Blob
test_df = df[df['blobName'] == blob_name]

##########################
# RUN AGAINST LSI
##########################
# Training Info:
train_df = read_csv('./training_sentences/train_greeting/greeting_sentences.csv', header=False)
# train_df = df
# train_df = trim_test_set(train_df, trim_min_perc=0, trim_max_perc=0.2)
# train_df = train_df.groupby('blobName').agg({'lexical': lambda x: ' '.join(x)})
train_df = train_df.fillna('')
train_text_column = 'lexical'
write_new_model = True
num_topics = 3

# Obtain Sentence DF
test_df = trim_test_set(test_df, trim_min_perc=0, trim_max_perc=0.2)
# test_df = test_df.groupby('blobName').agg({'lexical': lambda x: ' '.join(x)})
model_folder_name = "greeting_model_dim3"
test_text_column = 'lexical'
test_conf_thresh = 0.1

lsi_result = test_LSI_cluster_model(
    test_df, 
    model_folder_name, 
    test_text_column = test_text_column, 
    test_conf_thresh = test_conf_thresh,
    train_df = train_df,
    train_text_column = train_text_column,
    write_new_model = write_new_model,
    num_topics = num_topics
    )

##########################
# RUN AGAINST LUIS
##########################
if len(lsi_result) == 0:
    status_message("No Results Found when running against model: {}".format(model_folder_name))

for index in lsi_result.keys():
    status_message("Running LUIS Against {} Sentence(s)".format(len(lsi_result)))
    luis_greeting_score, found_entities = get_luis_response(test_df, test_text_column, index)

##########################
# GATHER RESULTS
##########################
if len(lsi_result) > 0:
    status_message("Found Proper Greeting!")
    for response in lsi_result.keys():
        print("")
        response_conf = 100*(luis_greeting_score[response]*0.7 + lsi_result[response]*0.3)
        status_message("Messages Considered Greeting:")
        status_message("    {}".format(test_df[test_text_column].iloc[sentence]))
        status_message("        Confidence: {:.2f}".format(response_conf))
    if len(found_entities) == 0:
        print("")
        status_message("No Entities Found")
    for entity_type in found_entities:
        print("")
        status_message("Found Entities in Message(s):")
        status_message("    Entity: {}".format(entity_type['entity_type']))
        for entity_values in entity_type['values']:
            status_message("        Value: {}".format(entity_values['value']))
            status_message("        Confidence: {}".format(entity_values['score']))



###############################
#### DISPLAY TEXT FOR DEMO ####
# display_text_df = pd.read_csv("./transcription_csv/full_transcription_df.csv")
# display_text_df[display_text_df['blobName'] == blob_name]['fullPhraseLex'].iloc[0]
###############################

