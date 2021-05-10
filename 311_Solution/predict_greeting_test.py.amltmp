import pandas as pd
from datetime import datetime
from call_LUIS import get_luis_response
from utils.coo_311_utils import lsi_model_req, status_message
from LSI_testing_functions import test_LSI_cluster_model, trim_test_set

pd.options.mode.chained_assignment = None

# Read in Sentences
status_message("Reading in Sentences")
df = pd.read_csv('./transcription_csv/phrase_options_df.csv')
df = df[df['phraseIndex'] == 0]
df['lexical'] = df['lexical'].fillna('')

blob_name1 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506363_UCMAVoice_2020102817165384.mp3.json'
# blob_name1 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506321_UCMAVoice_2020102817050870.mp3.json'
# blob_name2 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506795_UCMAVoice_2020102823190387.mp3.json'
# blob_name3 = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506336_UCMAVoice_2020102817070432.mp3.json'
# blob_name4 = 'd647a564-6497-45ae-9bd4-444fe5b0fca7/audio-adastra-processing/2449909_UCMAVoice_2020101414163493.mp3.json'
# blob_name5 = '3d67ef8c-3f66-4059-b138-79a7a10d6fb2/audio-adastra-processing/2489298_UCMAVoice_2020102409415730.mp3.json'

test_blob_names = {}
test_blob_names[blob_name1] = True
# test_blob_names[blob_name2] = False
# test_blob_names[blob_name3] = True
# test_blob_names[blob_name4] = True
# test_blob_names[blob_name5] = True

# Training Info:
train_df = pd.read_csv("./training_sentences/train_greeting/small_greeting_sentences.csv", header=None)
train_df = train_df.fillna('')
train_text_column = 0
write_new_model = True
num_topics = 3

test_text_column = 'lexical'
test_conf_thresh = 0.7


test_model_folder_names = [
    # 'first_sentence_model_dim10',
    # 'greeting_model_dim3',
    # 'greeting_model_dim10',
    # 'small_greeting_model_dim2',
    'small_greeting_model_dim3'
    ]

test_result = pd.DataFrame()
blob_name_dict = dict(zip(test_blob_names, range(len(test_blob_names))))
for model_folder_name in test_model_folder_names:
    for blob_name in test_blob_names.keys():
        # Test on Single Blob
        test_df = df[df['blobName'] == blob_name]
        ##########################
        # RUN AGAINST LSI
        ##########################
        # Obtain Sentence DF
        test_df = trim_test_set(test_df, trim_min_perc=0, trim_max_perc=0.2)
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
        sentences_returned = [test_df.loc[key]['lexical'] for key in lsi_result.keys()]
        sentences_returned_dict = dict(zip(lsi_result.keys(), sentences_returned))
        test_result = test_result.append(pd.DataFrame({
            'lsi_result': lsi_result,
            'model': model_folder_name,
            'blob_name': blob_name_dict[blob_name],
            'expected_result': test_blob_names[blob_name],
            'sentences': sentences_returned_dict
            }))

