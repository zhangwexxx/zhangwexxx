import pandas as pd
from message_utils import lsi_model_req, status_message
from LSI_testing_functions import trim_test_set, filter_on_q_LSI, write_threshold_csv

########################################################################################
# Read in Sentences
status_message("Reading in Sentences")
df = pd.read_csv('./transcription_csv/phrase_options_df.csv')
df = df[df['phraseIndex'] == 0]
df['lexical'] = df['lexical'].fillna('')

## SET PARAMETERS
## Get Sentences that match query
model_name = 'LSImodel_sentence_topic'
query = 'thank you for calling the city'
topic = 'greeting'
conf_thresh = 0.8
test_df = df

#########################
# Test on Single Blob
#########################
blob_name = '28e54f69-90a1-4efb-bc0d-4b7893af2dea/audio-adastra-processing/2506321_UCMAVoice_2020102817050870.mp3.json'
test_df = df[df['blobName'] == blob_name]
text_col = 'lexical'

#########################
# Trim sentences
#########################
# min_perc = 0
# max_perc = 0.1
# trimmed_df = trim_test_set(df, min_perc, max_perc)
# # DF to Use
# test_df = trimmed_df


# Get LSI Result
thresh_list = filter_on_q_LSI(
    query, 
    model_name,
    test_df = test_df,
    text_col = text_col,
    conf_thresh = conf_thresh
    )

# Gather Phrases from LSI Result
thresh_list_df = pd.DataFrame(thresh_list, columns = ["index", "corr_conf"])
thresh_list_df['phrase'] = thresh_list_df['index'].apply(lambda x: test_df.iloc[x]['lexical'])
thresh_list_df = thresh_list_df.drop_duplicates(subset=['phrase'])

# Write Threshold
write_threshold_csv(thresh_list_df, topic)


# Gather Sentences
# df['phraseIndexByBlob'] = df.groupby(['blobName']).cumcount() + 1
# df[df['phraseIndexByBlob'] > 1][df['phraseIndexByBlob'] < 5]['lexical'].to_csv('./training_sentences/train_greeting/not_greeting.csv', index=False)
