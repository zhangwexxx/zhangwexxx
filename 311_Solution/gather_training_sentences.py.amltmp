import pandas as pd
from utils.coo_311_utils import lsi_model_req, status_message
from LSI_testing_functions import filter_on_q_LSI

######## GREETING ##########
test_df = pd.read_csv('./transcription_csv/phrase_options_df.csv')
test_df = test_df[test_df['phraseIndex'] == 0]

text_col = 'lexical'
test_df[text_col] = test_df[text_col].fillna('')
conf_thresh = 0.7
model_name = 'LSImodel_sentence_topic'
query = ['transfer']
topic = 'transfer'
# Get LSI Result
thresh_list = filter_on_q_LSI(
    query, 
    model_name,
    test_df = test_df,
    text_col = text_col,
    conf_thresh = conf_thresh, 
    )

# Write it to CSV
complete_list = []
for topic_list in thresh_list:
    complete_list.append(topic_list)

complete_list = [i[0] for i in complete_list[0]]
complete_list_df = test_df.iloc[complete_list]
complete_list_unique_df = pd.DataFrame(complete_list_df[text_col].unique(), columns = ['sentence'])

out_path = './training_sentences/' + topic + '_model/all_transer_sentences.csv'
complete_list_unique_df.to_csv(out_path, index=False)

