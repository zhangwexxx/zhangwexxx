import pandas as pd
from utils.coo_311_utils import lsi_model_req, status_message, find_qna_pair_from_display, acquire_final_qna

full_df = pd.read_csv('./transcription_csv/full_transcription_df.csv')
full_df['fullPhraseLex'] = full_df['fullPhraseLex'].fillna('')
full_df['fullPhraseDisplay'] = full_df['fullPhraseDisplay'].fillna('')

full_df['last_qna_pairs'] = full_df['fullPhraseDisplay'].apply(lambda x: acquire_final_qna(x))

full_df['final_sentence'] = full_df['last_qna_pairs'].apply(lambda x: ['. '.join(x[1])] if len(x[0])==2 else x[1])

final_sentence_df = full_df['final_sentence'].apply(lambda x: x[0] if len(x) == 1 else '')
final_sentence_df = final_sentence_df.to_frame()
final_sentence_df['final_sentence_cut'] = final_sentence_df['final_sentence'].apply(lambda x: '.'.join(x.split('.')[-3:]).strip() if len(x.split('.')) > 3 else x.strip())

final_sentence_df['final_sentence_cut'].to_csv("./training_sentences/train_ending/all_ending_sentences.csv", index=False)
