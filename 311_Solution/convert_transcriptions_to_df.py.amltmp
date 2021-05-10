# Join Azure ML to Azure Blob Storage
import ast
import json
import pandas as pd
from utils.coo_311_utils import status_message
from azure.storage.blob import ContainerClient, BlobClient

def parse_phrases(r):
    if isinstance(r['nBest'], str):
        phrase_options = ast.literal_eval(r['nBest'])
    else:
        phrase_options = r['nBest']
    pdf = pd.DataFrame.from_dict(phrase_options)
    pdf['blobName'] = r['blobName']
    pdf['offset'] = r['offset']
    pdf = pdf.sort_values('confidence',ascending=False)
    pdf['phraseIndex'] = list(range(len(pdf)))
    return(pdf)

def generate_dataframes(transcription_container_name,  conn_string):
    cont_client = ContainerClient.from_connection_string(conn_str=conn_string, container_name=transcription_container_name)
    num_blob = 0
    blob_list = list(cont_client.list_blobs())
    length_blobs = len(blob_list)
    full_transcription_df = pd.DataFrame()
    phrase_transcription_df = pd.DataFrame()
    for blob in blob_list:
        num_blob += 1
        if (num_blob%500 == 0):
            status_message("Converting Transcription to DataFrame: {} / {}".format(num_blob, length_blobs))
        # Read audio blobs in
        in_blob_name = blob['name']
        in_blob_client = BlobClient.from_connection_string(conn_str=conn_string, container_name=transcription_container_name, blob_name=in_blob_name)
        # Download text file
        status_message("    Downloading blob")
        transc_json = json.loads(in_blob_client.download_blob().content_as_text())
        if 'recognizedPhrases' in transc_json.keys():
            # Capture call-level info
            status_message("    Capturing call-level info")
            timestamp = transc_json['timestamp']
            fullTranscription_dict = [{
                'blobName': in_blob_name,
                'timestamp': timestamp,
                'duration': transc_json['duration'],
                'durationTicks': transc_json['durationInTicks'],
                'fullPhraseDisplay': transc_json['combinedRecognizedPhrases'][0]['display'],
                'fullPhraseLex': transc_json['combinedRecognizedPhrases'][0]['lexical'],
                'numPhrases': len(transc_json['recognizedPhrases'])
            }]
            status_message("    Appending call info")
            full_transcription_df = full_transcription_df.append(pd.DataFrame.from_dict(fullTranscription_dict))
            # Capture phrase-level info
            status_message("    Capturing phrase-level info")
            phrases = pd.DataFrame.from_dict(transc_json['recognizedPhrases'])
            phrases['blobName'] = in_blob_name
            phrases['timestamp'] = timestamp
            status_message("    Appending phrase info")
            phrase_transcription_df = phrase_transcription_df.append(phrases)
    # Create Phrase List by Phrase
    phrase_transcription_df = pd.concat(list(phrase_transcription_df.apply(parse_phrases,axis=1)))
    # full_transcription_file_name = './transcription_csv/full_transcription_df.csv'
    # # phrase_transcription_file_name = './transcription_csv/phrase_transcription_df.csv'
    # phrase_transcription_file_name = './transcription_csv/phrase_options_df.csv'
    # full_transcription_df.to_csv(full_transcription_file_name)
    # phrase_transcription_df.to_csv(phrase_transcription_file_name)
    status_message("Created Transcription DataFrames")
    return full_transcription_df, phrase_transcription_df

