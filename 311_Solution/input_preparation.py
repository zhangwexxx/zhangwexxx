import pandas as pd
from batch_transcription import transcribe
from utils.coo_311_utils import status_message
from azure.storage.blob import BlobClient, ContainerClient, ContentSettings
from convert_transcriptions_to_df import generate_dataframes

def move_blobs(conn_str, input_container_name, output_container_name, blob_name, upload_settings=None):
    src_blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=input_container_name, blob_name=blob_name)
    dst_blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=output_container_name, blob_name=blob_name)
    dst_blob.upload_blob(src_blob.download_blob().readall(), overwrite=True)

def gather_input(blob_keys, speech_keys, recordings_container_sas_uri, destination_container_sas_uri):
    ####### Transcribe Input Audio Files
    input_audio_container_name = 'input-audio'
    transcription_container_name = 'transcribed-audio'
    processed_audio_container_name = 'processed-audio'
    training_container_name = 'training-transcriptions'
    connection_string = blob_keys['storage_connection_string']
    input_audio_container = ContainerClient.from_connection_string(conn_str = connection_string, container_name = input_audio_container_name)
    if len(list(input_audio_container.list_blobs())) == 0:
        status_message("No Audio Files found in Input Blob!")
    # Generate Container SAS URIs
    transcribe(recordings_container_sas_uri, destination_container_sas_uri, speech_keys)
    ####### Move all Transcribed Audio Files to Final Destination Container
    status_message("Moving Complete Audio Files from Container {} to Container {}".format(input_audio_container_name, processed_audio_container_name))
    try:
        for audio_blob_ind in range(len(list(input_audio_container.list_blobs()))):
            if audio_blob_ind % 100 == 0:
                status_message("Moved {} / {} Blobs".format(audio_blob_ind, len(list(input_audio_container.list_blobs()))))
            audio_blob_name = list(input_audio_container.list_blobs())[audio_blob_ind]['name']
            move_blobs(connection_string, input_audio_container_name, processed_audio_container_name, audio_blob_name)
        input_audio_container.delete_blobs(*input_audio_container.list_blobs())
        status_message("Moving Audio Files Complete")
    except:
        status_message("Unable to move Blobs from {} to {}".format(input_audio_container_name, processed_audio_container_name))
    ####### Convert Transcriptions to DataFrame
    all_blob_full_df, all_blob_sentence_df = generate_dataframes(transcription_container_name, connection_string)
    all_blob_full_df['fullPhraseLex'] = all_blob_full_df['fullPhraseLex'].fillna('')
    all_blob_full_df['fullPhraseDisplay'] = all_blob_full_df['fullPhraseDisplay'].fillna('')
    all_blob_sentence_df = all_blob_sentence_df[all_blob_sentence_df['phraseIndex'] == 0]
    all_blob_sentence_df['lexical'] = all_blob_sentence_df['lexical'].fillna('')
    all_blob_sentence_df['display'] = all_blob_sentence_df['display'].fillna('')
    # Move Transcriptions to Training Folder
    transcription_container = ContainerClient.from_connection_string(conn_str = connection_string, container_name = transcription_container_name)
    training_container = ContainerClient.from_connection_string(conn_str = connection_string, container_name = training_container_name)
    try:
        for transcription_blob_ind in range(len(list(transcription_container.list_blobs()))):
            if transcription_blob_ind % 100 == 0:
                status_message("Moved {} / {} Blobs".format(transcription_blob_ind, len(list(transcription_container.list_blobs()))))
            transcription_blob_name = list(transcription_container.list_blobs())[transcription_blob_ind]['name']
            move_blobs(connection_string, transcription_container_name, training_container_name, transcription_blob_name)
        transcription_container.delete_blobs(*transcription_container.list_blobs())
        status_message("Moving Transcriptions Complete")
    except Exception as e:
        status_message("Unable to move Blobs from {} to {}".format(transcription_container_name, training_container_name))
        print(e)
    return all_blob_full_df, all_blob_sentence_df

