import json
import pandas as pd
import http.client, urllib.request, urllib.parse, urllib.error, base64

#### THIS DOES NOT WORK UNDER THE PRICE MODEL F0 ####

sentence_path = './training_sentences/train_greeting/first_sentence.csv'
topic = 'greeting'

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '854b40f7472b4d798b685f293f297e75',
}

params = urllib.parse.urlencode({
})

sentences = pd.read_csv(sentence_path)
luis_utterance_body = [{'text': i[0], 'intentName': topic} for i in sentences.values[100:150]]

try:
    conn = http.client.HTTPSConnection('canadacentral.api.cognitive.microsoft.com')
    conn.request("POST", "/luis/api/v2.0/apps/{appId}/versions/{versionId}/examples?%s" % params, json.dumps(luis_utterance_body), headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

