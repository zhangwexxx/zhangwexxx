class lsi_model_req(object):
    def __init__(self, dictionary, tfidf_matrix, lsi_base_corpus):
        self.dictionary = dictionary
        self.tfidf_matrix = tfidf_matrix
        self.lsi_base_corpus = lsi_base_corpus

def status_message(imsg):
    s = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    s = s[:-4]
    timestamp = '[' + s + ']'
    print(timestamp + ' ' + imsg)
