import sys
import subprocess

def import_required_libraries():
    subprocess.call([sys.executable, "-m", "pip", "install", "azure-storage-blob"])
    subprocess.call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])
    subprocess.call([sys.executable, "-m", "pip", "install", "gensim"])
    subprocess.call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    subprocess.call([sys.executable, "-m", "pip", "install", "nltk"])
