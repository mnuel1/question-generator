python -m venv {foldername}


python -m nltk.downloader universal_tagset
python -m spacy download en 

wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
