wget https://openslr.elda.org/resources/12/train-clean-100.tar.gz
wget https://openslr.elda.org/resources/12/test-clean.tar.gz
wget https://openslr.elda.org/resources/12/dev-clean.tar.gz
tar -zxvf train-clean-100.tar.gz
tar -zxvf test-clean.tar.gz
tar -zxvf dev-clean.tar.gz
python move_flac_to_wav.py