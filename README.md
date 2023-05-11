# End-to-End ASR-NER
* Speech Data: http://openslr.org/33/
* Aishell-NER (Input speech, output transcript and NE): https://github.com/Alibaba-NLP/AISHELL-NER
* CNERTA (Input speech and transcript, output NE): https://github.com/DianboWork/CNERTA

## Input speech, output transcript and NE
* Json Form
```python
    {"key": "BAC009S0724W0121", "wav": "your_path_to_this_wav_file/BAC009S0724W0121.wav", "txt": "广州市房地产中介协会分析", "ner_txt": "<广州市房地产中介协会>分析", "bio_lst": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O"]}
    {"key": "BAC009S0724W0122", "wav": "your_path_to_this_wav_file/BAC009S0724W0122.wav", "txt": "广州市房地产中介协会还表示", "ner_txt": "<广州市房地产中介协会>还表示", "bio_lst": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O"]}
```

## Install
```python
conda create --name speech python=3.8.13
conda activate speech
# check your cuda version and select the right cudatoolkit version from https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.11.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## Usage

* Train
```python
python -m main_ctcattasr train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/end2end_ner/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --device 0
```
* Inference
```python
python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --input data/end2end/test.json \
                     --path exp/end2end_ner \
                     --res end2end-test.pred \
                     --decode_mode attention \
                     --batch_size 16 \
                     --device 0
```

## Input speech and transcript, output NE
* Train
```python
python -m main_ctcbertner train --train data/sp_ner/new_train.json \
                     --dev data/sp_ner/new_valid.json \
                     --test data/sp_ner/new_test.json \
                     --path exp/debug/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --use_tokenized \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /your-bert-dir/bert-base-chinese \
                     --seed 999 \
                     --device 0
```
You can add "--if_flat" to fit the flat ner setting, default nested ner.

* Inference
```python
python -m main_ctcbertner evaluate --path exp/debug/ \
                     --input data/sp_ner/new_test.json \
                     --res pred.txt \
                     --use_same_tokenizer \
                     --use_tokenized \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /your-bert-dir/bert-base-chinese \
                     --device 0

```

* Results
```python
2023-05-11 18:22:11 INFO test:  4445

2023-05-11 18:23:21 INFO UP: 86.73% UR: 80.90% UF: 83.71% P: 84.56% R: 78.88% F: 81.62%
2023-05-11 18:23:21 INFO 0:01:09.917616s elapsed
```