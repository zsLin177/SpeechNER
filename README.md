# End-to-End ASR-NER

## Input speech, output transcript and NE
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