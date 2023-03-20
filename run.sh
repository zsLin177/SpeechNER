nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/debug/ \
                     --batch_size 16 \
                     --device 0 > debug.log 2>&1 &
                     (58042)

python -m main_asr evaluate --input data/aishell1_asr/test.json \
                     --path exp/debug \
                     --batch_size 8 \
                     --device 0


nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/debug2/ \
                     --batch_size 16 \
                     --device 0 > debug2.log 2>&1 &
                     (112705)

nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/debug3/ \
                     --batch_size 16 \
                     --device 0 > debug3.log 2>&1 &
                     (119878)

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text pred.txt > wer.txt

python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/debug4/ \
                     --add_bert \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --batch_size 16 \
                     --device 0

python -m main_asr evaluate --input data/aishell1_asr/test.json \
                     --path exp/ctc-proj768 \
                     --batch_size 8 \
                     --device 0


nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctc-proj768/ \
                     --batch_size 16 \
                     --device 0 > ctc-proj768.log 2>&1 &
                     (216867)


nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctc-nonlinearproj768/ \
                     --batch_size 16 \
                     --device 0 > ctc-nonlinearproj768.log 2>&1 &


nohup python -m main_asr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctc-linearproj768/ \
                     --batch_size 16 \
                     --device 0 > ctc-linearproj768.log 2>&1 &
                     (38547)


python -m main_asr evaluate --input data/aishell1_asr/test.json \
                     --path exp/ctc-linearproj768 \
                     --batch_size 8 \
                     --device 0


nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert/ \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --fix_bert \
                     --batch_size 16 \
                     --config conf/ctcbert_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert.log 2>&1 &
                     (62297)


python -m main_ctcbert evaluate --input data/aishell1_asr/test.json \
                     --path exp/ctcbert/ \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 8 \
                     --config conf/ctcbert_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-mtl/ \
                     --ctc_path None \
                     --fix_bert \
                     --with_ctc_loss \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-mtl.log 2>&1 &
                     (140338)

python -m main_ctcbert evaluate --input data/aishell1_asr/test.json \
                     --path exp/ctcbert-mtl/ \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 8 \
                     --config conf/ctcbert_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text pred.txt > wer.txt


nohup python -m main_ctcbertner train --path exp/ctcbert-flatner/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-flatner.log 2>&1 &
                     (175939)

nohup python -m main_ctcbertner train --path exp/ctcbert-flatner-lr1e5/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-flatner-lr1e5.log 2>&1 &


python -m main_ctcbertner evaluate --path exp/ctcbert-flatner-lr1e5/ \
                     --train data/sp_ner/new_train.json \
                     --input data/sp_ner/new_test.json \
                     --batch_size 16 \
                     --if_flat \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0


nohup python -m main_ctcbertner train --path exp/bert-flatner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > bert-flatner-lr1e5.log 2>&1 &
                     (32613)


nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-flatner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-nopretrain-flatner-lr1e5.log 2>&1 &
                     (74464)


nohup python -m main_ctcbertner train --path exp/ctcbert-0.05ctc-flatner-lr1e5/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --with_ctc_loss \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-0.05ctc-flatner-lr1e5.log 2>&1 &
                     (122797)


nohup python -m main_ctcbertner train --path exp/ctcbert-0.1ctc-flatner-lr1e5/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --with_ctc_loss \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-0.1ctc-flatner-lr1e5.log 2>&1 &
                     (147772)


nohup python -m main_ctcbertner train --path exp/bert-flatner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > bert-flatner-lr1e5-s666.log 2>&1 &
                     (164929)

nohup python -m main_ctcbertner train --path exp/bert-flatner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > bert-flatner-lr1e5-s999.log 2>&1 &
                     (165272)


nohup python -m main_ctcbertner train --path exp/ctcbert-flatner-lr1e5-s666/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-flatner-lr1e5-s666.log 2>&1 &
                     (199923)

nohup python -m main_ctcbertner train --path exp/ctcbert-flatner-lr1e5-s999/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-flatner-lr1e5-s999.log 2>&1 &

nohup python -m main_ctcbertner train --path exp/ctcbert-0.05ctc-flatner-lr1e5-s666/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --with_ctc_loss \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-0.05ctc-flatner-lr1e5-s666.log 2>&1 &
                     (2972)


nohup python -m main_ctcbertner train --path exp/ctcbert-0.05ctc-flatner-lr1e5-s999/ \
                     --ctc_bert_path exp/ctcbert-mtl/best.model \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --with_ctc_loss \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-0.05ctc-flatner-lr1e5-s999.log 2>&1 &
                     (16046)


nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-flatner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-nopretrain-flatner-lr1e5-s666.log 2>&1 &
                     (32570)


nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-flatner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-nopretrain-flatner-lr1e5-s999.log 2>&1 &
                     (43241)

nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-flatner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-pretrain-woalign-flatner-lr1e5.log 2>&1 &
                     (61555)

nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-flatner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-pretrain-woalign-flatner-lr1e5-s666.log 2>&1 &
                     (72361)


nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-flatner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --if_flat \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-pretrain-woalign-flatner-lr1e5-s999.log 2>&1 &
                     (118790)


nohup python -m main_ctcbertner train --path exp/bert-nestedner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > bert-nestedner-lr1e5.log 2>&1 &
                     (99084)


nohup python -m main_ctcbertner train --path exp/bert-nestedner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > bert-nestedner-lr1e5-s666.log 2>&1 &
                     (171481)


nohup python -m main_ctcbertner train --path exp/bert-nestedner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --text_only \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > bert-nestedner-lr1e5-s999.log 2>&1 &
                     (195731)


python -m main_ctcbertner evaluate --path exp/bert-nestedner-lr1e5/ \
                     --train data/sp_ner/new_train.json \
                     --input data/sp_ner/new_test.json \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0


nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-nestedner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-nopretrain-nestedner-lr1e5.log 2>&1 &
                     (220916)

nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-nestedner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-nopretrain-nestedner-lr1e5-s666.log 2>&1 &
                     (4496)

nohup python -m main_ctcbertner train --path exp/ctcbert-nopretrain-nestedner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path None \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-nopretrain-nestedner-lr1e5-s999.log 2>&1 &
                     (60667)


nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-nestedner-lr1e5/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-pretrain-woalign-nestedner-lr1e5.log 2>&1 &
                     (81356)


nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-nestedner-lr1e5-s666/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0 > ctcbert-pretrain-woalign-nestedner-lr1e5-s666.log 2>&1 &
                     (118560)


nohup python -m main_ctcbertner train --path exp/ctcbert-pretrain-woalign-nestedner-lr1e5-s999/ \
                     --ctc_bert_path None \
                     --ctc_path exp/ctc-linearproj768/best.model \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0 > ctcbert-pretrain-woalign-nestedner-lr1e5-s999.log 2>&1 &
                     (146087)


nohup python -m main_ctcattasr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcattasr/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --device 0 > ctcattasr.log 2>&1 &
                     (112897)


python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --input data/aishell1_asr/test.json \
                     --path exp/ctcattasr \
                     --res tmp.pred \
                     --decode_mode attention \
                     --batch_size 2 \
                     --beam_size 3 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text ctcatt-asr-att.pred > wer.txt

python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --input data/aishell1_asr/test.json \
                     --path exp/ctcattasr \
                     --res ctcatt-asr-ctcgreedy.pred \
                     --decode_mode ctc_greedy_search \
                     --batch_size 8 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text ctcatt-asr-ctcgreedy.pred > wer.txt


nohup python -m main_ctcattasr train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/end2end_ner/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --device 0 > end2end_ner.log 2>&1 &
                     (194088)


python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --input data/end2end/test.json \
                     --path exp/end2end_ner \
                     --res end2end-spner.pred \
                     --decode_mode attention \
                     --batch_size 16 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text end2end-sp.pred > wer.txt

nohup python -m main_ctcattasr train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/end2end_ner_fs20/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --frame_length 25 \
                     --frame_shift 20 \
                     --device 0 > end2end_ner_fs20.log 2>&1 &
                     (6692)


nohup python -m main_ctcattasr train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/end2end_ner_fs10_mfn512/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > end2end_ner_fs10_mfn512.log 2>&1 &
                     (44057)

python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --input data/end2end/test.json \
                     --path exp/end2end_ner_fs10_mfn512 \
                     --res end2end-spner.pred \
                     --decode_mode attention \
                     --batch_size 16 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/test.text end2end-sp.pred > wer.txt

nohup python -m main_bartend2end train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1/ \
                     --batch_size 32 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_stage1.log 2>&1 &
                     (104883)

python -m main_bartend2end evaluate --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --input data/end2end/test.json \
                     --path exp/bart_asr_stage1 \
                     --res bart-asr.pred \
                     --batch_size 16 \
                     --device 0


nohup python -m main_bartend2end train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1_debug/ \
                     --batch_size 32 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > exp/bart_asr_stage1_debug.log 2>&1 &
                     (172429)

nohup python -m main_bartend2end train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage2/ \
                     --batch_size 32 \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > exp/bart_asr_stage2.log 2>&1 &
                     (206722)

nohup python -m main_bartend2end train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1_debug/ \
                     --batch_size 32 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_stage1_debug.log 2>&1 &
                     (27232)

nohup python -m main_bartend2end train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1_debug_trainspeechnoaug/ \
                     --batch_size 32 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_stage1_debug_trainspeechnoaug.log 2>&1 &
                     (30471)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1/ \
                     --batch_size 16 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_stage1.log 2>&1 &
                     (84223)


nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_stage1_ctc/ \
                     --batch_size 16 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --add_ctc \
                     --device 0 > bart_asr_stage1_ctc.log 2>&1 &
                     (123271)


nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_onestage/ \
                     --batch_size 16 \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_onestage.log 2>&1 &
                     (198554)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_onestage_basedon_trainloss/ \
                     --batch_size 16 \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0 > bart_asr_onestage_basedon_trainloss.log 2>&1 &
                     (9010)

python -m main_bartspeech evaluate --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --input data/end2end/train.json \
                     --path exp/bart_asr_onestage_basedon_trainloss \
                     --res bart-asr.pred \
                     --batch_size 16 \
                     --device 0

python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --input data/aishell1_asr/test.json \
                     --path exp/ctcattasr \
                     --res att_asr_test.pred \
                     --decode_mode attention \
                     --batch_size 64 \
                     --beam_size 3 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/aishell1_asr/train.text att_asr_train.pred > wer.txt


python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_asrcorrect/ \
                     --batch_size 64 \
                     --seed 777 \
                     --config conf/bart_txt.yaml \
                     --device 0

python -m main_barttxt evaluate --config conf/bart_txt.yaml \
                     --input data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_asrcorrect/ \
                     --res bart_asrcorr_test.pred \
                     --batch_size 16 \
                     --beam_size 3 \
                     --device 0


python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/bart_asr_twostage_basedon_trainloss/ \
                     --batch_size 32 \
                     --first_step \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --device 0


nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --add_ctc \
                     --path exp/att_ctc_asr_with_bartinit/ \
                     --batch_size 16 \
                     --seed 777 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 512 \
                     --config conf/bart_speech_encoder.yaml \
                     --device 0 > att_ctc_asr_with_bartinit.log 2>&1 &
                     (108406)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/att_asr_with_bartinit/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/initbart_sp_asr.yaml \
                     --device 0 > att_asr_with_bartinit.log 2>&1 &
                     (180889)
                     
                     
nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/att_asr_bart_onestage-5Wwarm/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder.yaml \
                     --device 0 > att_asr_bart_onestage-5Wwarm.log 2>&1 &
                     (217950)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/base-bart-base-chinese \
                     --path exp/att_asr_with_bartinit-base/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/initbart-base_sp_asr.yaml \
                     --device 0 > att_asr_with_bart-base_init.log 2>&1 &
                     (9841)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/base-bart-base-chinese \
                     --path exp/att_asr_with_bartinit-base-5W/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/initbart-base_sp_asr.yaml \
                     --device 0 > att_asr_with_bartinit-base-5W.log 2>&1 &
                     (13088)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/base-bart-base-chinese \
                     --path exp/att_asr_with_bartinit-base-5W-bs64/ \
                     --batch_size 64 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/initbart-base_sp_asr.yaml \
                     --device 0 > att_asr_with_bartinit-base-5W-bs64.log 2>&1 &
                     (17713)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/asr_bart_firststage_proj/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --first_step \
                     --config conf/bart_speech_encoder.yaml \
                     --device 0 > asr_bart_firststage_proj.log 2>&1 &
                     (66750)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --bart_model None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/base-bart-base-chinese \
                     --path exp/att_asr_with_bartinit-base-proj/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/initbart-base-proj_sp_asr.yaml \
                     --device 0 > att_asr_with_bartinit-base-proj.log 2>&1 &
                     (75938)


nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_txt.yaml \
                     --device 0 > bart_ner.log 2>&1 &
                     (142261)

python -m main_barttxt evaluate --config conf/bart_txt.yaml \
                     --input data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner/ \
                     --res bart_ner_test.pred \
                     --batch_size 16 \
                     --beam_size 5 \
                     --e2ener \
                     --device 0

nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T5_wp8k/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner/best.model \
                     --tem 5.0 \
                     --device 0 > bart_ner_distilled_T5_wp8k.log 2>&1 &
                     (205892)

nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T3/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner/best.model \
                     --tem 3.0 \
                     --device 0 > bart_ner_distilled_T3.log 2>&1 &
                     (186171)


python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T3_hw0/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 3.0 \
                     --hard_weight 0.0 \
                     --device 0

nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_new/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_txt.yaml \
                     --device 0 > bart_ner_new.log 2>&1 &
                     (228049)


nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T2_hw0/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 2.0 \
                     --hard_weight 0.0 \
                     --device 0 > bart_ner_distilled_T2_hw0.log 2>&1 &
                     (5962)


nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T2_hw0.2/ \
                     --batch_size 64 \
                     --seed 1 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 2.0 \
                     --hard_weight 0.2 \
                     --device 0 > bart_ner_distilled_T2_hw0.2.log 2>&1 &
                     (16096)


nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T2_hw0.2_v1/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 2.0 \
                     --hard_weight 0.2 \
                     --device 0 > bart_ner_distilled_T2_hw0.2_v1.log 2>&1 &
                     (32783)

nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T2_hw0.2_v2/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt_v2.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 2.0 \
                     --hard_weight 0.2 \
                     --device 0 > bart_ner_distilled_T2_hw0.2_v2.log 2>&1 &
                     (53064)


nohup python -m main_barttxt train --train data/end2end/joint_correct_ner/asr_corr_ner-train.json \
                     --dev data/end2end/joint_correct_ner/asr_corr_ner-dev.json \
                     --test data/end2end/joint_correct_ner/asr_corr_ner-test.json \
                     --path exp/bart_ner_distilled_T2_hw0.2_v3/ \
                     --batch_size 64 \
                     --seed 777 \
                     --e2ener \
                     --config conf/bart_distill_txt_v3.yaml \
                     --distill \
                     --bart_config /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --bart /opt/data/private/slzhou/PLMs/bart-base-chinese \
                     --teach_model exp/bart_ner_new/best.model \
                     --tem 2.0 \
                     --hard_weight 0.2 \
                     --device 0 > bart_ner_distilled_T2_hw0.2_v3.log 2>&1 &
                     (77993)


python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --path exp/asr_bart_firststage_proj/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --first_step \
                     --config conf/bart_speech_encoder.yaml \
                     --device 0

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr /opt/data/private/slzhou/speech_help_ner/asr_help_ner/exp/bart_ner_distilled_T2_hw0.2_v1/best.model \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder.yaml \
                     --device 0 > bart_speechner.log 2>&1 &
                     (117310)


python -m main_bartspeech evaluate --input data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr /opt/data/private/slzhou/speech_help_ner/asr_help_ner/exp/bart_ner_distilled_T2_hw0.2_v1/best.model \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder.yaml \
                     --res bart_speechner_test.pred \
                     --device 0


nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr /opt/data/private/slzhou/speech_help_ner/asr_help_ner/exp/bart_ner_distilled_T2_hw0.2_v1/best.model \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner_v2/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder_v2.yaml \
                     --device 0 > bart_speechner_v2.log 2>&1 &
                     (217759)

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner_v2_nobart_asrcorr/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder_v2.yaml \
                     --device 0 > bart_speechner_v2_nobart_asrcorr.log 2>&1 &
                     (213355)


python -m main_bartspeech evaluate --input data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner_v2_nobart_asrcorr/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder_v2.yaml \
                     --res bart_speechner_test.pred \
                     --device 0

nohup python -m main_bartspeech train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --e2ener \
                     --bart_asrcorr None \
                     --bart_tokenizer /opt/data/private/slzhou/PLMs/small-bart-base-chinese \
                     --path exp/bart_speechner_v3_nobart_asrcorr/ \
                     --batch_size 16 \
                     --frame_length 25 \
                     --frame_shift 10 \
                     --max_frame_num 1024 \
                     --config conf/bart_speech_encoder_v3.yaml \
                     --device 0 > bart_speechner_v3_nobart_asrcorr.log 2>&1 &
                     (14166)


nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-onlyctc-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-onlyctc-sametokenizer.log 2>&1 &
                     (49899)

python -m main_ctcbert evaluate --input data/aishell1_asr/test.json \
                     --path exp/ctcbert-onlyctc-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --fix_bert \
                     --batch_size 8 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0


python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-nestedner-lr1e5-s999/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0


nohup bash run_pretrain_sametok-wo_ctc_align.sh > run_pretrain_sametok-wo_ctc_align.log 2>&1 &
(122964)

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-onlyalign-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --frame_shift 20 \
                     --batch_size 64 \
                     --config conf/ctc_mel80_2d8.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-onlyalign-sametokenizer.log 2>&1 &
                     (33548)

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_onlyalign-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --frame_shift 20 \
                     --batch_size 64 \
                     --config conf/ctc_mel80_2d8.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_onlyalign-sametokenizer.log 2>&1 &
                     (43473)

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_onlyalign_fs15_2d4-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --frame_shift 15 \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_onlyalign_fs15_2d4-sametokenizer.log 2>&1 &
                     (65083)


python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_onlyalign_fs15_2d4-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --fix_bert \
                     --frame_shift 15 \
                     --batch_size 8 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_onlyalign_fs20_2d4_nofixbert-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --frame_shift 20 \
                     --batch_size 16 \
                     --config conf/ctc_mel80_notfixbert.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_onlyalign_fs20_2d4_nofixbert-sametokenizer.log 2>&1 &
                     (73760)


python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_onlyalign_fs20_2d4_nofixbert-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --fix_bert \
                     --frame_shift 20 \
                     --batch_size 8 \
                     --config conf/ctc_mel80_notfixbert.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_ctcalign_fs20_2d4-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --with_ctc_loss \
                     --frame_shift 20 \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_ctcalign_fs20_2d4-sametokenizer.log 2>&1 &
                     (81398)


python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_ctcalign_fs20_2d4-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --with_align_loss \
                     --with_ctc_loss \
                     --fix_bert \
                     --frame_shift 20 \
                     --batch_size 8 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0
                     (这个切出来有点像样子，但是会有很多帧被分配给最后一个字符(sep))

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_lstmpdtor_ctcalign_fs20_2d4-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --use_lstm_predictor \
                     --with_align_loss \
                     --with_ctc_loss \
                     --frame_shift 20 \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_lstmpdtor_ctcalign_fs20_2d4-sametokenizer.log 2>&1 &
                     (159715)


python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_lstmpdtor_ctcalign_fs20_2d4-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --use_lstm_predictor \
                     --with_align_loss \
                     --with_ctc_loss \
                     --fix_bert \
                     --frame_shift 20 \
                     --batch_size 8 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

python -m main_ctcbertner train --path exp/ctcbert-use_toksprepr-pretrain_sametok-wo_ctc_align-nestedner-lr1e5-s999/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --use_tokenized \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0

nohup bash run_use_toksprepr_pretrain_sametok-wo_ctc_align.sh > run_use_toksprepr_pretrain_sametok-wo_ctc_align.log 2>&1 &
(45681)

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_ctcalignaddblank_fs25_2d4-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --bert_insert_blank \
                     --with_align_loss \
                     --with_ctc_loss \
                     --frame_shift 25 \
                     --batch_size 64 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_ctcalignaddblank_fs25_2d4-sametokenizer.log 2>&1 &
                     (143773)

python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_ctcalignaddblank_fs25_2d4-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --bert_insert_blank \
                     --with_align_loss \
                     --with_ctc_loss \
                     --fix_bert \
                     --frame_shift 25 \
                     --batch_size 3 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

nohup python -m main_ctcbert train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcbert-faster_alignaddblank_fs25_2d4-sametokenizer/ \
                     --ctc_path None \
                     --fix_bert \
                     --use_same_tokenizer \
                     --bert_insert_blank \
                     --with_align_loss \
                     --frame_shift 25 \
                     --batch_size 64 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0 > ctcbert-faster_alignaddblank_fs25_2d4-sametokenizer.log 2>&1 &
                     (174452)

python -m main_ctcbert evaluate --input data/aishell1_asr/dev.json \
                     --path exp/ctcbert-faster_alignaddblank_fs25_2d4-sametokenizer/ \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --bert_insert_blank \
                     --with_align_loss \
                     --fix_bert \
                     --frame_shift 25 \
                     --batch_size 3 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

python -m main_ctcattasr evaluate --config conf/ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --input data/end2end/dev.json \
                     --path exp/end2end_ner \
                     --res end2end-dev.pred \
                     --decode_mode attention \
                     --batch_size 16 \
                     --device 0

python tools/compute-wer.py --char=1 --v=1 \
      data/end2end/aishell_end2end_test.text end2end-test.pred > end2end-test-wer.txt

python -m main_ctcattasr train --train data/aishell1_asr/train.json \
                     --dev data/aishell1_asr/dev.json \
                     --test data/aishell1_asr/test.json \
                     --path exp/ctcattasr-debug/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/ctcattasr.yaml \
                     --device 0

python -m main_ctcattasr train --train data/end2end/train.json \
                     --dev data/end2end/dev.json \
                     --test data/end2end/test.json \
                     --add_context \
                     --train_context_vocab data/end2end/aishell_train_ner_allmost600.vocab \
                     --dev_context_vocab data/end2end/aishell_dev_ner_allmost300.vocab \
                     --path exp/context_ner_asr/ \
                     --batch_size 16 \
                     --seed 777 \
                     --config conf/context_ctcattasr.yaml \
                     --e2ener \
                     --char_dict data/end2end/ner_char_dict.txt \
                     --device 0

python -m main_ctcbertner evaluate --path exp/ctcbert-use_toksprepr-pretrain_sametok-wo_ctc_align-nestedner-lr1e5-s999/ \
                     --input data/sp_ner/new_test.json \
                     --use_same_tokenizer \
                     --use_tokenized \
                     --batch_size 16 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0