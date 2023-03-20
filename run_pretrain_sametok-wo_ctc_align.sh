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

python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-nestedner-lr1e5-s666/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0

python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-nestedner-lr1e5/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0

python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-flatner-lr1e5-s999/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --if_flat \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 999 \
                     --device 0

python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-flatner-lr1e5-s666/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --if_flat \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --seed 666 \
                     --device 0

python -m main_ctcbertner train --path exp/ctcbert-pretrain_sametok-wo_ctc_align-flatner-lr1e5/ \
                     --ctc_bert_path exp/ctcbert-onlyctc-sametokenizer/best.model \
                     --ctc_path None \
                     --use_same_tokenizer \
                     --if_flat \
                     --batch_size 16 \
                     --lr 1e-5 \
                     --config conf/ctc_mel80.yaml \
                     --bert /opt/data/private/slzhou/PLMs/bert-base-chinese \
                     --device 0