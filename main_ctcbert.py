import argparse
from asr import CTCBertParser
from supar.utils.logging import init_logger, logger
from supar.utils import Config
import os
import torch

def parse(parser):
    parser.add_argument('--path', help='path to model file')
    parser.add_argument('--device',
                        '-d',
                        default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_workers',
                        default=8,
                        type=int)
    parser.add_argument('--char_dict', 
                        default='data/sp_ner/chinese_char.txt', 
                        help='path to the char dict file')
    parser.add_argument('--cmvn', 
                        default='data/sp_ner/global_cmvn_mel80', 
                        help='global cmvn file')
    parser.add_argument('--config', 
                        default='conf/ctc_mel80.yaml', 
                        help='config file')
    parser.add_argument('--bert', 
                        default='bert-base-chinese', 
                        help='which bert model to use')
    parser.add_argument('--use_same_tokenizer', action='store_true', help='whether the asr part use the bert tokenizer')
    parser.add_argument('--fix_bert', action='store_true', help='whether to fix bert during training')
    parser.add_argument('--frame_shift',
                        default=10,
                        type=int)
    parser.add_argument('--with_align_loss', action='store_true', help='whether to simultaneously use align loss and ctc loss')
    parser.add_argument('--with_ctc_loss', action='store_true', help='whether to simultaneously use align loss and ctc loss')
    parser.add_argument('--use_lstm_predictor', action='store_true', help='')
    parser.add_argument('--bert_insert_blank', action='store_true', help='')


    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.manual_seed(args.seed)
    init_logger(logger, f"{args.path}{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = CTCBertParser(args)
        logger.info(f'{parser.model}\n')
        parser.train()
    elif args.mode == 'evaluate':
        parser = CTCBertParser(args)
        logger.info(f'{parser.model}\n')
        parser.load_model()
        parser.eval_asr()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--train', default='data/sp_ner/new_train.json', help='path to train file')
    subparser.add_argument('--dev', default='data/sp_ner/new_valid.json', help='path to dev file')
    subparser.add_argument('--test', default='data/sp_ner/new_test.json', help='path to test file')
    subparser.add_argument('--ctc_path', default='exp/ctc-linearproj768/best.model', help='path to the pre-trained ctc asr model')

    subparser = subparsers.add_parser('evaluate', help='Evaluation.')
    subparser.add_argument('--input', default='data/aishell1_asr/test.json', help='path to input file')
    subparser.add_argument('--ctc_path', default='exp/ctc-linearproj768/best.model', help='path to the pre-trained ctc asr model')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')


    parse(parser)
    
