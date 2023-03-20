import argparse
from asr import Parser
from supar.utils.logging import init_logger, logger
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
    parser.add_argument('--add_bert',
                        action='store_true', 
                        help='whether to add bert')
    parser.add_argument('--bert', 
                        default='bert-base-chinese', 
                        help='which bert model to use')

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.manual_seed(args.seed)
    init_logger(logger, f"{args.path}{args.mode}.log")

    if args.mode == 'train':
        parser = Parser(args)
        logger.info(f'{parser.model}\n')
        parser.train()
    elif args.mode == 'evaluate':
        parser = Parser(args)
        logger.info(f'{parser.model}\n')
        parser.load_model()
        parser.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--train', default='data/sp_ner/new_train.json', help='path to train file')
    subparser.add_argument('--dev', default='data/sp_ner/new_valid.json', help='path to dev file')
    subparser.add_argument('--test', default='data/sp_ner/new_test.json', help='path to test file')

    subparser = subparsers.add_parser('evaluate', help='Evaluation.')
    subparser.add_argument('--input', default='data/aishell1_asr/test.json', help='path to input file')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')


    parse(parser)
    
