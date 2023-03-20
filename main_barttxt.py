import argparse
from asr import BartASRCorrectionParser
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
    parser.add_argument('--e2ener', 
                        action='store_true',
                        help='whether it is an e2ener model')
    parser.add_argument('--config', 
                        default='conf/bart_txt.yaml', 
                        help='config file')
    parser.add_argument('--bart', 
                        default='/opt/data/private/slzhou/PLMs/bart-base-chinese', 
                        help='which bart model to use')
    parser.add_argument('--bart_config', 
                        default='/opt/data/private/slzhou/PLMs/bart-base-chinese', 
                        help='which bart model to use')
    parser.add_argument('--distill', 
                        action='store_true', 
                        help='')

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    torch.manual_seed(args.seed)
    init_logger(logger, f"{args.path}{args.mode}.log")

    if args.mode == 'train':
        parser = BartASRCorrectionParser(args)
        logger.info(f'{parser.model}\n')
        parser.train()
    elif args.mode == 'evaluate':
        parser = BartASRCorrectionParser(args)
        logger.info(f'{parser.model}\n')
        parser.load_model()
        parser.eval()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--train', default='data/end2end/joint_correct_ner/asr_corr_ner-train.json', help='path to train file')
    subparser.add_argument('--dev', default='data/end2end/joint_correct_ner/asr_corr_ner-dev.json', help='path to dev file')
    subparser.add_argument('--test', default='data/end2end/joint_correct_ner/asr_corr_ner-test.json', help='path to test file')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')
    subparser.add_argument('--beam_size', default=3, type=int, help='beam size')
    subparser.add_argument('--tem', default=5.0, type=float, help='Tempurture')
    subparser.add_argument('--teach_model', default='None', help='path to teacher model file')
    subparser.add_argument('--hard_weight', default=0.5, type=float, help='')


    subparser = subparsers.add_parser('evaluate', help='Evaluation.')
    subparser.add_argument('--input', default='data/aishell1_asr/test.json', help='path to input file')
    subparser.add_argument('--res', default='pred.txt', help='path to input file')
    subparser.add_argument('--beam_size', default=10, type=int, help='beam size')

    parse(parser)
    
