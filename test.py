import torch
from supar.utils.fn import stripe
from transformers import BertTokenizer, BartForConditionalGeneration
from model.bart_speech_ner import BartForEnd2EndSpeechNER


if __name__ == '__main__':
    # dp = torch.arange(25).view(5, 5)

    tokenizer = BertTokenizer.from_pretrained("/opt/data/private/slzhou/PLMs/bart-base-chinese")
    # # model = BartForConditionalGeneration.from_pretrained("/opt/data/private/slzhou/PLMs/bart-base-chinese")
    # model = BartForEnd2EndSpeechNER.from_pretrained("/opt/data/private/slzhou/PLMs/bart-base-chinese")
    # print(model)

    # test poisson distribution
    rate = 3
    poisson = torch.distributions.Poisson(rate)
    samples = poisson.sample(sample_shape=(10,))
    print(samples)
    for i in range(10):
        a = poisson.sample(sample_shape=(1,))
        print(a)
        import pdb
        pdb.set_trace()
    
