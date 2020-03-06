# coding:utf-8

'''
March 2019 by Chen Jun
https://github.com/chenjun2hao/Attention_ocr.pytorch

'''

import torch, sys
from torch.autograd import Variable
import src.utils
import src.dataset
from PIL import Image
import models.crnn_lang as crnn
import argparse

use_gpu = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help="path to encoder")
    parser.add_argument('--decoder', type=str, help='path to decoder')
    parser.add_argument('--image', type=str, help='path to image')

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=992, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=32, help='size of the lstm hidden state')
    parser.add_argument('--max_width', type=int, default=500, help='the width of the featuremap out from cnn')
    parser.add_argument('--alphabet', type=str, default="Num", help='Vocabulary type: Num, NumAlpha...')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()

    if not opt.alphabet:
        from src.utils import alphabet
    elif opt.alphabet == "Num":
        alphabet = "0123456789"
    elif opt.alphabet == "NumSpace":
        alphabet = "0123456789 "
    elif opt.alphabet == "NumAlpha":
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    elif opt.alphabet == "NumAlphaSpace":
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz "
    else:
        print(" ** Error in alphabet:",opt.alphabet)
        sys.exit(1)

    encoder_path = opt.encoder
    decoder_path = opt.decoder
    img_path = opt.image
    max_length = 2
    EOS_TOKEN = 1
    hidden = opt.nh
    width = opt.imgW
    height = opt.imgH
    nclass = len(alphabet) + 3
    print(" Classes ",nclass,alphabet)

    encoder = crnn.CNN(height, 1, opt.nh)
    # decoder = crnn.decoder(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
    decoder = crnn.decoderV2(hidden, nclass)
    print(encoder)
    print(decoder)

    if encoder_path and decoder_path:
        print('loading pretrained models ......')
        try:
            encoder.load_state_dict(torch.load(encoder_path))
        except RuntimeError as e:
            print("** ERROR loading encoder: ",encoder_path)
            print(e)
            sys.exit(1)
        try:
            decoder.load_state_dict(torch.load(decoder_path))
        except RuntimeError as e:
            print("** ERROR loading decoder: ",decoder_path)
            print(e)
            sys.exit(1)
    else:
        print(" ** ERROR with paths")
        sys.exit(0)
    if torch.cuda.is_available() and use_gpu:
        encoder = encoder.cuda()
        decoder = decoder.cuda()


    converter = src.utils.strLabelConverterForAttention(alphabet)

    transformer = src.dataset.resizeNormalize((width, height))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available() and use_gpu:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    encoder.eval()
    decoder.eval()
    encoder_out = encoder(image)

    decoded_words = []
    prob = 1.0
    decoder_attentions = torch.zeros(max_length, width//8)
    decoder_input = torch.zeros(1).long()
    decoder_hidden = decoder.initHidden(1)
    if torch.cuda.is_available() and use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_hidden = decoder_hidden.cuda()
    loss = 0.0

    for di in range(max_length):  # 最大字符串的长度
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        #decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        if ni == EOS_TOKEN:
            # decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(converter.decode(ni))

    words = ''.join(decoded_words)
    prob = prob.item()
    print('predict_str:%-20s => prob:%-20s' % (words, prob))
