# coding:utf-8

'''
March 2019 by Chen Jun
https://github.com/chenjun2hao/Attention_ocr.pytorch

'''
import os
import torch, sys
from torch.autograd import Variable
import src.utils
import src.dataset
from PIL import Image
import models.crnn_lang as crnn
from models.summary import model_summary
import argparse

use_gpu = True
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help="path to encoder")
    parser.add_argument('--decoder', type=str, help='path to decoder')
    parser.add_argument('--image', nargs='+', type=str, help='path to image')

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=992, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=32, help='size of the lstm hidden state')
    parser.add_argument('--max_width', type=int, default=500, help='the width of the featuremap out from cnn')
    parser.add_argument('--alphabet', type=str, default="Num", help='Vocabulary type: Num, NumAlpha...')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose mode')
    parser.add_argument('--logfile', help='Write results in csv file')
    opt = parser.parse_args()


    alphabet = src.utils.getAlphabetStr(opt.alphabet)

    encoder_path = opt.encoder
    decoder_path = opt.decoder
    img_paths = []
    for i in opt.image:
        if os.path.isdir(i):
            pimages = [i+os.sep+x for x in os.listdir(i) if x.split(".")[-1].lower() in ["jpg","tif","png"]]
            img_paths.extend(pimages)
        else:
            img_paths.append(i)
    max_length = opt.max_width
    EOS_TOKEN = 1
    hidden = opt.nh
    width = opt.imgW
    height = opt.imgH
    nclass = len(alphabet) + 3

    encoder = crnn.CNN(height, 1, opt.nh)
    # decoder = crnn.decoder(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
    decoder = crnn.decoderV2(hidden, nclass)
    if opt.verbose:
        print(encoder)
        model_summary(encoder.cnn)
        print(decoder)
        model_summary(decoder)
    if encoder_path and decoder_path:
        if opt.verbose:
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

    if opt.logfile:
        with open(opt.logfile,"w") as p:
            pass

    for img_path in img_paths:
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
            import numpy
            #with open("log","a") as p:
            #    p.write(str([int(i) for i in torch.log(decoder_attention.data).squeeze()])+"\n")
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
        if opt.verbose:
            print(os.path.basename(img_path),' result: %-20s   => prob:%-3s' % (words, prob))
        if opt.logfile:
            with open(opt.logfile,"a") as p:
                p.write("\t".join([img_path,words,str(prob)+"\n"]))