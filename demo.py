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
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help="path to encoder")
    parser.add_argument('--decoder', type=str, help='path to decoder (optional)')
    parser.add_argument('--params', type=str, help='path to params.json')    
    parser.add_argument('--image', nargs='+', type=str, help='path to image')
    parser.add_argument('--out', type=str, help='folder for outputs')

    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=800, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=32, help='size of the lstm hidden state')
    parser.add_argument('--cnn_size', type=int, default=16, help='CNN kernel size')
    parser.add_argument('--max_width', type=int, default=500, help='the width of the featuremap out from cnn')
    parser.add_argument('--alphabet', type=str, default="NumAlphaSpace", help='Vocabulary type: Num, NumAlpha...')

    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=False)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--verbose', default=False, action='store_true', help='Verbose mode')
    parser.add_argument('--logfile', help='Write results in csv file')
    opt = parser.parse_args()

    use_gpu = opt.cuda

    encoder_path = opt.encoder
    decoder_path = opt.decoder
    outputdir=None
    if opt.out:
        print(" -- Writing attention!")
        outputdir=opt.out
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)

    if not encoder_path or not os.path.exists(encoder_path):
        print(" -- Missing encoder_path")
        sys.exit(0)
    elif not opt.decoder:
        print(" -- Missing encoder_path... guessing name")
        decoder_path = os.path.dirname(encoder_path)+os.sep+os.path.basename(encoder_path).replace("encoder_","decoder_")
    elif not os.path.exists(opt.decoder):
        print(" -- Error in decoder_path")
        sys.exit(0)
    
    if opt.params:
        print(" -- Using params.json to fill the parameters")
        import json
        params={}
        with open(opt.params) as p:
            params = json.loads(p.read())
        opt.alphabet = params["alphabet"]
        opt.nh = params["nh"]
        width = params["imgW"]
        height = params["imgH"]
        opt.cnn_size = params["cnn_size"]
        max_length = params["max_width"]
    else:
        max_length = opt.max_width
        width = opt.imgW
        height = opt.imgH

    img_paths = []
    for i in opt.image:
        if os.path.isdir(i):
            pimages = [i+os.sep+x for x in os.listdir(i) if x.split(".")[-1].lower() in ["jpg","tif","png"]]
            img_paths.extend(pimages)
        else:
            img_paths.append(i)


    EOS_TOKEN = 1
    alphabet = src.utils.getAlphabetStr(opt.alphabet)
    nclass = len(alphabet) + 3

    encoder = crnn.CNN(height, 1, opt.nh, cnn_size=opt.cnn_size)#, cnn_size=16)
    # decoder = crnn.decoder(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
    decoder = crnn.decoderV2(opt.nh, nclass)
    if opt.verbose:
        print(encoder)
        model_summary(encoder.cnn)
        print(decoder)
        model_summary(decoder)
    if encoder_path and decoder_path:
        if opt.verbose:
            print('loading pretrained models ......')
            print("   - encoder_path:",encoder_path)
            print("   - decoder_path:",decoder_path)
        try:
            encoder.load_state_dict(torch.load(encoder_path,map_location='cpu' if not use_gpu else None))
        except RuntimeError as e:
            print("** ERROR loading encoder: ",encoder_path)
            print(e)
            sys.exit(1)
        try:
            decoder.load_state_dict(torch.load(decoder_path,map_location='cpu' if not use_gpu else None))
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
    else:
        use_gpu=False
        torch.set_num_threads(2)


    converter = src.utils.strLabelConverterForAttention(alphabet)
    transformer = src.dataset.resizeNormalize((width, height))

    if opt.logfile:
        with open(opt.logfile,"w") as p:
            pass

    for img_path in img_paths:
        img = Image.open(img_path)
        image = img.convert('L')
        image = transformer(image)
        if use_gpu:
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        t0 = time.time()

        encoder.eval()
        decoder.eval()

        decoded_words = []
        prob = 1.0
        decoder_attentions = torch.zeros(max_length, width//4)
        decoder_input = torch.zeros(1).long()
        decoder_hidden = decoder.initHidden(1)
        if use_gpu:
            decoder_input = decoder_input.cuda()
            decoder_hidden = decoder_hidden.cuda()
        loss = 0.0

        encoder_out = encoder(image)

        att_output = None
        with open("log","a") as p:
            p.write(img_path+"\n")

        for di in range(max_length):  # 最大字符串的长度
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_out)
            probs = torch.exp(decoder_output)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            prob *= probs[:, ni]
            if ni == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(converter.decode(ni))
        t1 = time.time()

        words = ''.join(decoded_words)
        prob = prob.item()
        if opt.verbose:
            print(os.path.basename(img_path),' result: %-20s   => prob:%-3s' % (words, prob))
            print(" Time: ",t1-t0)
        if opt.logfile:
            with open(opt.logfile,"a") as p:
                p.write("\t".join([img_path,words,str(prob)+"\n"]))

        if outputdir:
            import numpy as np
            ofile = outputdir + os.sep + os.path.basename(img_path[:-4]+"_weights.jpg")
            a = np.log(np.array(decoder_attentions.data)[:di])
            a = 255 + np.clip(a*10,-255,0)
            aima = Image.fromarray(a).convert("L").resize((width,di*4),Image.NEAREST)

            oimage = Image.new("L",(width,di*4+height))
            oimage.paste(img.resize((width,height)))
            oimage.paste(aima,(0,height))
            oimage.save(ofile)
