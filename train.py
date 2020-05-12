# coding:utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import src.utils as utils
import src.dataset as dataset
import src.h5dataset as h5dataset
import time

import models.crnn_lang as crnn
from models.summary import model_summary


SOS_token = 0
EOS_TOKEN = 1              # 结束标志的标签
BLANK = 2                  # blank for padding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainlist',  default='elys_train_list.txt')
    parser.add_argument('--vallist',  default='elys_test_list.txt')
    parser.add_argument('--outdir', default='./expr/elys2020', help='Where to store samples and models')

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=800, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=32, help='size of the lstm hidden state')
    parser.add_argument('--cnn_size', type=int, default=16, help='CNN kernel size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Decoder Dropout')
    
    parser.add_argument('--max_width', type=int, default=500, help='the width of the featuremap out from cnn')
    parser.add_argument('--alphabet', type=str, default="NumAlphaSpace", help='Vocabulary type: Num, NumAlpha...')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
    parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
    parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
    parser.add_argument('--valInterval', type=int, default=1, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=1, help='Interval to be displayed')
    parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
    parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
    opt = parser.parse_args()

    
    print(crnn.__name__)

    if opt.outdir is None:
        opt.outdir = 'expr'
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    else:
        print(" ** Running experiment folder already exists:",opt.outdir)
        
    import json

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transform = None
    if opt.trainlist.lower().endswith(".h5"):
        train_dataset = h5dataset.H5Dataset(opt.trainlist)
        opt.imgW, h= train_dataset.imaDims()
        opt.imgH = int(16 * round(h/16))
        print(" ** TrainSize:",(opt.imgW,opt.imgH))
        opt.max_width = train_dataset.prof_lenght
    else:
        train_dataset = dataset.listDataset(list_file =opt.trainlist, transform=transform)

    print(" -- Saving parameters json:")
    with open(opt.outdir+os.sep+"params.json","w") as p:
        p.write(json.dumps(vars(opt)))

    assert train_dataset
    if not opt.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=False, sampler=sampler,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

    if opt.vallist.endswith(".h5"):
        #h5file, datasetImage='/train/image', datasetProf='/train/prof'        
        test_dataset = h5dataset.H5Dataset(opt.vallist, 
                                                datasetImage='/test/image', 
                                                datasetProf='/test/prof',
                                                transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
    else:
        test_dataset = dataset.listDataset(list_file =opt.vallist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))

    alphabet = utils.getAlphabetStr(opt.alphabet)

    nclass = len(alphabet) + 3          # decoder的时候，需要的类别数,3 for SOS,EOS和blank 
    print(" -- Number of classes:", nclass)
    nc = 1

    converter = utils.strLabelConverterForAttention(alphabet)
    alphabet = utils.getAlphabetStr(opt.alphabet)

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()              # 最后的输出要为log_softmax


    encoder = crnn.CNN(opt.imgH, nc, opt.nh, cnn_size=opt.cnn_size)
    # decoder = crnn.decoder(opt.nh, nclass, dropout_p=0.1, max_length=opt.max_width)        # max_length:w/4,为encoder特征提取之后宽度方向上的序列长度
    decoder = crnn.decoderV2(opt.nh, nclass, dropout_p=opt.dropout)        # For prediction of an indefinite long sequence
    encoder.apply(utils.weights_init)
    decoder.apply(utils.weights_init)
    # continue training or use the pretrained model to initial the parameters of the encoder and decoder
    if opt.encoder:
        print('loading pretrained encoder model from %s' % opt.encoder)
        encoder.load_state_dict(torch.load(opt.encoder))
    if opt.decoder:
        print('loading pretrained encoder model from %s' % opt.decoder)
        decoder.load_state_dict(torch.load(opt.decoder))
    print(encoder)
    model_summary(encoder)
    print(decoder)
    model_summary(decoder)

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.LongTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        encoder.cuda()
        decoder.cuda()
        # encoder = torch.nn.DataParallel(encoder, device_ids=range(opt.ngpu))
        # decoder = torch.nn.DataParallel(decoder, device_ids=range(opt.ngpu))
        image = image.cuda()
        text = text.cuda()
        criterion = criterion.cuda()

    # loss averager
    loss_avg = utils.averager()
    loss_epoch = utils.averager()

    # setup optimizer
    if opt.adam:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                            betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr)
    else:
        encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
        decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)


    def val(encoder, decoder, criterion, batchsize, dataset, teach_forcing=False, max_iter=100):

        for e, d in zip(encoder.parameters(), decoder.parameters()):
            e.requires_grad = False
            d.requires_grad = False

        encoder.eval()
        decoder.eval()
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=batchsize, num_workers=int(opt.workers))
        val_iter = iter(data_loader)

        n_correct = 0
        n_total = 0
        loss_avg = utils.averager()

        max_iter = min(max_iter, len(data_loader))
        # max_iter = len(data_loader) - 1
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts = data
            b = cpu_images.size(0)
            #utils.loadData(image, cpu_images)
            if opt.cuda:
                image = cpu_images.to('cuda')
            else:
                image = cpu_images            
            target_variable = converter.encode(cpu_texts)
            n_total += len(cpu_texts[0]) + 1                       # 还要准确预测出EOS停止位

            decoded_words = []
            decoded_label = []
            decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)
            encoder_outputs = encoder(image)            # cnn+biLstm做特征提取
            
            if opt.cuda:               # cnn+biLstm做特征提取
                target_variable = target_variable.cuda()
                decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出
                decoder_hidden = decoder.initHidden(b).cuda()
            else:
                target_variable = target_variable
                decoder_input = target_variable[0]      # 初始化decoder的开始,从0开始输出
                decoder_hidden = decoder.initHidden(b)            
            loss = 0.0
            if not teach_forcing:
                # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
                for di in range(1, target_variable.shape[0]):  # 最大字符串的长度
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
                    loss_avg.add(loss)
                    # ATTENTION VALUES
                    #decoder_attention = decoder_attention.squeeze()
                    #decoder_attentions[di-1,:decoder_attention.shape[0]] = decoder_attention.data
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze(1)
                    decoder_input = ni
                    if ni == EOS_TOKEN:
                        decoded_words.append('<EOS>')
                        decoded_label.append(EOS_TOKEN)
                        break
                    else:
                        decoded_words.append(converter.decode(ni))
                        decoded_label.append(ni)

            # 计算正确个数
            for pred, target in zip(decoded_label, target_variable[1:,:]):
                if pred == target:
                    n_correct += 1

            if i % 100 == 0:                 # 每100次输出一次
                texts = cpu_texts[0]
                print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

        accuracy = n_correct / float(n_total)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
        return (loss_avg.val(), accuracy)

    def trainBatch(encoder, decoder, train_iter, criterion, encoder_optimizer, decoder_optimizer, teach_forcing_prob=1):
        '''
            target_label:采用后处理的方式，进行编码和对齐，以便进行batch训练
        '''
        data = train_iter.next()
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        target_variable = converter.encode(cpu_texts)
        
        #utils.loadData(image, cpu_images)
        if opt.cuda:
            image = cpu_images.to('cuda')
        else:
            image = cpu_images

        encoder_outputs = encoder(image)
        if opt.cuda:               # cnn+biLstm做特征提取
            target_variable = target_variable.cuda()
            decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出
            decoder_hidden = decoder.initHidden(b).cuda()
        else:
            target_variable = target_variable
            decoder_input = target_variable[0]      # 初始化decoder的开始,从0开始输出
            decoder_hidden = decoder.initHidden(b)

        loss = 0.0
        teach_forcing = True if random.random() > teach_forcing_prob else False
        if teach_forcing:
            # 教师强制：将目标label作为下一个输入
            for di in range(1, target_variable.shape[0]):           # 最大字符串的长度
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])          # 每次预测一个字符
                decoder_input = target_variable[di]  # Teacher forcing/前一次的输出
        else:
            for di in range(1, target_variable.shape[0]):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
                topv, topi = decoder_output.data.topk(1)
                ni = topi.squeeze()
                decoder_input = ni
        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss

    if opt.outdir:
        # write log with all epochs history
        with open(opt.outdir+os.sep+"log.csv","w") as p:
            p.write("train_loss, test_loss, acc\n")

    t0 = time.time()
    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        loss_epoch.reset()
        loss_avg.reset()
        i = 0
        while i < len(train_loader)-1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            cost = trainBatch(encoder, decoder, train_iter, criterion, encoder_optimizer, 
                              decoder_optimizer, teach_forcing_prob=opt.teaching_forcing_prob)
            loss_avg.add(cost)
            loss_epoch.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opt.niter, i, len(train_loader), loss_avg.val()), end=' ')
                loss_avg.reset()
                t1 = time.time()
                print('time elapsed %d' % (t1-t0))
                t0 = time.time()
        loss,acc = val(encoder, decoder, criterion, 1, dataset=test_dataset, teach_forcing=False)            # batchsize:1
        

        if opt.outdir:
            # write log with all epochs history
            with open(opt.outdir+os.sep+"log.csv","a") as p:
                p.write("%1.5f; %1.5f; %1.5f\n"%(loss_epoch.val(), loss, acc))


        # do checkpointing
        if (epoch+1) % opt.saveInterval == 0:
            print(" -- Saving checkpoint",epoch)
            torch.save(
                encoder.state_dict(), '{0}/encoder_{1}.pth'.format(opt.outdir, epoch))
            torch.save(
                decoder.state_dict(), '{0}/decoder_{1}.pth'.format(opt.outdir, epoch))