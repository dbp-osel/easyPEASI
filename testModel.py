#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""easyPEASI
Software to generate results in submission to ACM KDD'20 conference (tentative citation pending review):

David O. Nahmias and Kimberly L. Kontson. 2020. Easy Perturbation EEG Algorithm for Spectral Importance (easyPEASI):
A simple method to identifyimportant spectral features of EEG in deep learning models. 
In Proceedings of The 26th ACM SIGKDD Conference on Knowledge Discovery and DataMining, August 22–27, 2020,
San Diego, CA, USA. (KDD ’20).ACM, New York,NY, USA, 9 pages

If you have found this software useful please consider citing our publication.

Public domain license
"""

""" Disclaimer:
This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees
of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code,
this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge,
to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives,
and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other
parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied,
about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA
or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that
any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.
"""

__author__ = 'David Nahmias'
__copyright__ = 'No copyright - US Government, 2020 , easyPEASI'
__credits__ = ['David Nahmias']
__license__ = 'Public domain'
__version__ = '0.0.1'
__maintainer__ = 'David Nahmias'
__email__ = 'david.nahmias@fda.hhs.gov'
__status__ = 'alpha'

import logging
import time
from copy import copy
import sys

from scipy import stats
from collections import Counter
import random
import numpy as np
from numpy.random import RandomState
import resampy
from torch import optim
import torch.nn.functional as F
import torch as th
from torch.nn.functional import elu
from torch import nn

from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.util import np_to_var
from braindecode.torch_ext.util import set_random_seeds
from braindecode.torch_ext.modules import Expression
from braindecode.experiments.experiment import Experiment
from braindecode.datautil.iterators import CropsFromTrialsIterator
from braindecode.experiments.monitors import (RuntimeMonitor, LossMonitor,
                                              MisclassMonitor)
from braindecode.experiments.stopcriteria import MaxEpochs
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from braindecode.datautil.iterators import get_balanced_batches
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import var_to_np
from braindecode.torch_ext.functions import identity

from dataset import DiagnosisSet
from collections import OrderedDict
from monitors import compute_preds_per_trial, CroppedDiagnosisMonitor

log = logging.getLogger(__name__)
log.setLevel('DEBUG')
import os
import pdb
from loadNEDC import loadNEDCdata,loadSubNormData,addDataNoise,loadNEDCtestData

from auto_diagnosis import create_set,TrainValidTestSplitter,TrainValidSplitter,run_exp

import config
from itertools import combinations

import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


cudnn.benchmark = True

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom', fontsize=16)
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

def autolabel(rects,labels,err,ax):
    i = 0
    for rect in rects:
        h = rect.get_height()
        #ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
        #        ha='center', va='bottom')
        s = labels[i]
        if i == 0:
            coeff = 1.0
        else:
            coeff = 0.99
        ax.text(rect.get_x()+rect.get_width()/2., 1.0*(h+err[i]), '%s'%s,
                ha='center', va='bottom',fontsize=20)
        i += 1

def plotBarsDCNN(mode,acc,res,sig,ax):
    accN = np.mean(np.mean(res,axis=2),axis=0)
    errN = np.std(np.mean(res,axis=2),axis=0)
    labels = ['Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-12Hz)', 'Mu (12-16Hz)', 'Beta (16-25Hz)', 'Gamma (25-40Hz)']
    bands = np.array([[i,i+1] for i in range(1,6)])
    #labels = [str(b[0]) for b in bands]
    #accA = (60.39, 68.75, 68.85)
    #errA = (5.03, 6.29, 6.87)
    #accN = (59.12, 64.12, 64.71)
    #errN = (7.49, 8.19, 6.31)
    bandNum = len(labels)
    ind = np.arange(bandNum)
    width= 1.


    lineX = np.linspace(0-width,bandNum+width,bandNum)
    meanAcc = np.mean(np.mean(acc,axis=1),axis=0)
    meanAccArray = np.array([meanAcc for i in range(len(ind))])
    stdAcc = np.std(np.mean(acc,axis=1),axis=0)
    errorBarsInd = np.linspace(0-width/2,bandNum+width/2,bandNum)
    #errorBarsInd[0] = 0
    #errorBarsInd[-1] = ind[-1]+width
    #ax.plot(lineX,meanAccArray,'g_',markersize=350,markeredgewidth=3)
    #ax.fill_between(lineX, meanAccArray-stdAcc, meanAccArray+stdAcc,
    #    alpha=0.5,edgecolor='#3F7F4C', facecolor='#7EFF99',linewidth=0)
    ax.plot(lineX,meanAccArray,color='#808080',marker='_',markersize=350,markeredgewidth=3)

    ax.fill_between(lineX, meanAccArray-stdAcc, meanAccArray+stdAcc,
        alpha=0.5,edgecolor='#C0C0C0', facecolor='#C0C0C0',linewidth=0)

    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2,'barsabove':True}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    barsN = ax.bar(ind, accN, **bar_kwargsN)
    errsN = ax.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    #bar_kwargsA = {'width':width,'color':'g','linewidth':2,'zorder':5,'align':'center','fill':False,'hatch':'\\\\'}
    #err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #barsA = plt.bar(ind+width, accA, **bar_kwargsA)
    #errsA = plt.errorbar(ind+width, accA, yerr=errA, **err_kwargsA)
    white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Peturbed accuracy')
    green_bar = mlines.Line2D([], [], color='green',lw=3,
                  markersize=15, label='Training accuracy')
    ax.legend(handles=[green_bar,white_patch],fontsize=18,loc='lower right')

    #ax.legend( (barsN[0]), ('Delta'),loc='upper right',fontsize=14)

    #barplot_annotate_brackets(1, 2, 'n.s. - p = .976', ind+width, accA, dh=.1)
    #barplot_annotate_brackets(0, 1, '** - p = .006', ind+width, accA, dh=.15)
    #barplot_annotate_brackets(0, 2, '** - p = .008', ind+width, accA, dh=.195)
    sigLabels = []
    for f in range(bandNum):
        if sig[f] < 0.001:
            sigLabels.append('**\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
        elif sig[f] < 0.01:
            sigLabels.append('*\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
        #elif sig[f] < 0.05:
        #    sigLabels.append('*\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
        else:
            sigLabels.append('n.s.\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))

    autolabel(barsN,sigLabels,errN,ax)
    #autolabel(barsA,['*','***','***'],errA,ax)




    ax.set_ylim(ymin=50,ymax=100)
    ax.set_xlim(xmin=-0.5,xmax=5.5)

    if mode[2] == 0:
        normStatus = 'normal'
    if mode[2] == 1:
        normStatus = 'abnormal'
    labels.insert(0,'')
    #plt.xticks(ind+width/2, labels, color='k',fontsize=18)#,rotation=45)
    #plt.xticks(ind, labels, color='k',fontsize=18)#,rotation=45)
    ax.set_xticklabels(labels,color='k',fontsize=12)
    ax.tick_params(axis='y',labelsize=20)
    #ax.set_yticklabels(fontsize=20)
    ax.set_ylabel('Mean percent accuracy',fontsize=24)
    ax.set_xlabel('Frequency bands',fontsize=24)

    if mode[0] == 'none':
        mode[0] = 'No medications'
        ax.legend(handles=[green_bar,white_patch],fontsize=16,loc='lower right')

    elif mode[1] == 'none':
        mode[1] = 'No medications'
        ax.legend(handles=[green_bar,white_patch],fontsize=16,loc='lower right')


    mode[0] = mode[0].capitalize()
    mode[1] = mode[1].capitalize()

    ax.set_title('%s vs. %s for subjects with %s EEG'%(mode[1],mode[0],normStatus),fontsize=24)


    #plt.show()


def plotTracesDCNN(mode,acc,res,sig,ax):
    accN = np.mean(np.mean(res,axis=2),axis=0)
    errN = np.std(np.mean(res,axis=2),axis=0)
    accN = np.append(accN,accN[-1])
    errN = np.append(errN,errN[-1])
    labels = ['Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-12Hz)', 'Mu (12-16Hz)', 'Beta (16-25Hz)', 'Gamma (25-40Hz)']
    bands = np.array([[i,i+1] for i in range(1,40)])
    labels = [str(b[0]) for b in bands]
    #labels.append(str(bands[-1][1]))
    #accA = (60.39, 68.75, 68.85)
    #errA = (5.03, 6.29, 6.87)
    #accN = (59.12, 64.12, 64.71)
    #errN = (7.49, 8.19, 6.31)
    bandNum = len(bands)+1
    ind = np.arange(bandNum)
    width= 1.

    lineX = np.linspace(0-width,bandNum+width,bandNum)
    meanAcc = np.mean(np.mean(acc,axis=1),axis=0)
    meanAccArray = np.array([meanAcc for i in range(len(ind))])
    stdAcc = np.std(np.mean(acc,axis=1),axis=0)

    ax.plot(lineX,meanAccArray,'g-',markersize=350,markeredgewidth=3)
    ax.fill_between(lineX, meanAccArray-stdAcc, meanAccArray+stdAcc,
        alpha=0.5,edgecolor='#3F7F4C', facecolor='#7EFF99',linewidth=0)
    #pl.bar(range(1,7),[accN[0],accA[0],accN[1],accA[1],accN[2],accA[2]],yerr=[errN[0],errA[0],errN[1],errA[1],errN[2],errA[2]])
    # Pull the formatting out here
    ax.plot(lineX,accN,'b-',markeredgewidth=3)
    ax.fill_between(lineX,accN-errN,accN+errN,alpha=0.5,edgecolor='#1B2ACC', facecolor='#089FFF',linewidth=0)

    #bar_kwargsN = {'width':width,'color':'b','linewidth':2,'zorder':5,'align':'center','fill':False}
    #err_kwargsN = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #barsN = plt.bar(ind, accN, **bar_kwargsN)
    #errsN = plt.errorbar(ind, accN, yerr=errN, **err_kwargsN)

    #bar_kwargsA = {'width':width,'color':'g','linewidth':2,'zorder':5,'align':'center','fill':False,'hatch':'\\\\'}
    #err_kwargsA = {'zorder':0,'fmt':'none','elinewidth':2,'ecolor':'k','capsize':15,'capthick':2,'markeredgewidth':2}  #for matplotlib >= v1.4 use 'fmt':'none' instead
    #barsA = plt.bar(ind+width, accA, **bar_kwargsA)
    #errsA = plt.errorbar(ind+width, accA, yerr=errA, **err_kwargsA)
    #white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Peturbed input training accuracy')
    
    blue_bar = mlines.Line2D([], [], color='blue',lw=3,
                  markersize=15, label='Peturbed training accuracy')
    green_bar = mlines.Line2D([], [], color='green',lw=3,
                  markersize=15, label='True training accuracy')
    ax.legend(handles=[green_bar,blue_bar],fontsize=14,loc='upper left')

    #ax.legend( (barsN[0]), ('Delta'),loc='upper right',fontsize=14)

    #barplot_annotate_brackets(1, 2, 'n.s. - p = .976', ind+width, accA, dh=.1)
    #barplot_annotate_brackets(0, 1, '** - p = .006', ind+width, accA, dh=.15)
    #barplot_annotate_brackets(0, 2, '** - p = .008', ind+width, accA, dh=.195)
    #sigLabels = []
    #for f in range(bandNum):
    #    if sig[f] < 0.001:
    #        sigLabels.append('***\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
    #    elif sig[f] < 0.01:
    #        sigLabels.append('**\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
    #    elif sig[f] < 0.05:
    #        sigLabels.append('*\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))
    #    else:
    #        sigLabels.append('n.s.\n%s=%.02f'%(u'Δ',meanAcc-accN[f]))

    #autolabel(barsN,sigLabels,errN,ax)
    #autolabel(barsA,['*','***','***'],errA,ax)




    ax.set_ylim(ymin=50,ymax=100)
    ax.set_xlim(xmin=lineX[0],xmax=lineX[-1])

    if mode[2] == 0:
        normStatus = 'normal'
    if mode[2] == 1:
        normStatus = 'abnormal'

    #plt.xticks(ind+width/2, labels, color='k',fontsize=18)#,rotation=45)
    labels.append(str(bands[-1][1]))
    labels.insert(0,'')
    #ind = np.append(ind, np.arange(bandNum+1)[-1])
    #lineX = np.append(lineX,lineX[-1]+width)
    #plt.xticks(lineX, labels, color='k',fontsize=18)#,rotation=45)
    ax.set_xticklabels(labels)
    #plt.yticks(fontsize=18)
    ax.set_ylabel('Mean train percent',fontsize=16)
    ax.set_xlabel('Frequency',fontsize=16)

    ax.set_title('%s v. %s for %s EEG'%(mode[1],mode[0],normStatus),fontsize=24)



    #plt.show()


def splitDataRandom_Loaded(allData,allLabels,mode):
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    labels0 = allLabels[allLabels == 0]
    labels1 = allLabels[allLabels == 1]
    data0 = allData[allLabels == 0]
    data1 = allData[allLabels == 1]


    #fullRange = list(range(numberEqSamples))
    #random.shuffle(fullRange)

    #np.save('dataRangeOrder%s'%CLASSY,fullRange)

    fullRange = np.load('dataRangeOrder%s%s.npy'%(mode[:4],mode[4]))

    testIndecies = fullRange[trainSamplesNum:]
    trainIndecies = fullRange[:trainSamplesNum]

    allDataTrain = np.concatenate((data0[trainIndecies],data1[trainIndecies]),axis=0)
    allLabelsTrain = np.concatenate((labels0[trainIndecies],labels1[trainIndecies]),axis=0)

    allDataTest = np.concatenate((data0[testIndecies],data1[testIndecies]),axis=0)
    allLabelsTest = np.concatenate((labels0[testIndecies],labels1[testIndecies]),axis=0)

    return allDataTrain,allLabelsTrain,allDataTest,allLabelsTest

def runModel(data,labels,mode):

    n_classes = 2
    model = Deep4Net(config.n_chans, n_classes,
                             n_filters_time=config.n_start_chans,
                             n_filters_spat=config.n_start_chans,
                             input_time_length=config.input_time_length,
                             n_filters_2 = int(config.n_start_chans * config.n_chan_factor),
                             n_filters_3 = int(config.n_start_chans * (config.n_chan_factor ** 2.0)),
                             n_filters_4 = int(config.n_start_chans * (config.n_chan_factor ** 3.0)),
                             final_conv_length=config.final_conv_length,
                            stride_before_pool=True).create_network()

    to_dense_prediction_model(model)

    if config.cuda:
    	model.cuda()
    test_input = np_to_var(np.ones((2, config.n_chans, config.input_time_length, 1), dtype=np.float32))
    if config.cuda:
        test_input = test_input.cuda()

    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    iterator = CropsFromTrialsIterator(batch_size=config.batch_size,
                                           input_time_length=config.input_time_length,
                                           n_preds_per_input=n_preds_per_input)

    #model.add_module('softmax', nn.LogSoftmax(dim=1))

    model.eval()

    mode[2] = str(mode[2])
    mode[3] = str(mode[3])
    modelName = '-'.join(mode[:4])

    #params = th.load(mode+'Model.pt')

    params = th.load('%sModel%s.pt'%(modelName,mode[4]))

    model.load_state_dict(params)
    
    test_set = SignalAndTarget(data, labels)

    setname = 'test'
    #print("Compute predictions for {:s}...".format(setname))
    dataset = test_set
    if config.cuda:
        preds_per_batch = [var_to_np(model(np_to_var(b[0]).cuda()))
                  for b in iterator.get_batches(dataset, shuffle=False)]
    else:
        preds_per_batch = [var_to_np(model(np_to_var(b[0])))
                  for b in iterator.get_batches(dataset, shuffle=False)]
    preds_per_trial = compute_preds_per_trial(
        preds_per_batch, dataset,
        input_time_length=iterator.input_time_length,
        n_stride=iterator.n_preds_per_input)
    mean_preds_per_trial = [np.mean(preds, axis=(0, 2)) for preds in
                                preds_per_trial]
    mean_preds_per_trial = np.array(mean_preds_per_trial)

    all_pred_labels = np.argmax(mean_preds_per_trial, axis=1).squeeze()
    all_target_labels = dataset.y
    acc_per_class = []
    for i_class in range(n_classes):
        mask = all_target_labels == i_class
        acc = np.mean(all_pred_labels[mask] ==
                      all_target_labels[mask])
        acc_per_class.append(acc)
    misclass = 1 - np.mean(acc_per_class)
    print('Accuracy:{}'.format(1-misclass))
    print('Class 0:{}, Class 1:{}'.format(acc_per_class[0],acc_per_class[1]))
    return np.array([acc_per_class[0],acc_per_class[1]])


def testEachSingle(mode='subNorm',tests=['bands']):
    bands=[1,4,8,12,16,25,40]
    bands = range(1,41)
    channels = range(19)

    bandResults = np.zeros((len(bands)-1,2))
    channelResults = np.zeros((len(channels),2))
    bandChannelResults = np.zeros((len(bands)-1,len(channels),2))

    print('\n',mode,'Classifications')
    '''
    if dataset == 'train':
        if mode == 'subNorm':
            test_X,test_y,X,y = loadSubNormData(mode='all')
        else:
            test_X,test_y,X,y = loadNEDCdata(mode='all',classy=mode)#loadNEDCtestData(classy=mode)
    elif dataset == 'test':
        if mode == 'subNorm':
            X,y,test_X,test_y = loadSubNormData(mode='all')
        else:
            test_X,test_y = loadNEDCtestData(classy=mode)
    '''
    data = np.load('data%s.npy'%mode[:3])
    labels = np.load('labels%s.npy'%mode[:3])

    X,y,test_X,test_y = splitDataRandom_Loaded(data,labels,mode)
    test_X,test_y,X,y = X,y,test_X,test_y

    #print('trainSize:',test_X.shape, 'testSize:',X.shape)
    #return 0,0
    print('True Test:')
    accT=runModel(test_X,test_y,mode)
    
    modeName = '-'.join([mode[0],mode[1],str(mode[2]),str(mode[3]),mode[4]])  

    if 'bands' in tests:
        for b in range(len(bands)-1):
            print('Test with band:',bands[b],'-',bands[b+1])
            test_X_noise = addDataNoise(test_X,band=[bands[b],bands[b+1]])
            acc = runModel(test_X_noise,test_y,mode)
            bandResults[b,:] = acc
        
        np.save('PeturbResults/'+modeName+'FrequencyResults',bandResults)


    if 'channels' in tests:
        for c in channels:
            print('Test with channel:',c+1)
            test_X_noise = addDataNoise(test_X,channels=[c])
            acc = runModel(test_X_noise,test_y,mode)
            channelResults[c,:] = acc
        
        np.save('PeturbResults/'+modeName+'ChannelResults',channelResults)

    if 'bandChannels' in tests:
        for b in range(len(bands)-1):
            for c in channels:
                print('Bands:',bands[b],'-',bands[b+1],'Test with channel:',c+1)
                test_X_noise = addDataNoise(test_X,band=[bands[b],bands[b+1]],channels=[c])
                acc = runModel(test_X_noise,test_y,mode)
                bandChannelResults[b,c,:] = acc
        
        np.save('PeturbResults/'+modeName+'BandChannelResults',bandChannelResults)

    #np.save('../auto-eeg-diagnosis-example-master/'+dataset+'PeturbResults/'+mode+'BandResults',bandResults)
    #np.save('../auto-eeg-diagnosis-example-master/'+dataset+'PeturbResults/'+mode+'ChannelResults',channelResults)
    #np.save('../auto-eeg-diagnosis-example-master/'+dataset+'PeturbResults/'+mode+'BandChannelResults',bandChannelResults)

    if 'bandChannels' in tests:
        return accT,bandChannelResults
    elif 'channels' in tests:
        return accT,channelResults
    elif 'bands' in tests:
        return accT,bandResults

    return 0

def testAllCombo(mode='subNorm'):
    bands=np.array([[1,4],[4,8],[8,12],[12,16],[16,25],[25,40]])
    featureNames = range(6)

    print('\n',mode,'Classifications')
    '''
    if dataset == 'train':
        if mode == 'subNorm':
            test_X,test_y,X,y = loadSubNormData(mode='all')
        else:
            test_X,test_y,X,y = loadNEDCdata(mode='all',classy=mode)#loadNEDCtestData(classy=mode)
    elif dataset == 'test':
        if mode == 'subNorm':
            X,y,test_X,test_y = loadSubNormData(mode='all')
        else:
            test_X,test_y = loadNEDCtestData(classy=mode)
    '''
    data = np.load('data%s.npy'%mode[:3])
    labels = np.load('labels%s.npy'%mode[:3])

    X,y,test_X,test_y = splitDataRandom_Loaded(data,labels,mode)
    test_X,test_y,X,y = X,y,test_X,test_y
    print('True Test:')
    acc=runModel(test_X,test_y,mode)
    return acc
    featsCombos = [combinations(featureNames, i) for i in range(1,len(featureNames)+1)]
    feats = [list(elems) for comb in featsCombos for elems in comb]

    resultsF = [[[] for x in range(len(featureNames))] for y in range(len(featureNames))]
    resultsFmean = [[[] for x in range(len(featureNames))] for y in range(len(featureNames))]
    featureResultsF = [[[] for x in range(len(featureNames))] for y in range(len(featureNames))]

    for f in feats:

        bandList = bands[f].tolist()
        print('Test with band:',bandList)
        test_X_noise = addDataNoise(test_X,band=bandList)
        acc = runModel(test_X_noise,test_y,mode)

        for n in range(len(featureNames)):
            if featureNames[n] in f:
                resultsF[n][len(f)-1].append(acc)
                resultsFmean[n][len(f)-1].append(np.mean(acc))
                featureResultsF[n][len(f)-1].append(f)

    modeName = '-'.join([mode[0],mode[1],str(mode[2]),str(mode[3]),mode[4]])  
            
    np.save('PeturbResults/'+modeName+'BandComboResults',resultsF)
    np.save('PeturbResults/'+modeName+'BandComboMeanResults',resultsFmean)
    np.save('PeturbResults/'+modeName+'BandComboOrder',featureResultsF)

    return 0


def displayrRes(mode,bins,ax):
    #data = np.load('data%s.npy'%mode[:3])
    #labels = np.load('labels%s.npy'%mode[:3])
    if bins == 6:
        bands = np.array([[1,4],[4,8],[8,12],[12,16],[16,25],[25,40]])
        bandLoadLabel = 'Band'
    elif bins == 40:
        bands = np.array([[i,i+1] for i in range(1,40)])
        bandLoadLabel = 'Frequency'
    
    numBands = len(bands)

    curMode = [mode[0],mode[1],mode[2],'']
    modeName = '-'.join([curMode[0],curMode[1],str(curMode[2]),curMode[3]])  

    allAcc = np.load('PeturbResults/'+modeName+'Results.npy')
    
    #allAcc = np.zeros((10,2))
    allRes = np.zeros((10,numBands,2))
    delta = np.zeros((10,numBands,2))
    deltaStd = np.zeros((10,numBands,2))
    sig = np.zeros((numBands,1))


    for r in range(1,11):
        curMode = [mode[0],mode[1],mode[2],r,'']
        #X,y,test_X,test_y = splitDataRandom_Loaded(data,labels,curMode)
        #test_X,test_y,X,y = X,y,test_X,test_y
        #print('True Test:')
        #acc=runModel(test_X,test_y,curMode)
        #allAcc[r-1,:] = acc

        modeName = '-'.join([curMode[0],curMode[1],str(curMode[2]),str(curMode[3]),curMode[4]])  
        res = np.load('PeturbResults/'+modeName+bandLoadLabel+'Results.npy')
        allRes[r-1,:,:] = res
        delta[r-1,:,:] = np.subtract(allAcc[r-1,:],res)
    for f in range(numBands):
        F,sig[f] = stats.wilcoxon(np.mean(allAcc,axis=1),np.mean(allRes,axis=2)[:,f]) 
        F,sig[f] = stats.f_oneway(np.mean(allAcc,axis=1),np.mean(allRes,axis=2)[:,f]) 
        F,sig[f] = stats.kruskal(np.mean(allAcc,axis=1),np.mean(allRes,axis=2)[:,f]) 

    print('%s - Acc: %f (%f)'%(curMode,np.mean(np.mean(allAcc,axis=1),axis=0),np.std(np.mean(allAcc,axis=1),axis=0)))


    for f in range(numBands):
        print('Acc %s Band: %d-%d Hz: %f (%f)'%(curMode,bands[f,0],bands[f,1],np.mean(np.mean(allRes,axis=2)[:,f]),np.std(np.mean(allRes,axis=2)[:,f])))
        print('Acc Diff %s Band: %d-%d Hz: %f (%f)'%(curMode,bands[f,0],bands[f,1],np.mean(np.mean(delta,axis=2)[:,f]),np.std(np.mean(delta,axis=2)[:,f])))
        print('Significance Results %s Band: %d-%d Hz- p =%f'%(curMode,bands[f,0],bands[f,1],sig[f]))


    #pdb.set_trace()
    #modeName = '-'.join([curMode[0],curMode[1],str(curMode[2]),curMode[4]])  
    #np.save('PeturbResults/'+modeName+'Results',allAcc)
    if bandLoadLabel == 'Band':
        plotBarsDCNN(mode,100.*allAcc,100.*allRes,sig,ax)
    elif bandLoadLabel == 'Frequency':
        plotTracesDCNN(mode,100.*allAcc,100.*allRes,sig,ax)
    #pdb.set_trace()



if __name__ == '__main__':
    #mode = str(sys.argv[1])
    startGlobal = time.time()

    runAll = False
    runTop = False
    
    plot1 = False
    plot2 = False
    plot3 = True
    
    binNum = 6
    if binNum == 6:
        bands = np.array([[1,4],[4,8],[8,12],[12,16],[16,25],[25,40]])
    elif binNum == 40:
        bands = np.array([[i,i+1] for i in range(1,40)])

    if runAll == True:
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','keppra',0],['none','dilantin',1],['none','keppra',1]]
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]

        allResults = []
        allResultsPerturb = []

        for mode in classes:
            modeResult = []
            modeResultPerturb = []
            modeResultRand = []
            for r in range(1,11):
                curMode = [mode[0],mode[1],mode[2],r,'']
                #print(curMode)
                #curResult = testAllCombo(curMode)
                curResult,curResultPerturb = testEachSingle(curMode,['bands'])
                modeResult.append(curResult)
                modeResultPerturb.append(curResultPerturb)

                curModeRand = [mode[0],mode[1],mode[2],r,'-shuffle']
                #print(curMode)
                curResultRand = testAllCombo(curModeRand)
                #curResultRand = testEachSingle(curModeRand)
                modeResultRand.append(curResultRand)

            allResults.append(modeResult)
            allResultsPerturb.append(modeResultPerturb)

            modeResult=np.mean(modeResult,axis=1)
            modeResultPerturb=np.mean(np.array(modeResultPerturb),axis=2)
            modeResultRand=np.mean(modeResultRand,axis=1)
            #np.mean(np.mean(modeResult,axis=0),axis=1)
            print('%s - Acc: %f (%f)'%(curMode,np.mean(modeResult),np.std(modeResult)))
            print('%s - Acc: %f (%f)'%(curModeRand,np.mean(modeResultRand),np.std(modeResultRand)))
            for f in range(modeResultPerturb.shape[-1]):
                print('Acc %s Band: %d-%d Hz: %f (%f)'%(curMode,bands[f,0],bands[f,1],np.mean(modeResultPerturb[:,f]),np.std(modeResultPerturb[:,f])))
                print('Acc Diff %s Band: %d-%d Hz: %f (%f)'%(curMode,bands[f,0],bands[f,1],np.mean(np.subtract(modeResult,modeResultPerturb[:,f])),np.std(np.subtract(modeResult,modeResultPerturb[:,f]))))

                F,p = stats.kruskal(modeResultPerturb[:,f],modeResult) 
                print('True Results %s Band: %d-%d Hz- p =%f'%(curMode,bands[f,0],bands[f,1],p))

                F,p = stats.kruskal(modeResultPerturb[:,f],modeResultRand) 
                print('Rand Results %s Band: %d-%d Hz - p =%f'%(curMode,bands[f,0],bands[f,1],p))


        toCompare = [[0,2],[0,3],[1,4],[1,5],[2,3],[4,5]]
        toCompare = [[0,2],[0,4],[1,3],[1,5],[2,4],[3,5]]

        allResultsPerturb = np.mean(np.array(allResultsPerturb),axis=-1)
        for c in toCompare:
            for f in range(modeResultPerturb.shape[-1]):
                F,p = stats.kruskal(allResultsPerturb[c[0],:,f],allResultsPerturb[c[1],:,f]) 

                print('Significance band %d-%d Hz: %s v. %s - p =%f'%(bands[f,0],bands[f,1],classes[c[0]],classes[c[1]],p))

    if runTop == True:
        classes = [['dilantin','keppra',0,3],['dilantin','keppra',1,4],['none','dilantin',0,3],['none','keppra',0,6],['none','dilantin',1,4],['none','keppra',1,2]]
        classes = [['dilantin','keppra',0,3],['dilantin','keppra',1,4],['none','dilantin',0,3],['none','dilantin',1,4],['none','keppra',0,6],['none','keppra',1,2]]

        allResults = []

        for mode in classes:
            modeResult = []
            modeResultRand = []
            curMode = [mode[0],mode[1],mode[2],mode[3],'']
            #print(curMode)
            #curResult = testAllCombo(curMode)
            curResult,curResultPerturb = testEachSingle(curMode,['bandChannels'])

            modeResult.append(curResultPerturb)

            allResults.append(modeResult)

            #F,p = stats.f_oneway(modeResult,modeResultRand) 
            print('%s - Acc: %f (%f)'%(curMode,np.mean(modeResult),np.std(modeResult)))
            #print('%s - Acc: %f (%f)'%(curModeRand,np.mean(modeResultRand),np.std(modeResultRand)))

            #print('%s - p =%f'%(curMode,p))


    if plot1 == True:
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]
        figMedMed,axMedMed = plt.subplots(2,1)
        figMedNone,axMedNone = plt.subplots(2,2)
        numMode = 0
        for mode in classes:
            if numMode < 2:
                displayrRes(mode,binNum,axMedMed[numMode])
                if numMode == 0:
                    axMedMed[numMode].xaxis.set_ticklabels([])
                    axMedMed[numMode].xaxis.set_label_text("")
                figMedMed.suptitle("Frequency band importances identifying anti-convulsants", fontsize=24)
            else:
                div,rem = np.divmod(numMode-2,2)
                displayrRes(mode,binNum,axMedNone[div,rem])
                
                if div == 0 and rem == 0:
                    axMedNone[div,rem].xaxis.set_ticklabels([])
                    axMedNone[div,rem].xaxis.set_label_text("")
                elif div == 0 and rem == 1:
                    axMedNone[div,rem].xaxis.set_ticklabels([])
                    axMedNone[div,rem].xaxis.set_label_text("")
                    axMedNone[div,rem].yaxis.set_ticklabels([])
                    axMedNone[div,rem].yaxis.set_label_text("")
                elif div == 1 and rem == 1:
                    axMedNone[div,rem].yaxis.set_ticklabels([])
                    axMedNone[div,rem].yaxis.set_label_text("")
                
                figMedNone.suptitle("Different EEG clinical types", fontsize=30)
                figMedNone.text(x=.001, y=.70, s="Different anti-convulsants", fontsize=30, rotation='90')
            numMode += 1
        
        figMedMed.subplots_adjust(left=0.06,bottom=0.07,right=0.99,top=0.90,wspace=0.04,hspace=0.2)
        figMedNone.subplots_adjust(left=0.08,bottom=0.07,right=0.99,top=0.90,wspace=0.04,hspace=0.15)


    if plot2 == True:
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]
        figN = plt.figure()
        figA = plt.figure()
        grid = plt.GridSpec(2,2)
        numMode = 0
        for mode in classes:
            div,rem = np.divmod(numMode-2,2)

            if mode[2] == 0:
                if numMode < 2:
                    ax = figN.add_subplot(grid[0,:])
                    ax.xaxis.set_label_text("")

                else:
                    ax = figN.add_subplot(grid[1,div])
                    if div == 1:
                        ax.yaxis.set_ticklabels([])
                        ax.yaxis.set_label_text("")
            else:
                if numMode < 2:
                    ax = figA.add_subplot(grid[0,:])
                    ax.xaxis.set_label_text("")

                else:
                    ax = figA.add_subplot(grid[1,div])
                    if div == 1:
                        ax.yaxis.set_ticklabels([])
                        ax.yaxis.set_label_text("")

            displayrRes(mode,binNum,ax)
            if mode[2] == 0:
                if numMode < 2:
                    ax.xaxis.set_label_text("")
                else:
                    if div == 1:
                        ax.yaxis.set_ticklabels([])
                        ax.yaxis.set_label_text("")
            else:
                if numMode < 2:
                    ax.xaxis.set_label_text("")

                else:
                    if div == 1:
                        ax.yaxis.set_ticklabels([])
                        ax.yaxis.set_label_text("")
            white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Peturbed accuracy')
            green_bar = mlines.Line2D([], [], color='green',lw=3,
                  markersize=15, label='Training accuracy')
            ax.legend(handles=[green_bar,white_patch],fontsize=16,loc='lower right')
            ax.set_title('%s vs. %s'%(mode[1],mode[0]),fontsize=24)


            numMode += 1
    
            
        figN.suptitle("Frequency band importances for subjects with normal EEGs", fontsize=24)
        figA.suptitle("Frequency band importances for subjects with abnormal EEGs", fontsize=24)

        #figMedNone.suptitle("Different EEG clinical types", fontsize=30)
        #figMedNone.text(x=.001, y=.70, s="Different anti-convulsants", fontsize=30, rotation='90')
        
        figN.subplots_adjust(left=0.07,bottom=0.07,right=0.99,top=0.91,wspace=0.04,hspace=0.2)
        figA.subplots_adjust(left=0.07,bottom=0.07,right=0.99,top=0.91,wspace=0.04,hspace=0.2)


    if plot3 == True:
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','dilantin',1],['none','keppra',0],['none','keppra',1]]
        classes = [['dilantin','keppra',0],['dilantin','keppra',1],['none','dilantin',0],['none','keppra',0],['none','dilantin',1],['none','keppra',1]]
        
        white_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Peturbed accuracy')    
        green_bar = mlines.Line2D([], [], color='gray',lw=3,
                  markersize=15, label='Training accuracy')
        labels = ['Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-12Hz)', 'Mu (12-16Hz)', 'Beta (16-25Hz)', 'Gamma (25-40Hz)']
        labels.insert(0,'')
        
        figMed,axMed = plt.subplots(2,1)
        figNorm,axNorm = plt.subplots(2,1)
        figAbnorm,axAbnorm = plt.subplots(2,1)
        numMode = 0
        for mode in classes:
            div,rem = np.divmod(numMode,2)

            if numMode < 2:
                displayrRes(mode,binNum,axMed[rem])
                axMed[rem].legend(handles=[green_bar,white_patch],fontsize=20,loc='lower right')
                axMed[rem].set_xticklabels(labels,color='k',fontsize=20)
                axMed[rem].tick_params(axis='y',labelsize=20)
                axMed[rem].set_ylabel('Mean percent accuracy',fontsize=24)
                axMed[rem].set_xlabel('Perturbed frequency bands',fontsize=24)
                
                if rem == 0:
                    axMed[rem].xaxis.set_ticklabels([])
                    axMed[rem].xaxis.set_label_text("")
                figMed.suptitle("Frequency band importance identifying anticonvulsants", fontsize=24)
                figMed.text(.02,.92,'a.',fontsize=26)
                figMed.text(.02,.47,'b.',fontsize=26)


            elif numMode < 4:
                displayrRes(mode,binNum,axNorm[rem])
                axNorm[rem].set_title('%s vs. %s'%(mode[1],mode[0]),fontsize=24)
                axNorm[rem].legend(handles=[green_bar,white_patch],fontsize=20,loc='lower right')
                axNorm[rem].set_xticklabels(labels,color='k',fontsize=20)
                axNorm[rem].tick_params(axis='y',labelsize=20)
                axNorm[rem].set_ylabel('Mean percent accuracy',fontsize=24)
                axNorm[rem].set_xlabel('Perturbed frequency bands',fontsize=24)

                if rem == 0:
                    axNorm[rem].xaxis.set_ticklabels([])
                    axNorm[rem].xaxis.set_label_text("")
                figNorm.suptitle("Frequency band importance identifying anticonvulsants from no medications \n for subjects with normal EEG", fontsize=24)                
                figNorm.text(.02,.89,'a.',fontsize=26)
                figNorm.text(.02,.46,'b.',fontsize=26)

            elif numMode < 6:
                displayrRes(mode,binNum,axAbnorm[rem])
                axAbnorm[rem].set_title('%s vs. %s'%(mode[1],mode[0]),fontsize=24)
                axAbnorm[rem].legend(handles=[green_bar,white_patch],fontsize=20,loc='lower right')
                axAbnorm[rem].set_xticklabels(labels,color='k',fontsize=20)
                axAbnorm[rem].tick_params(axis='y',labelsize=20)
                axAbnorm[rem].set_ylabel('Mean percent accuracy',fontsize=24)
                axAbnorm[rem].set_xlabel('Perturbed frequency bands',fontsize=24)

                if rem == 0:
                    axAbnorm[rem].xaxis.set_ticklabels([])
                    axAbnorm[rem].xaxis.set_label_text("")
                figAbnorm.suptitle("Frequency band importance identifying anticonvulsants from no medications \n for subjects with abnormal EEG", fontsize=24)                
                figAbnorm.text(.02,.89,'a.',fontsize=26)
                figAbnorm.text(.02,.46,'b.',fontsize=26)
                
            numMode += 1
        
        figMed.subplots_adjust(left=0.06,bottom=0.08,right=0.99,top=0.90,wspace=0.04,hspace=0.2)
        figNorm.subplots_adjust(left=0.06,bottom=0.08,right=0.99,top=0.87,wspace=0.04,hspace=0.2)
        figAbnorm.subplots_adjust(left=0.06,bottom=0.08,right=0.99,top=0.87,wspace=0.04,hspace=0.2)

    endGlobal = time.time()

    print('Time elapsed for all tests: %s'%(str(endGlobal-startGlobal)))
    
    plt.show()

        