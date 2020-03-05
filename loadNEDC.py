import numpy as np
import pickle
from time import time
import pdb
import os
from collections import Counter
import random
import datetime
from scipy import fftpack


def loadSubNormData(mode='all'):
    txtFile = '/home/david/Documents/AbnormalProcessing/normEDFfilesB.txt'
    f = open(txtFile,'rU')
    content = f.readlines()
    allDataTrain = []
    allLabelsTrain = []
    allDataTest = []
    allLabelsTest = []
    #pdb.set_trace()

    numFile = len(content)

    numMissed = 0

    count = 0
    count1 = 0
    count2 = 0

    for s in content:
        listS = s.split('/')
        subjInfo = listS[-1].split('.')[0].split('_')
        subjName = int(subjInfo[0])
        session = int(subjInfo[1].split('s')[1])
        recording = subjInfo[2].split('t')[1]

        recordingInfo = str(subjName)+'_S'+str(session)+'_'+recording
        sessionInfo = str(subjName)+'_S'+str(session)

        sessionDate = listS[-2].split('_')
        dateS = datetime.date(int(sessionDate[1]),int(sessionDate[2]),int(sessionDate[3]))

        labelS = listS[9]
        setS = listS[8]

        try:
            file = '/media/david/Data1/pyEDF/'+recordingInfo+'.npy'
            curLoad = np.load(file.strip(),encoding='bytes')
            curLabel = curLoad[0]
            curData = np.float32(curLoad[1])
            
        except FileNotFoundError:
            numMissed += 1
            print('Not Found:',file.strip())

            continue
            ''' Find same session or same subject
            try:
                curSIndex = subjSessLoad.index(sessionInfo)
            except ValueError:
                try:
                    curSIndex = subjLoad.index(str(subjName))
                except ValueError:
                    numMissed +=1
                    continue
            '''
        print('Loading',file.strip(),count,'of',numFile,'(',count1,',',count2,')')

        if setS == 'train':
            allDataTrain.append(curData)

            if labelS == 'normal':
                allLabelsTrain.append(0)
                count1 += 1
            elif labelS == 'abnormal':
                allLabelsTrain.append(1)
                count2 += 1

        elif setS == 'eval':
            allDataTest.append(curData)

            if labelS == 'normal':
                allLabelsTest.append(0)
                count1 += 1
            elif labelS == 'abnormal':
                allLabelsTest.append(1)
                count2 += 1 

        count += 1

    X,y,X_test,y_test = allDataTrain,np.array(allLabelsTrain),allDataTest,np.array(allLabelsTest)

    print('Train Samples:',len(X),'Test Samples:',len(X_test))

    if mode=='train':
        return X, y
    elif mode=='eval':
        return X_test, y_test
    else:
        return X, y, X_test, y_test

def loadNEDCdata(mode='all',classy='norm'):
    path = '../auto-eeg-diagnosis-example-master/rawNEDCdata5.txt'
    allData = []
    allLabels = []
    f = open(path,'rU')
    content = f.readlines()

    numFile = len(content)
    count = 0
    count1 = 0
    count2 = 0

    for file in content:

        print('Loading '+'{0:05}'.format(count)+' of '+'{0:05}'.format(numFile)+' ('+'{0:04}'.format(count1)+','+'{0:04}'.format(count2)+')',end="\r")
        #print('Loading'+file.strip()+' '+'{0:05}'.format(count)+' of '+'{0:05}'.format(numFile)+' ('+'{0:04}'.format(count1)+','+'{0:04}'.format(count2)+')')#,end="\r")

        curLoad = np.load(file.strip(),encoding='bytes')

        curLabel = curLoad[0]
        curData = np.float32(curLoad[1])

        if classy == 'norm':
            if (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[5] == 1) and (count2 < count1):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy == 'gender':
            if (curLabel[2].decode('utf-8') == 'male'):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female'):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy in ['lowAge','midAge','highAge']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAge':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAge':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAge':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] not in curAge):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'ageDiff':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'genderNorm':
            if (curLabel[2].decode('utf-8') == 'male') and (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female') and (curLabel[5] == 0):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy in ['lowAgeNorm','midAgeNorm','highAgeNorm']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAgeNorm':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAgeNorm':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAgeNorm':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] not in curAge) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'ageDiffNorm':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy in ['dilantin','motrin','ativan','keppra']:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy) and (len(medWords.split()) == 1):
                    allLabels.append(1)
                    allData.append(curData)
                    count1 += 1
                    break
                elif (w.lower() == 'none') and (len(medWords.split()) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    count2 += 1
                    break
        
        elif classy == 'medNorm':
            badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
                'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
                'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
                'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran']
            try:
                medList = [w.decode('utf-8').lower() for w in curLabel[4].split()]
            except UnicodeDecodeError:
                medList = ['']
                
            if (curLabel[5] == 0) and ('none' in medList) and (len(medList) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
            elif (curLabel[5] == 1) and (any(x in max(badMeds,medList,key=len) for x in min(badMeds,medList,key=len))):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1
        
        elif len(classy)==3:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy[0]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                    break
                elif (w.lower() == classy[1]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1
                    break



        count += 1

        #if count1 > (358/2.-1):
        #    break
        #if count1 > (20/2.-1):
        #    break

    np.save('data%s'%classy,np.array(allData))
    np.save('labels%s'%classy,np.array(allLabels))

    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    sampleTrain1 = 0
    sampleTrain2 = 0
    sampleTest1 = 0
    sampleTest2 = 0
    allDataTrain = []
    allLabelsTrain = []
    allDataTest = []
    allLabelsTest = []
    for s in range(len(allLabels)):
        if (allLabels[s] == 0) and (sampleTrain1 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain1 += 1
        elif (allLabels[s] == 0) and (sampleTest1 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest1 += 1
        if (allLabels[s] == 1) and (sampleTrain2 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain2 += 1
        elif (allLabels[s] == 1) and (sampleTest2 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest2 += 1

    '''
    num = range(len(allData))
    if mode=='train':
        trainMask = [i for i in num if i%10!=0 ]

    elif mode=='eval':
        trainMask = [i for i in num if i%10==0 ]
        
    X = [allData[i] for i in trainMask]
    y = [allLabels[i] for i in trainMask]
    '''
    #random.shuffle(allLabelsTrain)
    #random.shuffle(allLabelsTest)
    X,y,X_test,y_test = allDataTrain,np.array(allLabelsTrain),allDataTest,np.array(allLabelsTest)

    print('Train Samples:',len(X),'Test Samples:',len(X_test))

    if mode=='train':
        return X, y
    elif mode=='eval':
        return X_test, y_test
    else:
        return X, y, X_test, y_test

def loadNEDCtestData(classy='norm'):
    path = '../auto-eeg-diagnosis-example-master/'+classy+'TestRawNEDCdata.txt'
    allData = []
    allLabels = []
    f = open(path,'rU')
    content = f.readlines()

    numFile = len(content)
    count = 0
    count1 = 0
    count2 = 0

    for file in content:

        #print('Loading',file.strip(),count,'of',numFile,'(',count1,',',count2,')')
        curLoad = np.load(file.strip(),encoding='bytes')

        curLabel = curLoad[0]
        curData = np.float32(curLoad[1])

        if classy == 'norm':
            if (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[5] == 1) and (count2 < count1):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy == 'gender':
            if (curLabel[2].decode('utf-8') == 'male'):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female'):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy in ['lowAge','midAge','highAge']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAge':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAge':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAge':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] not in curAge):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'ageDiff':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'genderNorm':
            if (curLabel[2].decode('utf-8') == 'male') and (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female') and (curLabel[5] == 0):
                allLabels.append(1)
                allData.append(curData)
                count2 += 1

        elif classy in ['lowAgeNorm','midAgeNorm','highAgeNorm']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAgeNorm':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAgeNorm':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAgeNorm':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] not in curAge) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy == 'ageDiffNorm':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1

        elif classy in ['dilantin','motrin','ativan','keppra']:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy) and (len(medWords.split()) == 1):
                    allLabels.append(1)
                    allData.append(curData)
                    count1 += 1
                    break
                elif (w.lower() == 'none') and (len(medWords.split()) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    count2 += 1
                    break
        
        elif classy == 'medNorm':
            badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
                'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
                'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
                'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran']
            try:
                medList = [w.decode('utf-8').lower() for w in curLabel[4].split()]
            except UnicodeDecodeError:
                medList = ['']
                
            if (curLabel[5] == 0) and ('none' in medList) and (len(medList) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
            elif (curLabel[5] == 1) and (any(x in max(badMeds,medList,key=len) for x in min(badMeds,medList,key=len))):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1


        elif len(classy)==3:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy[0]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                    break
                elif (w.lower() == classy[1]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1
                    break

        count += 1

        #if count1 > (358/2.-1):
        #    break
        #if count1 > (20/2.-1):
        #    break
    '''
    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    sampleTrain1 = 0
    sampleTrain2 = 0
    sampleTest1 = 0
    sampleTest2 = 0
    allDataTrain = []
    allLabelsTrain = []
    allDataTest = []
    allLabelsTest = []
    for s in range(len(allLabels)):
        if (allLabels[s] == 0) and (sampleTrain1 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain1 += 1
        elif (allLabels[s] == 0) and (sampleTest1 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest1 += 1
        if (allLabels[s] == 1) and (sampleTrain2 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            sampleTrain2 += 1
        elif (allLabels[s] == 1) and (sampleTest2 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            sampleTest2 += 1
    '''
    '''
    num = range(len(allData))
    if mode=='train':
        trainMask = [i for i in num if i%10!=0 ]

    elif mode=='eval':
        trainMask = [i for i in num if i%10==0 ]
        
    X = [allData[i] for i in trainMask]
    y = [allLabels[i] for i in trainMask]
    '''
    #random.shuffle(allLabelsTrain)
    #random.shuffle(allLabelsTest)
    X_test,y_test = allData,np.array(allLabels)

    print('Test Samples:',len(X_test))

    return X_test, y_test
    
def getTestList(mode='all',classy='norm'):
    path = '../auto-eeg-diagnosis-example-master/rawNEDCdata.txt'
    allData = []
    allLabels = []
    fileNames = []
    f = open(path,'rU')
    content = f.readlines()

    numFile = len(content)
    count = 0
    count1 = 0
    count2 = 0

    for file in content:

        print('Loading',file.strip(),count,'of',numFile,'(',count1,',',count2,')')
        curLoad = np.load(file.strip(),encoding='bytes')

        curLabel = curLoad[0]
        curData = np.float32(curLoad[1])

        if classy == 'norm':
            if (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                fileNames.append(file.strip())
                count1 += 1
            elif (curLabel[5] == 1) and (count2 < count1):
                allLabels.append(1)
                allData.append(curData)
                fileNames.append(file.strip())
                count2 += 1

        elif classy == 'gender':
            if (curLabel[2].decode('utf-8') == 'male'):
                allLabels.append(0)
                allData.append(curData)
                fileNames.append(file.strip())
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female'):
                allLabels.append(1)
                allData.append(curData)
                fileNames.append(file.strip())
                count2 += 1

        elif classy in ['lowAge','midAge','highAge']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAge':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAge':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAge':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
                elif (curLabel[3][0] not in curAge):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1

        elif classy == 'ageDiff':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]):# and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]):# and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1

        elif classy == 'genderNorm':
            if (curLabel[2].decode('utf-8') == 'male') and (curLabel[5] == 0):
                allLabels.append(0)
                allData.append(curData)
                fileNames.append(file.strip())
                count1 += 1
            elif (curLabel[2].decode('utf-8') == 'female') and (curLabel[5] == 0):
                allLabels.append(1)
                allData.append(curData)
                fileNames.append(file.strip())
                count2 += 1

        elif classy in ['lowAgeNorm','midAgeNorm','highAgeNorm']:
            ageRanges = [1,20,60,120]
            if classy == 'lowAgeNorm':
                curAge = range(ageRanges[0],ageRanges[1])
            if classy == 'midAgeNorm':
                curAge = range(ageRanges[1],ageRanges[2])
            if classy == 'highAgeNorm':
                curAge = range(ageRanges[2],ageRanges[3])                                
            if len(curLabel[3])>0:
                if (curLabel[3][0] in curAge) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
                elif (curLabel[3][0] not in curAge) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1

        elif classy == 'ageDiffNorm':
            ageRanges = [10,60]
            if len(curLabel[3])>0:
                if (curLabel[3][0] < ageRanges[0]) and (curLabel[5] == 0):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
                elif (curLabel[3][0] > ageRanges[1]) and (curLabel[5] == 0):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1

        elif classy in ['dilantin','motrin','ativan','keppra']:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy) and (len(medWords.split()) == 1):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
                    break
                elif (w.lower() == 'none') and (len(medWords.split()) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1
                    break
        
        elif classy == 'medNorm':
            badMeds = ['acetazolamide','carbamazepine','clobazam','clonazepam','eslicarbazepine acetate','ethosuximide','gabapentin','lacosamide','lamotrigine','levetiracetam','nitrazepam','oxcarbazepine','perampanel',
                'piracetam','phenobarbital','phenytoin','pregabalin','primidone','rufinamide','sodium valproate','stiripentol','tiagabine','topiramate','vigabatrin','zonisamide',
                'convulex','desitrend','diacomit','diamox sr','emeside','epanutin','epilim','epilim chrono','epilim chronosphere','epival','frisium','fycompa','gabitril','inovelon','keppra','lamictal','lyrica',
                'neurontin','nootropil','phenytoin sodium flynn','rivotril','sabril','tapclob','tegretol','topamax','trileptal','vimpat','zarontin','zebinix','zonegran']
            try:
                medList = [w.decode('utf-8').lower() for w in curLabel[4].split()]
            except UnicodeDecodeError:
                medList = ['']
                
            if (curLabel[5] == 0) and ('none' in medList) and (len(medList) == 1):
                    allLabels.append(0)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count1 += 1
            elif (curLabel[5] == 1) and (any(x in max(badMeds,medList,key=len) for x in min(badMeds,medList,key=len))):
                    allLabels.append(1)
                    allData.append(curData)
                    fileNames.append(file.strip())
                    count2 += 1


        elif len(classy)==3:
            medWords = curLabel[4]
            for w in medWords.split():
                try:
                    w = w.decode('utf-8')
                except UnicodeDecodeError:
                    w = ''
                if (w.lower() == classy[0]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(0)
                    allData.append(curData)
                    count1 += 1
                    break
                elif (w.lower() == classy[1]) and (len(medWords.split()) == 1) and (curLabel[5] == classy[2]):
                    allLabels.append(1)
                    allData.append(curData)
                    count2 += 1
                    break

        count += 1

        #if count1 > (358/2.-1):
        #    break
        #if count1 > (20/2.-1):
        #    break

    np.save('data%s'%classy,np.array(allData))
    np.save('labels%s'%classy,np.array(allLabels))

    numberEqSamples = min(Counter(allLabels).values())
    trainSamplesNum = int(np.ceil(numberEqSamples*0.9))
    testSamplesNum = numberEqSamples-trainSamplesNum

    sampleTrain1 = 0
    sampleTrain2 = 0
    sampleTest1 = 0
    sampleTest2 = 0
    allDataTrain = []
    allLabelsTrain = []
    allNamesTrain = []
    allDataTest = []
    allLabelsTest = []
    allNamesTest = []
    for s in range(len(allLabels)):
        if (allLabels[s] == 0) and (sampleTrain1 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            allNamesTrain.append(fileNames[s])
            sampleTrain1 += 1
        elif (allLabels[s] == 0) and (sampleTest1 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            allNamesTest.append(fileNames[s])
            sampleTest1 += 1
        if (allLabels[s] == 1) and (sampleTrain2 < trainSamplesNum):
            allLabelsTrain.append(allLabels[s])
            allDataTrain.append(allData[s])
            allNamesTrain.append(fileNames[s])
            sampleTrain2 += 1
        elif (allLabels[s] == 1) and (sampleTest2 < testSamplesNum):
            allLabelsTest.append(allLabels[s])
            allDataTest.append(allData[s])
            allNamesTest.append(fileNames[s])
            sampleTest2 += 1

    '''
    num = range(len(allData))
    if mode=='train':
        trainMask = [i for i in num if i%10!=0 ]

    elif mode=='eval':
        trainMask = [i for i in num if i%10==0 ]
        
    X = [allData[i] for i in trainMask]
    y = [allLabels[i] for i in trainMask]
    '''
    #random.shuffle(allLabelsTrain)
    #random.shuffle(allLabelsTest)
    X,y,X_test,y_test = allDataTrain,np.array(allLabelsTrain),allDataTest,np.array(allLabelsTest)

    print('Train Samples:',len(X),'Test Samples:',len(X_test))

    file = open('../auto-eeg-diagnosis-example-master/'+classy+'TestRawNEDCdata.txt',"w") 
    testNames = np.array(allNamesTest)
    for f in testNames:
        file.write(f+'\n') 
 
    file.close() 

    if mode=='train':
        return X, y
    elif mode=='eval':
        return X_test, y_test
    else:
        return X, y, X_test, y_test


def addDataNoise(origSignals,band=[],channels=[],srate=100):
    np.random.seed(seed=10102018)
    #origSignal = signals
    signals = []
    for s in range(len(origSignals)):
        signals.append(np.float32(np.zeros((origSignals[0].shape[0],origSignals[0].shape[1]))))
    #pdb.set_trace()

    if (len(band)+len(channels)) == 0:
        return origSignals
    
    if (len(channels)>0) and (len(band)==0):
        for s in range(len(signals)):
            for c in channels:
                cleanSignal = origSignals[s][c,:]
                timeDomNoise = np.random.normal(np.mean(cleanSignal), np.std(cleanSignal), size=len(cleanSignal))
                signals[s][c,:] = np.float32(timeDomNoise)#cleanSignal + timeDomNoise

    if (len(band) == 2) and (type(band[0]) == int):
        if len(channels)==0:
            channels = range(signals[0].shape[0])

        numSamples = signals[0].shape[1]
        W = fftpack.rfftfreq(numSamples,d=1./srate)
        lowHz = next(x[0] for x in enumerate(W) if x[1] > band[0])
        highHz = next(x[0] for x in enumerate(W) if x[1] > band[1])
        for s in range(len(signals)):
            for c in channels: #loop through channels
                dataDFT = fftpack.rfft(origSignals[s][c,:])
                cleanDFT = dataDFT[lowHz:highHz]
                freqDomNoise = np.random.normal(np.mean(cleanDFT), np.std(cleanDFT), size=len(cleanDFT))
                dataDFT[lowHz:highHz] =  freqDomNoise#cleanDFT + freqDomNoise
                signals[s][c,:] = np.float32(fftpack.irfft(dataDFT))

    elif (len(band)>0) and (type(band[0]) == list):
        if len(channels)==0:
            channels = range(signals[0].shape[0])
        
        numSamples = signals[0].shape[1]
        W = fftpack.rfftfreq(numSamples,d=1./srate)        
        for s in range(len(signals)):
            for c in channels: #loop through channels
                dataDFT = fftpack.rfft(origSignals[s][c,:])
                for b in band:
                    lowHz = next(x[0] for x in enumerate(W) if x[1] > b[0])
                    highHz = next(x[0] for x in enumerate(W) if x[1] > b[1])
                    cleanDFT = dataDFT[lowHz:highHz]
                    freqDomNoise = np.random.normal(np.mean(cleanDFT), np.std(cleanDFT), size=len(cleanDFT))
                    dataDFT[lowHz:highHz] =  freqDomNoise#cleanDFT + freqDomNoise
                signals[s][c,:] = np.float32(fftpack.irfft(dataDFT))


    return signals


if __name__ == '__main__':
    start = time()

    mode = 'all'
    classy = 'lowAgeNorm'
    #results = loadNEDCdata(mode=mode,classy=classy)

    classes = ['norm','medNorm','gender','genderNorm','lowAge','lowAgeNorm','highAge','highAgeNorm','ageDiff','ageDiffNorm','dilantin','keppra']
    for c in classes:
        results = getTestList(mode=mode,classy=c)
        #print('Train Samples:',len(results[0]),'Test Samples:',len(results[2]))
        #pdb.set_trace()

    end = time()
    print(os.linesep+'Time Elapsed: '+str(end-start)+os.linesep)

    #pdb.set_trace()