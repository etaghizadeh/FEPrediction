###################################################################
#                          FE Prediction                          #																
#                         Elham Taghizadeh                        #
###################################################################

import sys
import os
import h5py as h5
import scipy as sp
from scipy import stats
import matplotlib.mlab as mm
import math
import shutil
import subprocess
#from time import gmtime, strftime
from operator import itemgetter
import pylab as pl
from guidata.hdf5io import HDF5Reader, HDF5Writer
from guidata.tests.all_items import TestParameters
from guidata.dataset.dataitems import StringItem

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation

nfolds = 89                                                                                                                                                                                                                                                                                                                                                                                                                                                   
numBins = 1000
impThr = 0.95

corrCoef = np.zeros((nfolds, 1));
MSE_CT = np.zeros((nfolds, 1));
err_CT = np.zeros((nfolds, 1));
corrCoefImp = np.zeros((nfolds, 1));
MSE_CTImp = np.zeros((nfolds, 1));
err_CTImp = np.zeros((nfolds, 1));

corrCoefxRay = np.zeros((nfolds, 1));
MSE_xRay = np.zeros((nfolds, 1));
err_xRay = np.zeros((nfolds, 1));
corrCoefImpxRay = np.zeros((nfolds, 1));
MSE_xRayImp = np.zeros((nfolds, 1));
err_xRayImp = np.zeros((nfolds, 1));

corrCoefDirect = np.zeros((nfolds, 1));
MSE_Direct = np.zeros((nfolds, 1));
corrCoefSVR = np.zeros((nfolds, 1));
err_Direct = np.zeros((nfolds, 1));
errHist = np.zeros((nfolds, numBins));
Bins = np.zeros((nfolds, numBins+1))
errHistXray = np.zeros((nfolds, numBins))
BinsXray = np.zeros((nfolds, numBins+1))
errHistDirect = np.zeros((nfolds, numBins))
BinsDirect = np.zeros((nfolds, numBins+1))


errShape = np.zeros((nfolds, 1))
varErrShape = np.zeros((nfolds, 1));
err = np.zeros((nfolds, 1))
errTop = np.zeros((nfolds, 1))
errNeck = np.zeros((nfolds, 1))
errShaft = np.zeros((nfolds, 1))
errTroch = np.zeros((nfolds, 1))
errOther = np.zeros((nfolds, 1))
    
for fold in range(0, nfolds):
    with h5.File('G:/Dropbox (CBG)/Comp. Bioeng. Team/Projects/Statistical FEM/Full Leg SSM/FEPrediction/crossValid/' +str(fold+1) +'/train.hdf5', 'r') as hdf5read:
        lambdaShape = hdf5read[str('/model/eigVal_shape')][:].T
        lambdaDensity = hdf5read[str('/model/eigVal_density')][:].T
        wShape = 1 #np.sum(lambdaDensity) / np.sum(lambdaShape)
        L = list()
        hdf5read['/shapeScores'].visit(L.append)
        tempshape = hdf5read[str('/shapeScores/' + L[0])][:]
        tempdensity = hdf5read[str('/densityScores/' + L[0])][:]
        tempStress = hdf5read[str('/stressScores/' + L[0])][:]
        train = np.zeros([len(L), len(tempshape) + len(tempdensity)])
        trainS = np.zeros([len(L), len(tempStress)])
        trainLog = np.zeros((len(L), 4));
        trainxRay = np.zeros([len(L), 149])
        j = 0;
        for bones in L:
            shapes = hdf5read[str('/shapeScores/' + L[j])][:].T
            density = hdf5read[str('/densityScores/' + L[j])][:].T
            trainLog[j, :] = hdf5read[str('/logs/' + L[j])][:].T
            trainS[j, :] = hdf5read[str('/stressScores/' + L[j])][:].T
            train[j, :] = np.concatenate((shapes/wShape, density), axis=1);
            trainxRay[j, :] = hdf5read[str('/xRayFeat/' + L[j])][:].T
            j = j+1;
        #shapePC = hdf5read['/model/PC_shape'][:]
        #meanShape = hdf5read['/model/mean_shape'][:]
        #meanDensity = hdf5read['/model/mean_density'][:]
        #densityPC = hdf5read['/model/PC_density'][:]
        sVec = hdf5read['/model/PC_stress'][:]
        sMean = hdf5read['/model/mean_stress'][:].T
    with h5.File('G:/Dropbox (CBG)/Comp. Bioeng. Team/Projects/Statistical FEM/Full Leg SSM/FEPrediction/crossValid/' +str(fold+1) +'/test.hdf5', 'r') as hdf5read:
        L = list()
        hdf5read['/shapeScores'].visit(L.append)
        test = np.zeros([len(L), len(tempshape) + len(tempdensity)])
        testS = np.zeros([len(L), len(tempStress)])
        testxRay = np.zeros([len(L), 149])
        testLog = np.zeros((len(L), 4));
        j = 0;
        for bones in L:
            shapes = hdf5read[str('/shapeScores/' + L[j])][:].T
            density = hdf5read[str('/densityScores/' + L[j])][:].T
            testLog[j, :] = hdf5read[str('/logs/' + L[j])][:].T
            test[j, :] = np.concatenate((shapes/wShape, density), axis=1);
            testS[j, :] = hdf5read[str('/stressScores/' + L[j])][:].T
            testxRay[j, :] = hdf5read[str('/xRayFeat/' + L[j])][:].T
            j = j+1
    
    sMises = np.zeros((test.shape[0], sVec.shape[1]))
    with h5.File('G:/Dropbox (CBG)/Comp. Bioeng. Team/Projects/Statistical FEM/Full Leg SSM/FEPrediction/Data157.hdf5') as hdf5read:
        j = 0;
        for bones in L:
            data = hdf5read[str('/simOutput/stress/' + L[j])][:].T
            sMises[j, :] = data[1, :]
            j = j+1;
    #sMises = np.tile(sMean, (testS.shape[0], 1)) + testS.dot(sVec)
    sPredicted = np.zeros((test.shape[0], sMises.shape[1]))
    sPredictedxRay = np.zeros((test.shape[0], sMises.shape[1]))
    sPredictedDirect = np.zeros((test.shape[0], sMises.shape[1]))
    
    sPredictedImp = np.zeros((test.shape[0], sMises.shape[1]))
    sPredictedImpxRay = np.zeros((test.shape[0], sMises.shape[1]))

    sParam = np.zeros((test.shape[0], trainS.shape[1]))
    sParamxRay = np.zeros((test.shape[0], trainS.shape[1]))
    
    sParamImp = np.zeros((test.shape[0], trainS.shape[1]))
    sParamImpxRay = np.zeros((test.shape[0], trainS.shape[1]))


    data = list()
    data.append('1.')
    data.append('2.')
    data.append('3.')
    data.append('4.')
    data.append('5.')
    data.append('11.')
    data.append('6.')
    data.append('9.')
    data.append('7.')
    data.append('8.')
    data.append('10.')
    
    data.append('12.')
    data.append('15.')
    data.append('14.')
    data.append('13.')
    data.append('17.')
    data.append('21.')
    data.append('16.')
    data.append('20.')
    data.append('18.')
    data.append('19.')
    for i in range(0, 128):
        data.append('den.' + str(i+1))
    data.append('gender')
    data.append('age')
    data.append('height')
    data.append('weight')
    data = np.asanyarray(data)
    
    
    bParams = np.zeros((testxRay.shape[0], train.shape[1]))
    bParamsImp = np.zeros((testxRay.shape[0], train.shape[1]))
    
    fet_indXray = np.zeros((train.shape[1], trainxRay.shape[1] + 4), 'int')
    totalImportanceShape = np.zeros((trainxRay.shape[1] + 4, 1))
    for i in range(0, train.shape[1]):
        cvfolds = 5
        minScore = 9e10
        bestDepth = 15
        for j in range (5, 13):
            clfSSM = RandomForestRegressor(n_estimators=200, max_features=train.shape[1]/3, max_depth = j, random_state=0, bootstrap=False, n_jobs=-1)
            score = cross_validation.cross_val_score(clfSSM, np.concatenate((trainxRay, trainLog), axis=1), train[:, i], cv=cvfolds, scoring='mean_squared_error')
            MEAN = np.mean(abs(score))
            if ( MEAN< minScore):
                minScore = MEAN
                bestDepth = j
        clfSSM = RandomForestRegressor(n_estimators=1000, max_features=train.shape[1]/3, max_depth = bestDepth, random_state=0, bootstrap=False, n_jobs=-1)
        inpTrain = np.concatenate((trainxRay, trainLog), axis =1)
        clfSSM.fit(inpTrain, train[:, i]);
        inpTest = np.concatenate((testxRay, testLog), axis=1)
        bParams[:, i] = clfSSM.predict(inpTest)
               

    ####0: R_head, 1: R_troch, 2: R_Condyle1, 3: R_Condyle2, 4: bone length, 5:
    #### neckangle, 6: neck length, 7: condyle_size, 8: shaft width, 9:
    #### cortex width, 10: small trochanter to neck end(lm #2)
    ####lateral:
    ####11: R_head, 12: R_condylbot, 13: R_condyltop, 14: R_shaftCenter, 15:
    ####neck length, 16: neck angle, 17: bone length, 18: condyle size, 19:
    ####shaft width, 20: cortex width
    data = list();
    data.append('')

    trainMises = np.zeros((train.shape[0], sVec.shape[1]))
    for j in range(0, train.shape[0]):
        trainMises[j, :] = sMean + (trainS[j, :]).dot(sVec)
    
#    fig = plt.figure(figsize=(12,3));
#    ax = plt.subplot(111);
#    fet_ind = [i[0] for i in sorted(enumerate(totalImportanceShape), key = lambda x:x[1], reverse=True)]
#    totalImportanceShape = np.asarray(sorted(totalImportanceShape/train.shape[1], reverse=True))
#    plt.bar(np.arange(len(totalImportanceShape)), totalImportanceShape*100, width=1, lw=2);
#    plt.grid(False);
#    ax.set_xticks(np.arange(len(totalImportanceShape))+.5);
#    xTickLables = data[fet_ind[0:15]]
#    ax.set_xticklabels(xTickLables);
#    plt.xlabel('Variables')
#    plt.ylabel('Importance')
##    plt.ylim([0, 60])
#    plt.xlim(0, 15);
#    plt.title('bone number ' + str(fold+1) + ' Shape prediction')
#    plt.show()
    

    data = list();
    for i in range(0, tempshape.shape[0]):
        data.append('b.Sh' + str(i+1))
    for i in range(0, tempdensity.shape[0]):
        data.append('b.Dns' + str(i+1))
    data.append('gender')
    data.append('age')
    data.append('height')
    data.append('weight')
    data = np.asanyarray(data)
    
    
    totalImportance = np.zeros((train.shape[1]+4, 1))
    fet_ind = np.zeros((trainS.shape[1], train.shape[1] + 4), 'int')
    
    for i in range(0, trainS.shape[1]):
        cvfolds = 5
        minScore = 9e10
        bestDepth = 15
        for j in range (5, 13):
            clf = RandomForestRegressor(n_estimators=200, max_features=train.shape[1]/3, max_depth = j, random_state=0, bootstrap=False, n_jobs=-1)
            score = cross_validation.cross_val_score(clf, np.concatenate((train, trainLog), axis=1), trainS[:, j], cv=cvfolds, scoring='mean_squared_error')
            MEAN = np.mean(abs(score))
            if ( MEAN < minScore):
                minScore = MEAN
                bestDepth = j
        inpTrain = np.concatenate((train, trainLog), axis=1)
        inpTest = np.concatenate((test, testLog), axis = 1)
        clf = RandomForestRegressor(n_estimators=1000, max_features=train.shape[1]/3, max_depth = bestDepth, random_state=0, bootstrap=False, n_jobs=-1)
        clf.fit(np.concatenate((train, trainLog), axis=1), trainS[:, i])
        sParam[:, i] = clf.predict(np.concatenate((test, testLog), axis = 1))
        sParamxRay[:, i] = clf.predict(np.concatenate((bParams, testLog), axis=1))
        
        fet_ind[i, :] = np.argsort(clf.feature_importances_)[::-1]  
        fet_ind[i, :] = np.argsort(clf.feature_importances_)[::-1]
        fet_imp = clf.feature_importances_[fet_ind[i, :]]
        totalImportance[fet_ind[i, :], 0] = totalImportance[fet_ind[i, :], 0] + fet_imp
        if (i<3):
            fig = plt.figure(figsize=(8,3));    
            ax = plt.subplot(111);
            plt.bar(np.arange(len(fet_imp)), fet_imp*100, width=1, lw=2);
            plt.grid(False);
            ax.set_xticks(np.arange(len(fet_imp))+.5);
            xTickLables = data[fet_ind[i, 0:10]]
            ax.set_xticklabels(xTickLables);
            plt.xlabel('Variables')
            plt.ylabel('Importance')
            plt.ylim([0, 35])
            plt.xlim(0, 10);
            plt.show()
        numMostImp = len(fet_imp) - np.sum(np.cumsum(fet_imp)>impThr)
        clf = RandomForestRegressor(n_estimators=1000, max_features=numMostImp, max_depth = bestDepth, random_state=0, bootstrap=False, n_jobs=-1)
        clf.fit(inpTrain[:, fet_ind[i, 0:numMostImp]], trainS[:, i])
        sParamImp[:, i] = clf.predict(inpTest[:, fet_ind[i, 0:numMostImp]])
        inpTestxRay = np.concatenate((bParamsImp, testLog), axis=1)
        sParamImpxRay[:, i] = clf.predict(inpTestxRay[:, fet_ind[i, 0:numMostImp]])
        

    cvfolds = 5
    minScore = 9e10
    bestDepth = 15
    for j in range (5, 13):
        clf = RandomForestRegressor(n_estimators=200, max_features=train.shape[1]/3, max_depth = j, random_state=0, bootstrap=False, n_jobs=-1)
        score = cross_validation.cross_val_score(clf, np.concatenate((train, trainLog), axis=1), trainS, cv=cvfolds, scoring='mean_squared_error')
        MEAN = np.mean(abs(score))
        if ( MEAN < minScore):
            minScore = MEAN
            bestDepth = j
    inpTrain = np.concatenate((train, trainLog), axis=1)
    inpTest = np.concatenate((test, testLog), axis = 1)
    clf = RandomForestRegressor(n_estimators=1000, max_features=train.shape[1]/3, max_depth = bestDepth, random_state=0, bootstrap=False, n_jobs=-1)
    clf.fit(np.concatenate((train, trainLog), axis=1), trainS)
    sParam = clf.predict(np.concatenate((test, testLog), axis = 1))
        
    fet_ind = np.argsort(clf.feature_importances_)[::-1]  
    fet_ind = np.argsort(clf.feature_importances_)[::-1]
    fet_imp = clf.feature_importances_[fet_ind]
#
#    fig = plt.figure(figsize=(12,3));    
#    ax = plt.subplot(111);
#    plt.bar(np.arange(len(fet_imp)), fet_imp*100, width=1, lw=2);
#    plt.grid(False);
#    ax.set_xticks(np.arange(len(fet_imp))+.5);
#    xTickLables = data[fet_ind[0:15]]
#    ax.set_xticklabels(xTickLables);
#    plt.xlabel('Variables')
#    plt.ylabel('Importance')
#    plt.ylim([0, 20])
#    plt.xlim(0, 15);
#    plt.show()
#     
    
#    fig = plt.figure(figsize=(12,3));
#    ax = plt.subplot(111);
#    fet_ind = [i[0] for i in sorted(enumerate(totalImportance), key = lambda x:x[1], reverse=True)]
#    totalImportance = np.asarray(sorted(totalImportance/trainS.shape[1], reverse=True))
#    plt.bar(np.arange(len(totalImportance)), totalImportance*100, width=1, lw=2);

    for i in range (0, test.shape[0]):
        sPredicted[i, :] = sMean + (sParam[i, :]).dot(sVec)
        sPredictedImp[i, :] = sMean + (sParamImp[i, :]).dot(sVec)
    CC = np.zeros((test.shape[0], 1))
    CCImp = np.zeros((test.shape[0], 1))
    for i in range(0, test.shape[0]):
        CC[i] = np.corrcoef(sMises[i, :], sPredicted[i, :])[0, 1]
        CCImp[i] = np.corrcoef(sMises[i, :], sPredictedImp[i, :])[0, 1]
    corrCoef[fold] = np.mean(CC)
    corrCoefImp[fold] = np.mean(CCImp)
    MSE_CT[fold] = mean_squared_error(sMises[0, :], sPredicted[0, :])
#    err_CT[fold] = np.mean(sPredicted[0, :] - sMises[0, :])
    MSE_CTImp[fold] = mean_squared_error(sMises[0, :], sPredictedImp[0, :])
#    err_CT[fold] = np.mean(sPredictedImp[0, :] - sMises[0, :])
    
    #plt.plot(trainS[:, 0], '.')
    #plt.plot(testS[:, 0], 'r.')
    #plt.plot(sParam[:, 0], 'g.')
    #plt.show()

    f, axarr = plt.subplots(1, 1)

    for i in range (0, test.shape[0]):
        sPredictedxRay[i, :] = sMean + (sParamxRay[i, :]).dot(sVec)
        sPredictedImpxRay[i, :] = sMean + (sParamImpxRay[i, :]).dot(sVec)
    CCxRay = np.zeros((test.shape[0], 1))
    CCImpxRay = np.zeros((test.shape[0], 1))
    for i in range(0, test.shape[0]):
        CCxRay[i] = np.corrcoef(sMises[i, :], sPredictedxRay[i, :])[0, 1]
        CCImpxRay[i] = np.corrcoef(sMises[i, :], sPredictedImpxRay[i, :])[0, 1]
    corrCoefxRay[fold] = np.mean(CCxRay)
    corrCoefImpxRay[fold] = np.mean(CCImpxRay)
    MSE_xRay[fold] = mean_squared_error(sMises[0, :], sPredictedxRay[0, :])
    err_xRay[fold] = np.mean(sPredictedxRay[0, :] - sMises[0, :])
    MSE_xRayImp[fold] = mean_squared_error(sMises[0, :], sPredictedImpxRay[0, :])
    err_xRayImp[fold] = np.mean(np.abs(sPredictedImpxRay[0, :] - sMises[0, :]))

    #plt.plot(sMises[0, :], sMises[0, :], '.y')


    #axarr[0, 1].plot(sMises[1, :], sPredicted[1, :], '.')
    #axarr[0, 1].plot(sMean.T, sPredicted[1, :], '.r')
    #axarr[0, 1].plot(sMean.T, sMean.T, '.g')

    
    #axarr[1, 0].plot(sMises[2, :], sPredicted[2, :], '.')
    #axarr[1, 0].plot(sMean.T, sPredicted[2, :], '.r')
    #axarr[1, 0].plot(sMean.T, sMean.T, '.g')


    #axarr[1, 1].plot(sMises[3, :], sPredicted[3, :], '.')
    #axarr[1, 1].plot(sMean.T, sPredicted[3, :], '.r')
    #axarr[1, 1].plot(sMean.T, sMean.T, '.g')


    #axarr[2, 0].plot(sMises[4, :], sPredicted[4, :], '.')
    #axarr[2, 0].plot(sMean.T, sPredicted[4, :], '.r')
    #axarr[2, 0].plot(sMean.T, sMean.T, '.g')


    #axarr[2, 1].plot(sMises[5, :], sPredicted[5, :], '.')
    #axarr[2, 1].plot(sMean.T, sPredicted[5, :], '.r')
    #axarr[2, 1].plot(sMean.T, sMean.T, '.g')
    
    
#    CCDirect = np.zeros((test.shape[0], 1))
#    for i in range (0, test.shape[0]):
#        sPredictedDirect[i, :] = sMean + (sParamDirect[i, :]).dot(sVec)
#    for i in range(0, test.shape[0]):
#        CCDirect[i] = np.corrcoef(sMises[i, :], sPredictedDirect[i, :])[0, 1]    
#    corrCoefDirect[fold] = np.mean(CCDirect)
#    MSE_Direct[fold] = mean_squared_error(sMises[0, :], sPredictedDirect[0, :])
#    err_Direct[fold] = sPredicted[0, :] - sMises[0, :]

#    plt.plot(sMises[0, :], sPredicted[0, :], '.b')
#    plt.plot(sMises[0, :], sPredictedxRay[0, :], '.r')
#    #plt.plot(sMises[0, :], sPredictedDirect[0, :], '.y')
#    plt.plot(sMises[0, :], sMises[0, :], '.g')
    errHist[fold, :], Bins[fold, :] = np.histogram(err_CT, bins = numBins, density=True)
    errHistXray[fold, :], BinsXray[fold, :] = np.histogram(err_xRay, bins = numBins, density=True)
    errHistDirect[fold, :], BinsDirect[fold, :] = np.histogram(err_Direct, bins = numBins, density=True)
    print '*************************************************************************************************'
    print '*************************** Calculations for Bone Number "' + str(fold+1) +  '" finished ***************************'
    print '*************************************************************************************************'
print np.sqrt(MSE_CT)
print np.sqrt(MSE_xRay)
print np.sqrt(MSE_Direct)
