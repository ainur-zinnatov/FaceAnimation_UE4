import tensorflow as tf
import librosa
from tensorflow import keras
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.backend import set_session
from tqdm import tqdm
import copy
import math
import cv2
import json, codecs

import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

class landmarkFromAudio(TFPluginAPI):


    def onSetup(self):
        self.fs = 44100
        self.scripts_path = ue.get_content_dir() + "Scripts"
        self.model_directory = self.scripts_path + "/models/D40_C3.h5"
        self.test_file = self.scripts_path +'/test_samples/test1.flac'
        
        
       
        #self.model.summary()
        ue.log("Load modul")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        self.graph = tf.compat.v1.get_default_graph()

        set_session(self.session)
        self.model = load_model(self.model_directory)  
        self.model._make_predict_function()  
        self.sound, self.sr = librosa.load(self.test_file, sr=self.fs)
        ue.log("Load sound")



    def onJsonInput(self, jsonInput):

        if self.model is None:
            ue.log("Warning! No 'model' found. Did training complete?")
            return result

        num_features_Y = 136
        num_frames = 75
        wsize = 0.04
        hsize = wsize
        ctxWin = 3
        ue.log("StartPredict")

        zeroVecD = np.zeros((1, 64), dtype='int16')
        zeroVecDD = np.zeros((2, 64), dtype='int16')

        # Load speech and extract features
        melFrames = np.transpose(self.melSpectra(self.sound, self.sr, wsize, hsize))
        melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
        melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)

        features = np.concatenate((melDelta, melDDelta), axis=1)
        features = self.addContext(features, ctxWin)
        features = np.reshape(features, (1, features.shape[0], features.shape[1]))

        upper_limit = features.shape[1]
        lower = 0
        generated = np.zeros((0, num_features_Y))

        # Generates face landmarks one-by-one
        # This part can be modified to predict the whole sequence at one, but may introduce discontinuities
        for i in tqdm(range(0,upper_limit,2)):
            cur_features = np.zeros((1, num_frames, features.shape[2]))
            if i+1 > 75:
                lower = i+1-75
            cur_features[:,-i-1:,:] = features[:,lower:i+1,:]
            with self.graph.as_default():
                set_session(self.session)
                pred = self.model.predict(cur_features)
                #ue.log('pred:' + str(pred))
            #K.clear_session()
            generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0)

        # Shift the array to remove the delay
        generated = generated[1:, :]
        tmp = generated[-1:, :]
        for _ in range(1):
            generated = np.append(generated, tmp, axis=0)

        if len(generated.shape) < 3:
            generated = np.reshape(generated, (generated.shape[0], generated.shape[1]//2, 2))


        generated = self.alignEyePointsV2(600*generated) / 600.0 

        ue.log("LandmarkOut:")
        ue.log('Output array: ' + str(generated))

        #b = generated.tolist()
        #json_dump = json.dumps({'a': generated})
        #ue.log('Output json_dump: ' + str(json_dump))
        animData = {}
        data = []
        for i in range(generated.shape[0]):
            dataframes = {}
            landmarksArr = []
            for j in range(generated.shape[1]):
                values = {'x':generated[i,j,0], 'y':generated[i,j,1]}
                landmarksArr.append(values)
            dataframes['landmarks']=landmarksArr
            #data['frame'+str(i)] = dataframes
            data.append(dataframes)
            animData['frames'] = data


        result = animData
        result['framesNum'] = generated.shape[0]

        with open(ue.get_content_dir()+'Scripts/animData.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            
        return result



    def addContext(self, melSpc, ctxWin):
        ctx = melSpc[:,:]
        filler = melSpc[0, :]
        for i in range(ctxWin):
            melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
            ctx = np.append(ctx, melSpc, axis=1)
        return ctx

    def alignEyePointsV2(self, lmarkSeq):
        w = 600
        h = 600

        alignedSeq = copy.deepcopy(lmarkSeq)
        
        eyecornerDst = [ (np.float(0.3 * w ), np.float(h / 3)), (np.float(0.7 * w ), np.float(h / 3)) ]

        for i, lmark in enumerate(alignedSeq):
            curLmark = alignedSeq[i,:,:]
            eyecornerSrc  = [ (curLmark[36, 0], curLmark[36, 1]), (curLmark[45, 0], curLmark[45, 1]) ]
            tform = self.similarityTransform(eyecornerSrc, eyecornerDst);
            alignedSeq[i,:,:] = self.tformFlmarks(lmark, tform)

        return alignedSeq

    def tformFlmarks(self, flmark, tform):
        transformed = np.reshape(np.array(flmark), (68, 1, 2))           
        transformed = cv2.transform(transformed, tform)
        transformed = np.float32(np.reshape(transformed, (68, 2)))
        return transformed

    def similarityTransform(self, inPoints, outPoints):
        s60 = math.sin(60*math.pi/180)
        c60 = math.cos(60*math.pi/180)
      
        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        #tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)
        tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
        return tform[0]

    def melSpectra(self, y, sr, wsize, hsize):
        cnst = 1+(int(sr*wsize)/2)
        y_stft_abs = np.abs(librosa.stft(y,
                                      win_length = int(sr*wsize),
                                      hop_length = int(sr*hsize),
                                      n_fft=int(sr*wsize)))/cnst

        melspec = np.log(1e-16+librosa.feature.melspectrogram(sr=sr, 
                                                 S=y_stft_abs**2,
                                                 n_mels=64))
        return melspec





#required function to get our api
def getApi():
    #return CLASSNAME.getInstance()
    return landmarkFromAudio.getInstance()

    # Used for padding zeros to first and second temporal differences


