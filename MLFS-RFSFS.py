import torch
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from skmultilearn.adapt import MLkNN
from pathlib import Path
from LBLC import corr
# Metrics
from sklearn.metrics import coverage_error
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def one_error(outputs, test_target):
    err_cnt = 0
    for i in range(outputs.shape[0]):
        idx = np.argmax(outputs[i])
        if test_target[i, idx] != 1:
            err_cnt += 1
    one_error = err_cnt / outputs.shape[0]
    return one_error

GPU = False
if GPU:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs", torch.cuda.device_count())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    print("CPU")
    
def one_error(outputs, test_target):
    err_cnt = 0
    for i in range(outputs.shape[0]):
        idx = np.argmax(outputs[i])
        if test_target[i, idx] != 1:
            err_cnt += 1
    one_error = err_cnt / outputs.shape[0]
    return one_error


ppn=0

csv_file = f'RFSFS.csv'
if not os.path.isfile(csv_file):
    attrs = ['DB','lam1','lam2','lam3','MIC', 'MAC' , 'AVP', 'HML' , 'RNL'  , 'CVE', 'ZRE','One-E']
    df = pd.DataFrame(columns=attrs)
    df.to_csv(csv_file, index=False)
else:
        df = pd.read_csv(csv_file)

dataset_name = Path(f'{datasetML[db]}').stem

datas =['Arts','Business','Computers','Entertainment','Recreation','Society']

dat = loadmat('../Datasets/' + datas[0] + ".mat")

train = dat['train']
test = dat['test']

X_test = train[0][0].T
Y_test = train[0][1].T
Y_test[Y_test == -1] = 0

X = test[0][0].T
Xc = test[0][0].T
XTX = X.T @ X

Yc = test[0][1].T
Yc[Yc == -1] = 0
Y = Yc

# Feature
n, d = X.shape
# label
n, l = Y.shape

# label correlation
sigma = 12
epsilon = np.finfo(np.float32).eps

S, D = corr(Y)
L = D - S

dd =np.ones((d,d))
#Parameters Truning
alpha  = [0.01,0.1]
beta   = [0.01,0.1]
gamma  = [0.01,0.1]

t = 50

for a in alpha:
        for b in beta:
            for g in gamma:
                erro= torch.zeros(t)
                MICAVG = torch.zeros((3))
                MACAVG = torch.zeros((3))
                AVPAVG = torch.zeros((3))
                HMLAVG = torch.zeros((3))
                RNLAVG = torch.zeros((3))
                CVEAVG = torch.zeros((3))
                ZERAVG = torch.zeros((3))   
                AVPAVG = torch.zeros((3))  
                ONEAVG = torch.zeros((3))
                for avg in range(3):
                    W = np.random.rand(d,l)
                    for i in range(t):
                        Q  = 1/np.maximum( 2 *np.abs(W),epsilon)
                        Wu = (X.T @ Y + a * (W @ S) + g * (W))
                        Wd = (X.T @ X @ W + a * ( W @ D) + b * (Q * W) + g * (dd @ W))
                        W  = W * (Wu / np.maximum(Wd , epsilon))
                    WW = np.linalg.norm(W,axis=1,ord=2)
                    sQ = np.argsort(WW)
                    nosf = int (20 * d / 100)
                    sX = Xc[:,sQ[d-nosf:]]
                    classifier = MLkNN(k=4)
                    classifier.fit(sX, Yc.astype(int))
                    # predict
                    predictions = classifier.predict(X_test[:,sQ[d-nosf:]]).toarray()
                    scores = classifier.predict_proba(X_test[:,sQ[d-nosf:]]).toarray()

                    MIC = f1_score(Y_test, predictions, average='micro')
                    MICAVG[avg] = MIC
                    MAC = f1_score(Y_test, predictions, average='macro')
                    MACAVG[avg] = MAC
                    AVP = average_precision_score(Y_test.T,scores.T)
                    AVPAVG[avg] = AVP
                    HML = hamming_loss(Y_test,predictions)
                    HMLAVG[avg] = HML
                    RNL = label_ranking_loss(Y_test,scores)
                    RNLAVG[avg] = RNL
                    ZER = zero_one_loss(Y_test,predictions)
                    ZERAVG[avg] = ZER
                    CVE = coverage_error(Y_test,scores)
                    CVEAVG[avg] = CVE
                    ONE = one_error(predictions,Y_test)
                    ONEAVG[avg] = ONE

                MICNPY = MICAVG.numpy()
                MACNPY = MACAVG.numpy()
                AVPNPY = AVPAVG.numpy()
                HMLNPY = HMLAVG.numpy()
                RNLNPY = RNLAVG.numpy()
                CVENPY = CVEAVG.numpy()
                ZERNPY = ZERAVG.numpy()
                ONENPY = ONEAVG.numpy()
                new_row = pd.DataFrame({
                        'DB': [dataset_name],
                        'lam1': [a],
                        'lam2': [b],
                        'lam3':[g],
                        'MIC': [np.mean(MICNPY)],
                        'MAC': [np.mean(MACNPY)],
                        'AVP': [np.mean(AVPNPY)],
                        'HML': [np.mean(HMLNPY)],
                        'RNL': [np.mean(RNLNPY)],
                        'CVE': [np.mean(CVENPY)],
                        'ZRE': [np.mean(ZERNPY)],
                        'One-E':[np.mean(ONENPY)],
                        })
                df = pd.concat([df, new_row], ignore_index=True)
                print("Iter: ", ppn)
                ppn +=1
                df.to_csv(csv_file, index=False)
