import torch
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from skmultilearn.adapt import MLkNN
from LBLC import corr
# Metrics
from sklearn.metrics import coverage_error
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
import os
import warnings
warnings.filterwarnings('ignore')


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

datas =['Arts','Business','Computers','Entertainment','Recreation','Society']

dat = loadmat('../../Datasets/' + datas[0] + ".mat")
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
W = np.random.rand(d,l)
#Parameters Truning
alpha  = [0.01,0.1]
beta   = [0.01,0.1]
gamma  = [0.01,0.1]

t = 50
MiF1A=[]
MaF1A=[]
AVP_AVGA=[]
HML_AVGA=[]
RNL_AVGA=[]
CVE_AVGA=[]
ZER_AVGA=[]

for a in alpha:
        for b in beta:
            for g in gamma:
                erro= torch.zeros(t)
                MICAVG = np.zeros((5,20))
                MACAVG = np.zeros((5,20))
                AVPAVG = np.zeros((5,20))
                HMLAVG = np.zeros((5,20))
                RNLAVG = np.zeros((5,20))
                CVEAVG = np.zeros((5,20))
                ZERAVG = np.zeros((5,20))
                for avg in range(5):
                    for i in range(t):
                        Q  = 1/np.maximum( 2 *np.abs(W),epsilon)
                        Wu = (X.T @ Y + a * (W @ S) + g * (W))
                        Wd = (X.T @ X @ W + a * ( W @ D) + b * (Q * W) + g * (dd @ W))
                        W  = W * (Wu / np.maximum(Wd , epsilon))
                    WW = np.linalg.norm(W,axis=1,ord=2)
                    sQ = np.argsort(WW)
                    nu=0
                    for j in tqdm(range(20)):
                        j+=1
                        nosf = int (j * d / 100)
                        sX = X[:,sQ[d-nosf:]]
                        classifier = MLkNN(k=10)
                        classifier.fit(sX, Y.astype(int))
                        # predict
                        predictions = classifier.predict(X_test[:,sQ[d-nosf:]]).toarray()
                        scores = classifier.predict_proba(X_test[:,sQ[d-nosf:]]).toarray()

                        KNN_m1 = f1_score(Y_test, predictions, average='micro')
                        MICAVG[avg,nu] = KNN_m1
                        KNN_m2 = f1_score(Y_test, predictions, average='macro')
                        MACAVG[avg,nu]=KNN_m2
                        KAVP = average_precision_score(Y_test.T,scores.T)
                        AVPAVG[avg,nu]=KAVP
                        KHML = hamming_loss(Y_test,predictions)
                        HMLAVG[avg,nu]=KHML
                        KRNL = label_ranking_loss(Y_test,scores)
                        RNLAVG[avg,nu]=KRNL
                        ZLos = zero_one_loss(Y_test,predictions)
                        ZERAVG[avg,nu]=ZLos
                        KCOV = coverage_error(Y_test,scores)
                        CVEAVG[avg,nu]=KCOV
                        nu+=1
                for domean in range(20):
                    MiF1=np.mean(MICAVG[:,domean])
                    MiF1A.append(MiF1)
                    MaF1=np.mean(MACAVG[:,domean])
                    MaF1A.append(MaF1)
                    AVP_AVG=np.mean(AVPAVG[:,domean])
                    AVP_AVGA.append(AVP_AVG)
                    HML_AVG=np.mean(HMLAVG[:,domean])
                    HML_AVGA.append(HML_AVG)
                    RNL_AVG=np.mean(RNLAVG[:,domean])
                    RNL_AVGA.append(RNL_AVG)
                    CVE_AVG=np.mean(CVEAVG[:,domean])
                    CVE_AVGA.append(CVE_AVG)
                    ZER_AVG=np.mean(ZERAVG[:,domean])
                    ZER_AVGA.append(ZER_AVG)
                print('MiF1-MEAN-1->20%',MiF1A,'\n',
                      'MaF1-MEAN-1->20%',MaF1A,'\n',
                      'HML-MEAN-1->20%',HML_AVGA,'\n',
                      'RNL-MEAN-1->20%',RNL_AVGA,'\n',
                      'CVE-MEAN-1->20%',CVE_AVGA,'\n',
                      'ZRE-MEAN-1->20%',ZER_AVGA,'\n',)
