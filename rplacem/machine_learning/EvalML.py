import numpy as np
import os
from rplacem import var as var
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class AlgoParam:
    '''
    Object storing metadata/info about what produced the data
    type: str
        can be 'regression', 'continuous', 'classification'
    '''
    def __init__(self, type='regression', test2023=False, n_features=None, calibrate_pred=False,
                 num_rounds=None, learning_rate=None, max_depth=None, min_child_weight=None, subsample=None, colsample=None, 
                 log_subtract_transform=None, weight_highEarliness=1):
        self.test2023 = test2023
        self.num_rounds = num_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample
        self.log_subtract_transform = log_subtract_transform
        self.type = type
        self.weight_highEarliness = weight_highEarliness
        self.n_features = n_features
        self.calibrate_pred = calibrate_pred

    # transformation of researched output
    def transform_target(self, targ):
        return np.log10(100. + targ) - self.log_subtract_transform

    def invtransform_target(self, targ):
        return np.power(10, targ + self.log_subtract_transform) - 100.

class EvalML:
    '''
    Class containing evaluation results and hyper parameters used in the related training.

    classification: boolean
        True if it comes from classification, in which case the true_threshold is 0.5
    true_threshold: float
        Signal is considered to have (true value < true_threshold)
    reversePredCut: boolean
        When True, the signal is considered with a lower cut (true value > true_threshold)
    pred_thresholds: 1d numpy array of length [n_pred_cuts]
        all the thresholds that will be tested on the variable to form to ROC and PR curves
    computeTN: boolean
        the true negatives are needed only for accuracy, so it might not be calculated at user's request

    TP, TN, FP, FN: 1d numpy array of length [n_pred_cuts]
        numbers of true(T)/false(F) positives(P)/negatives(P)
    TPR, aka recall(), aka sensitivity(), aka signal efficiency: 1d numpy array of length [n_pred_cuts]
        true positive rate: (# true positives / # total signal)
    FPR, aka (1 - background efficiency), aka (1 - specificity()): 1d numpy array of length [n_pred_cuts]
        false positive rate: (# false positives / # total background)
    purity, aka precision(): 1d numpy array of length [n_pred_cuts]
        (# true positives / # all positives)
    F1: 1d numpy array of length [n_pred_cuts]
        2*TPR*purity / (purity+TPR)
    accuracy: 1d numpy array of length [n_pred_cuts]
        TP + TN / (# total events)
    '''
    def __init__(self, true, predicted, true_threshold, variableName='predicted_earliness', algo_param=AlgoParam(),
                 n_pred_cuts=200, pred_thresholds=None, transformed_earliness=False, computeTN=True, force_noReversePredCut=False,
                 trainsample=False, ratioEvals=False):
        
        self.algo_param = algo_param
        self.variableName = ('transformed_' if transformed_earliness else '') + variableName
        self.trainsample = trainsample
        self.ratioEvals = ratioEvals
        self.computeTN = computeTN
        self.reversePredCut = False if force_noReversePredCut else None
        self.force_noReversePredCut = force_noReversePredCut


        self.true_threshold = true_threshold
        self.true_thres_internal_ = 0.5 if self.algo_param.type == 'classification' else self.true_threshold #in the middle of true and false for classification results
        self.signalIsAboveThreshold_ = (self.algo_param.type == 'classification')
        self.n_pred_cuts = n_pred_cuts
            
        # filled in compute_all_measures()
        self.FN = None
        self.TPR = None
        self.FPR = None
        self.purity = None
        self.F1 = None
        self.accuracy = None
        self.ROCAUC = None
        self.PRAUC = None
        
        if pred_thresholds is None and not ratioEvals:
            self.set_pred_thresholds(true, predicted)
        else:
            self.pred_thresholds = pred_thresholds
            
        if not self.ratioEvals:
            # unique numbers for the whole sample
            self.n_tot = len(true)
            self.n_signal = np.count_nonzero(true > self.true_thres_internal_) if self.signalIsAboveThreshold_ else np.count_nonzero(true < self.true_thres_internal_)
            self.n_background = self.n_tot - self.n_signal
            self.signal_fraction = self.n_signal / self.n_tot

            # filled in compute_base()
            self.TP = np.zeros(len(self.pred_thresholds)+2)
            self.FP = np.zeros(len(self.pred_thresholds)+2)
            self.TN = np.zeros(len(self.pred_thresholds)+2)

            # compute basic variables
            self.compute_base(true, predicted)

    # aliases
    def recall(self):
        return self.TPR
    def sensitivity(self):
        return self.TPR
    def precision(self):
        return self.purity
    def specificity(self):
        return 1 - self.FPR

    def set_pred_thresholds(self, true, pred):
        pred_sigorbkg1 = pred[true > self.true_thres_internal_]
        pred_sigorbkg2 = pred[true < self.true_thres_internal_]
        pred_deciles1 = np.quantile(pred_sigorbkg1, np.linspace(0, 1, int(self.n_pred_cuts/2)+1))
        pred_deciles2 = np.quantile(pred_sigorbkg2, np.linspace(0, 1, int(self.n_pred_cuts/2)+1))
        self.pred_thresholds = np.unique(np.concatenate((pred_deciles1, pred_deciles2))) # outputs a sorted array

    def set_reversePredCut(self, true, pred):
        # compare values of TPR and FPR and the central value of pred_thresholds
        predCut_trial = self.pred_thresholds[int(len(self.pred_thresholds)/2)]
        if self.signalIsAboveThreshold_:
            TPR_trial = np.count_nonzero((true > self.true_thres_internal_) & (pred < predCut_trial)) / self.n_signal
            FPR_trial = np.count_nonzero((true <= self.true_thres_internal_) & (pred < predCut_trial)) / self.n_background
        else:
            TPR_trial = np.count_nonzero((true < self.true_thres_internal_) & (pred < predCut_trial)) / self.n_signal
            FPR_trial = np.count_nonzero((true >= self.true_thres_internal_) & (pred < predCut_trial)) / self.n_background
        if (TPR_trial == 0 and FPR_trial == 0) or (TPR_trial == 1 and FPR_trial == 1):
            print('EvalML set_reversePredCut() error: pred_thresholds do not seem appropriate, TPR and FPR are 0 or 1')
        self.reversePredCut = (TPR_trial < FPR_trial)

    def compute_base(self, true, pred):
        if not self.force_noReversePredCut:
            self.set_reversePredCut(true, pred)
        invertTrueCut = -1 if self.signalIsAboveThreshold_ else 1
        invertPredCut = -1 if self.reversePredCut else 1
        for i, pred_thres in enumerate(list(self.pred_thresholds)):
            self.TP[i+1] = np.count_nonzero((invertTrueCut*true < invertTrueCut*self.true_thres_internal_) & (invertPredCut*pred < invertPredCut*pred_thres))
            self.FP[i+1] = np.count_nonzero((invertTrueCut*true >= invertTrueCut*self.true_thres_internal_) & (invertPredCut*pred < invertPredCut*pred_thres))
            if self.computeTN:
                self.TN[i+1] = np.count_nonzero((invertTrueCut*true >= invertTrueCut*self.true_thres_internal_) & (invertPredCut*pred >= invertPredCut*pred_thres))

        # set the endpoint depending on the direction that TP,FP,TN are increasing
        self.complete_endpoints(self.TP, 0, self.n_signal)
        self.complete_endpoints(self.FP, 0, self.n_background)
        if self.computeTN:
            self.complete_endpoints(self.TN, 0, self.n_background)

    def set_FN(self):
        if self.FN is None:
            self.FN = self.n_signal - self.TP
    def set_TPR(self):
        if self.TPR is None:
            self.TPR = self.TP / self.n_signal
    def set_FPR(self):
        if self.FPR is None:
            self.FPR = self.FP / self.n_background

    def set_purity(self):
        if self.purity is None:
            with np.errstate(divide='ignore', invalid='ignore'):
                self.purity = self.TP / (self.TP + self.FP)
            self.treatNaNs(self.purity)

    def set_F1(self):
        if self.F1 is None:
            self.set_TPR()
            self.set_purity()
            with np.errstate(divide='ignore', invalid='ignore'):
                self.F1 = 2 * self.TPR * self.purity / (self.purity + self.TPR)
            self.treatNaNs(self.F1)

    def set_accuracy(self):
        if self.accuracy is None and self.computeTN:
            self.accuracy = (self.TP + self.TN) / self.n_tot

    def compute_all(self):
            if not self.ratioEvals:
                self.set_FN()
                self.set_TPR()
                self.set_FPR()
                self.set_purity()
                self.set_F1()
                self.set_accuracy()
                self.calc_ROCAUC()
                self.calc_PRAUC()

    def complete_endpoints(self, a, begpoint=0, endpoint=1, addelements=False):
        '''
        add clean endpoints to array a
        '''
        if endpoint < begpoint:
            print('WARNING complete_endpoints: must have begpoint < endpoint')
            return a
        if len(a) < 2:
            return a
        if addelements:
            a = np.concatenate(([begpoint], a, [endpoint]))
        increase = (a[-2] > a[1])
        a[0], a[-1] = (begpoint, endpoint) if increase else (endpoint, begpoint)
        return a
    
    def treatNaNs(self, a):
        '''
        Replace NaN's by the value of their closest neighbor
        '''
        naninds = np.where(np.isnan(a))[0]
        for i in naninds:
            if i<int(len(a)/2):
                np.min(~np.isnan(a))
                a[i] = a[np.min(np.where(~np.isnan(a))[0])]
            else:
                a[i] = a[np.max(np.where(~np.isnan(a))[0])]

    def plotROC(self, othername=''):
        plt.figure(num=1, clear=True)

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.plot([0,1],[0,1], color='black', linestyle='--')
        self.set_FPR()
        self.set_TPR()
        plt.plot(self.FPR, self.TPR)
        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'ROC_earlinessBelow'+str(self.true_threshold)+'s_'+self.variableName+('_LowerCut' if self.reversePredCut else '_UpperCut')+'.png' if othername == '' else othername), 
                    dpi=250, bbox_inches='tight')

    def plotPR(self, othername=''):
        plt.figure(num=1, clear=True)

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.ylabel('purity (precision)')
        plt.xlabel('true positive rate (recall)')
        self.set_purity()
        self.set_TPR()
        plt.plot(self.TPR, self.purity)

        plt.savefig(os.path.join(var.FIGS_PATH, 'ML', 'PR_earlinessBelow'+str(self.true_threshold)+'s_'+self.variableName+('_LowerCut' if self.reversePredCut else '_UpperCut')+'.png' if othername == '' else othername), 
                    dpi=250, bbox_inches='tight')

    def interp_bothdirections(x, xp, yp):
        if xp[2]>xp[1]:
            return np.interp(x, xp, yp)
        else:
            return np.interp(x, np.flip(xp), np.flip(yp))

    def calc_ROCAUC(self, maxFPR=1, verbose=False):
        if self.ROCAUC == None or maxFPR != 1:
            self.set_FPR()
            self.set_TPR()
            TPRatMaxFPR = np.interp(maxFPR, self.FPR, self.TPR)
            TPRtmp = self.complete_endpoints(self.TPR[self.FPR <= maxFPR], 0, TPRatMaxFPR, True)
            FPRtmp = self.complete_endpoints(self.FPR[self.FPR <= maxFPR], 0, maxFPR, True)
            if verbose:
                print(TPRtmp, FPRtmp,  simpson(TPRtmp, FPRtmp))
            auc = simpson(TPRtmp, FPRtmp) / maxFPR
            if maxFPR == 1:
                self.ROCAUC = auc
        return auc if maxFPR != 1 else self.ROCAUC
    
    def calc_PRAUC(self):
        if self.PRAUC == None:
            self.set_purity()
            self.set_TPR()
            self.PRAUC = simpson(self.precision(), self.TPR)
        return self.PRAUC

    def FPRatSomeTPR(self, TPRval=0.5):
        self.set_FPR()
        self.set_TPR()
        return np.interp(TPRval, self.TPR, self.FPR)

    def printResults(self):
        print()
        self.compute_all()
        print('********** Evaluation results for (', self.algo_param.type,
              ') signal = ',self.variableName,('>' if self.reversePredCut else '<'), self.true_threshold, 'train/test' if self.ratioEvals else '')
        if not self.ratioEvals:
            print('ntot, signal fraction = ', self.n_tot, self.signal_fraction)
        print('ROC AUC, PR AUC = ', self.ROCAUC, self.PRAUC)
        if not self.ratioEvals:
            print('ROC AUC with FPR<0.1 = ', self.calc_ROCAUC(maxFPR=0.1))
            print('ROC AUC with FPR<0.3 = ', self.calc_ROCAUC(maxFPR=0.3))
            print('FPR at TPR=(0.3, 0.5, 0.8) = ', self.FPRatSomeTPR(0.3), self.FPRatSomeTPR(0.5), self.FPRatSomeTPR(0.8))
            #print('TP, FP =',  self.TP, self.FP)
        #print('TPR, FPR, purity, F1, accuracy =',  self.TPR, self.FPR, self.purity, self.F1, self.accuracy)

def ratioVars(a, b):
    ratio = None
    if a is not None and b is not None:
        ratio = a / b
        ratio[ np.argwhere(np.isnan(ratio) | (np.abs(ratio) == np.inf)) ] = 1
    return ratio

def ratioEvals(eval1, eval2, printmore=False):
    eval1.compute_all()
    eval2.compute_all()
    evalout = EvalML(None, None, algo_param=eval1.algo_param, ratioEvals=True, true_threshold=eval1.true_threshold)
    evalout.TPR = ratioVars(eval1.TPR, eval2.TPR)
    evalout.FPR = ratioVars(eval1.FPR, eval2.FPR)
    evalout.purity = ratioVars(eval1.purity, eval2.purity)
    evalout.F1 = ratioVars(eval1.F1, eval2.F1)
    evalout.accuracy = ratioVars(eval1.accuracy, eval2.accuracy)
    evalout.ROCAUC = eval1.ROCAUC / eval2.ROCAUC
    evalout.PRAUC = eval1.PRAUC / eval2.PRAUC

    if printmore:
        evalout.printResults()
        print('ROC AUC with FPR<0.1 = ', eval1.calc_ROCAUC(maxFPR=0.1) / eval2.calc_ROCAUC(maxFPR=0.1))
        print('ROC AUC with FPR<0.3 = ', eval1.calc_ROCAUC(maxFPR=0.3) / eval2.calc_ROCAUC(maxFPR=0.3))
        print('FPR at TPR=(0.3, 0.5, 0.8) = ', eval1.FPRatSomeTPR(0.3) / eval2.FPRatSomeTPR(0.3), 
            eval1.FPRatSomeTPR(0.5) / eval2.FPRatSomeTPR(0.5), 
            eval1.FPRatSomeTPR(0.8) / eval2.FPRatSomeTPR(0.8))
    
    return evalout