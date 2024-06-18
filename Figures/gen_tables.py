

import argparse
from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d, check_array, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

def calc_eer_(scores_file, y_true_, cfad_provided=False):
    y_true, _, y_proba, y_probb = get_file_values(scores_file, y_true_, cfad_provided)
    fps, tps, _ = _binary_clf_curve(y_true, y_proba, pos_label=1)
    fpr_s, tpr_s = roc_single(fps, tps)
    eer = brentq(lambda x : 1. - x - interp1d(fpr_s, tpr_s)(x), 0., 1.)    
    return eer

def calc_tpfptnfn_(scores_file, y_true, cfad_provided=False):
    
    with open(scores_file, "r") as f:
        lines = f.readlines()

    true_pred = []
    for i, line in enumerate(lines):
        name, _, pred, _, _ = line.split(" ")
        if cfad_provided:
            true_pred.append((y_true[i], pred))
        else:
            true_pred.append((y_true.get(name), pred))
    
    true_pred = np.array(true_pred)
    y_true = true_pred[:,0].astype(np.int8)
    y_pred = true_pred[:,1].astype(np.int8)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    #gotta flip the Negative and Positive class to make deepfakes the positive class
    for i in range(len(true_pred)): 
        y_t = y_true[i]
        y_p = y_pred[i]
        if y_t==y_p==1:
           TP += 1
        if y_p==1 and y_t!=y_p:
           FP += 1
        if y_t==y_p==0:
           TN += 1
        if y_p==0 and y_t!=y_p:
           FN += 1

    return TP, FP, TN, FN

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):


    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    weight = 1.

    threshold_idxs = np.arange(0,y_true.size-1, 1)

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]

    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def roc_single(fps, tps):
     

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    fpr = fps / fps[-1]
    fpr[np.isnan(fpr)] = 0
    tpr = tps / tps[-1]

    return fpr, tpr

def get_y_true(file):
    with open(file, "r") as f:
        lines = f.readlines()
    y_true_dict = {}
    for line in lines:
        try:
            name, type = line.split(" ")
        except:
            name, name1, type = line.split(" ")
            name = f"{name} {name1}"
        type = type.replace("\n", "")
        if type == "bonafide": 
            type = 1
        else:
            type = 0
        y_true_dict[name] = type
    return y_true_dict

def get_y_true_cfad_provided(file):
    with open(file, "r") as f:
        lines = f.readlines()
    y_true = []
    for line in lines:
        _, target, _, _, _ = line.split(" ")
        if target == "1": 
            type = 1
        else:
            type = 0
        y_true.append(type)
    return y_true

def calc_tpfptnfn(scores_file, y_true, cfad_provided=False):
    
    with open(scores_file, "r") as f:
        lines = f.readlines()

    true_pred = []
    for i, line in enumerate(lines):
        name, _, pred, _, _ = line.split(" ")
        if cfad_provided:
            true_pred.append((y_true[i], pred))
        else:
            true_pred.append((y_true.get(name), pred))
    
    true_pred = np.array(true_pred)
    y_true = true_pred[:,0].astype(np.int8)
    y_pred = true_pred[:,1].astype(np.int8)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    #gotta flip the Negative and Positive class to make deepfakes the positive class
    for i in range(len(true_pred)): 
        y_t = y_true[i]
        y_p = y_pred[i]
        if y_t==y_p==1:
           TN += 1
        if y_p==1 and y_t!=y_p:
           FN += 1
        if y_t==y_p==0:
           TP += 1
        if y_p==0 and y_t!=y_p:
           FP += 1

    return TP, FP, TN, FN

def calc_tprfpr(input):
    TP, FP, TN, FN = input
    if TP+FN == 0:
        tpr = 0
    else:
        tpr = TP/(TP+FN)
    if FP+TN == 0:
        fpr = 0
    else:
        fpr = FP/(FP+TN)

    if TN+FP == 0:
        tnr = 0
    else:
        tnr = TN/(TN+FP)
    if FN+TP == 0:
        fnr = 0
    else:
        fnr = FN/(FN+TP)

    return fpr, tpr, fnr, tnr

def calc_bdr (TPR, FPR):
    base_rate = np.array([0.01,0.1,1,10,25,50,75,90])/100
    base_rate_format = (np.array([0.01,0.1,1,10,25,50,75,90])/100)
    normal_rate = 1 - base_rate
    bdrs = []
    for b, n in zip (base_rate_format, normal_rate):
        if TPR ==  0.0:
            TPR_temp = 1.0
        else:
            TPR_temp = TPR
        bayesian_detection_rate = (b * TPR_temp) / ((b * TPR_temp) + (n * FPR))
        bdrs.append(pretty_print(bayesian_detection_rate * 100))
    return " | ".join(bdrs)

def get_file_values(scores_file, y_true_, cfad_provided=False):

    with open(scores_file, "r") as f:
        lines = f.readlines()

    y_true = []
    y_pred = []
    y_proba = []
    y_probb = []
    for i, line in enumerate(lines):
        name, _, pred, proba, probb = line.split(" ")
        if cfad_provided:
            y_true.append(int(y_true_[i]))           
        else:
            y_true.append(int(y_true_.get(name)))
        
        y_pred.append(int(pred))
        y_proba.append(float(proba.replace("[","").replace(",", "")))
        y_probb.append(float(probb.replace("]","").replace("\n", "")))
            

    return y_true, y_pred, y_proba, y_probb

def calc_prec_rec_acc_f1(scores_file, y_true_dict):
    
    with open(scores_file, "r") as f:
        lines = f.readlines()

    y_true = []
    y_pred = []
    for line in lines:
        try:
            name, _, pred, _, _ = line.split(" ")
        except:
            name, name1, _, pred, _, _ = line.split(" ")
            name = f"{name} {name1}"

        y_true.append(y_true_dict.get(name))
        y_pred.append(int(pred))    

    cr = classification_report(y_true, y_pred, output_dict=True)

    prec = cr.get("macro avg").get("precision")
    recall = cr.get("macro avg").get("recall")
    f1 = cr.get("macro avg").get("f1-score")
    acc = cr.get("accuracy")

    return prec, recall, acc, f1

def calc_eer(scores_file, y_true_, cfad_provided=False):
    y_true, _, y_proba, y_probb = get_file_values(scores_file, y_true_, cfad_provided)
    fps, tps, _ = _binary_clf_curve(y_true, y_probb, pos_label=1)
    fpr_s, tpr_s = roc_single(fps, tps)
    eer = brentq(lambda x : 1. - x - interp1d(fpr_s, tpr_s)(x), 0., 1.)    
    return eer

def pretty_print(x):
    
    x_out = str(round(x,1))
    x_out = x_out.ljust(4,'0')
    if x_out == "100.0": x_out = "100 "
    return x_out

def print_to_fileconsole(fname, string, write_append="a"):

    with open(fname, write_append) as f:
        f.write(string + "\n")

    print(string)

def table3(y_true_asv, y_true_cfad, scores_file_type, cfad_provided=False):
    
    fname = "figs/table3.txt"

    scores_file_asv_LL = f"../ASVspoof/LFCC-LCNN/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_asv_RN = f"../ASVspoof/RawNet2/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_asv_SW = f"../ASVspoof/wav2vec/results/90-10_asv21-eval_{scores_file_type}.scores" 
    scores_file_asv_LG = "../ASVspoof/LFCC-GMM/results/90-10_asv21-eval_provided.scores"
    scores_file_asv_CG = "../ASVspoof/CQCC-GMM/results/90-10_asv21-eval_provided.scores"

    scores_file_cfad_LL = f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores"
    scores_file_cfad_RN = f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores"

    if cfad_provided:
        y_true_cfad_LL = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_cfad_RN = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores")

        cfad_LL = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL, y_true_cfad_LL, cfad_provided))
        cfad_RN = calc_tprfpr(calc_tpfptnfn_(scores_file_cfad_RN, y_true_cfad_RN, cfad_provided))
        cfad_LL_eer = calc_eer(scores_file_cfad_LL, y_true_cfad_LL, cfad_provided)
        cfad_RN_eer = calc_eer_(scores_file_cfad_RN, y_true_cfad_RN, cfad_provided)
    else:
        cfad_LL = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL, y_true_cfad))
        cfad_RN = calc_tprfpr(calc_tpfptnfn_(scores_file_cfad_RN, y_true_cfad))
        cfad_LL_eer = calc_eer(scores_file_cfad_LL, y_true_cfad_LL)
        cfad_RN_eer = calc_eer_(scores_file_cfad_RN, y_true_cfad_RN)

    asv_LL = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LL, y_true_asv))
    asv_RN = calc_tprfpr(calc_tpfptnfn(scores_file_asv_RN, y_true_asv))
    asv_SW = calc_tprfpr(calc_tpfptnfn(scores_file_asv_SW, y_true_asv))
    asv_LG = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LG, y_true_asv))
    asv_CG = calc_tprfpr(calc_tpfptnfn(scores_file_asv_CG, y_true_asv))

    asv_LL_eer = calc_eer(scores_file_asv_LL, y_true_asv)
    asv_RN_eer = calc_eer(scores_file_asv_RN, y_true_asv)
    asv_SW_eer = calc_eer(scores_file_asv_SW, y_true_asv)
    asv_LG_eer = calc_eer(scores_file_asv_LG, y_true_asv)
    asv_CG_eer = calc_eer(scores_file_asv_CG, y_true_asv)

    print_to_fileconsole(fname,f"M_asv-cg  | M | {asv_CG_eer:.3f} | {asv_CG[1]:.3f} | {asv_CG[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.253 |  --   |  --")
    print_to_fileconsole(fname,f"M_asv-lg  | M | {asv_LG_eer:.3f} | {asv_LG[1]:.3f} | {asv_LG[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.256 |  --   |  --")
    print_to_fileconsole(fname,f"M_asv-ll  | M | {asv_LL_eer:.3f} | {asv_LL[1]:.3f} | {asv_LL[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.235 |  --   |  --")
    print_to_fileconsole(fname,f"M_asv-rn  | M | {asv_RN_eer:.3f} | {asv_RN[1]:.3f} | {asv_RN[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.224 |  --   |  --")
    print_to_fileconsole(fname,f"M_asv-sw  | M | {asv_SW_eer:.3f} | {asv_SW[1]:.3f} | {asv_SW[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.029 |  --   |  --")
    print_to_fileconsole(fname,f"M_cfad-ll | M | {cfad_LL_eer:.3f} | {cfad_LL[1]:.3f} | {cfad_LL[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.097 |  --   |  --")
    print_to_fileconsole(fname,f"M_cfad-rn | M | {cfad_RN_eer:.3f} | {cfad_RN[1]:.3f} | {cfad_RN[0]:.3f}")
    print_to_fileconsole(fname,f"          | R | 0.239 |  --   |  --")

    return

def table4(y_true_asv, y_true_asvRO, y_true_cfad, y_true_cfadRO, scores_file_type, cfad_provided=False): #appendix

    fname = "figs/table4.txt"
    if scores_file_type == "provided": cfad_provided = True

    if cfad_provided:
        y_true_cfad_LL = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_cfad_RN = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores")

        y_true_cfadRO_LL = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_cfadRO_RN = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores")
    else:
        y_true_cfad_LL = y_true_cfad
        y_true_cfad_RN = y_true_cfad
        y_true_cfadRO_LL = y_true_cfadRO
        y_true_cfadRO_RN = y_true_cfadRO


    # region RAWNET2 ASVSPOOF
    # ########## 
    scores_file_asv_RN9010 = f"../ASVspoof/RawNet2/results/90-10_asv21-eval_{scores_file_type}.scores"
    asv_RN9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_RN9010, y_true_asv))
    asv_RN9010_eer = calc_eer(scores_file_asv_RN9010, y_true_asv)
    asv_RN9010_bdr = calc_bdr(asv_RN9010[1], asv_RN9010[0])    

    scores_file_asvRO_RN9010 = f"../ASVspoof/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores"
    asvRO_RN9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_RN9010, y_true_asvRO))
    asvRO_RN9010_bdr = calc_bdr(asvRO_RN9010[1], asvRO_RN9010[0])



    scores_file_asv_RN7225 = f"../ASVspoof/RawNet2/results/75-25_asv21-eval_{scores_file_type}.scores"
    asv_RN7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_RN7225, y_true_asv))
    asv_RN7525_eer = calc_eer(scores_file_asv_RN7225, y_true_asv)
    asv_RN7525_bdr = calc_bdr(asv_RN7525[1], asv_RN7525[0])    

    scores_file_asvRO_RN7525 = f"../ASVspoof/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores"
    asvRO_RN7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_RN7525, y_true_asvRO))
    asvRO_RN7525_bdr = calc_bdr(asvRO_RN7525[1], asvRO_RN7525[0])



    scores_file_asv_RN7225 = f"../ASVspoof/RawNet2/results/50-50_asv21-eval_{scores_file_type}.scores"
    asv_RN5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_RN7225, y_true_asv))
    asv_RN5050_eer = calc_eer(scores_file_asv_RN7225, y_true_asv)
    asv_RN5050_bdr = calc_bdr(asv_RN5050[1], asv_RN5050[0])    

    scores_file_asvRO_RN5050 = f"../ASVspoof/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores"
    asvRO_RN5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_RN5050, y_true_asvRO))
    asvRO_RN5050_bdr = calc_bdr(asvRO_RN5050[1], asvRO_RN5050[0])



    scores_file_asv_RN7225 = f"../ASVspoof/RawNet2/results/25-75_asv21-eval_{scores_file_type}.scores"
    asv_RN2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_RN7225, y_true_asv))
    asv_RN2575_eer = calc_eer(scores_file_asv_RN7225, y_true_asv)
    asv_RN2575_bdr = calc_bdr(asv_RN2575[1], asv_RN2575[0])    

    scores_file_asvRO_RN2575 = f"../ASVspoof/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores"
    asvRO_RN2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_RN2575, y_true_asvRO))
    asvRO_RN2575_bdr = calc_bdr(asvRO_RN2575[1], asvRO_RN2575[0])


    print_to_fileconsole(fname,f"RawNet2     | 9010 | ASV  | {pretty_print(asv_RN9010_eer * 100)} | {pretty_print(asv_RN9010[1] * 100)} | {pretty_print(asv_RN9010[0] * 100)} | {pretty_print(asv_RN9010[3] * 100)} | {pretty_print(asv_RN9010[2] * 100)} | {asv_RN9010_bdr}", write_append="w")
    print_to_fileconsole(fname,f"RawNet2     | 9010 | RO   | 0.00 | ---- | {pretty_print(asvRO_RN9010[0] * 100)} | {pretty_print(asvRO_RN9010[3] * 100)} | ---- | {asvRO_RN9010_bdr}")

    
    print_to_fileconsole(fname,f"RawNet2     | 7525 | ASV  | {pretty_print(asv_RN7525_eer * 100)} | {pretty_print(asv_RN7525[1] * 100)} | {pretty_print(asv_RN7525[0] * 100)} | {pretty_print(asv_RN7525[3] * 100)} | {pretty_print(asv_RN7525[2] * 100)} | {asv_RN7525_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 7525 | RO   | 0.00 | ---- | {pretty_print(asvRO_RN7525[0] * 100)} | {pretty_print(asvRO_RN7525[3] * 100)} | ---- | {asvRO_RN7525_bdr}")


    print_to_fileconsole(fname,f"RawNet2     | 5050 | ASV  | {pretty_print(asv_RN5050_eer * 100)} | {pretty_print(asv_RN5050[1] * 100)} | {pretty_print(asv_RN5050[0] * 100)} | {pretty_print(asv_RN5050[3] * 100)} | {pretty_print(asv_RN5050[2] * 100)} | {asv_RN5050_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 5050 | RO   | 0.00 | ---- | {pretty_print(asvRO_RN5050[0] * 100)} | {pretty_print(asvRO_RN5050[3] * 100)} | ---- | {asvRO_RN5050_bdr}")


    print_to_fileconsole(fname,f"RawNet2     | 2575 | ASV  | {pretty_print(asv_RN2575_eer * 100)} | {pretty_print(asv_RN2575[1] * 100)} | {pretty_print(asv_RN2575[0] * 100)} | {pretty_print(asv_RN2575[3] * 100)} | {pretty_print(asv_RN2575[2] * 100)} | {asv_RN2575_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 2575 | RO   | 0.00 | ---- | {pretty_print(asvRO_RN2575[0] * 100)} | {pretty_print(asvRO_RN2575[3] * 100)} | ---- | {asvRO_RN2575_bdr}")

    # endregion


    # region LFCC-LCNN ASVSPOOF
    # ########## 
    scores_file_asv_LL9010 = f"../ASVspoof/LFCC-LCNN/results/90-10_asv21-eval_{scores_file_type}.scores"
    asv_LL9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LL9010, y_true_asv))
    asv_LL9010_eer = calc_eer(scores_file_asv_LL9010, y_true_asv)
    asv_LL9010_bdr = calc_bdr(asv_LL9010[1], asv_LL9010[0])    

    scores_file_asvRO_LL9010 = f"../ASVspoof/LFCC-LCNN/results/90-10_ro-eval_{scores_file_type}.scores"
    asvRO_LL9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_LL9010, y_true_asvRO))
    asvRO_LL9010_bdr = calc_bdr(asvRO_LL9010[1], asvRO_LL9010[0])



    scores_file_asv_LL7225 = f"../ASVspoof/LFCC-LCNN/results/75-25_asv21-eval_{scores_file_type}.scores"
    asv_LL7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LL7225, y_true_asv))
    asv_LL7525_eer = calc_eer(scores_file_asv_LL7225, y_true_asv)
    asv_LL7525_bdr = calc_bdr(asv_LL7525[1], asv_LL7525[0])    

    scores_file_asvRO_LL7525 = f"../ASVspoof/LFCC-LCNN/results/75-25_ro-eval_{scores_file_type}.scores"
    asvRO_LL7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_LL7525, y_true_asvRO))
    asvRO_LL7525_bdr = calc_bdr(asvRO_LL7525[1], asvRO_LL7525[0])



    scores_file_asv_LL7225 = f"../ASVspoof/LFCC-LCNN/results/50-50_asv21-eval_{scores_file_type}.scores"
    asv_LL5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LL7225, y_true_asv))
    asv_LL5050_eer = calc_eer(scores_file_asv_LL7225, y_true_asv)
    asv_LL5050_bdr = calc_bdr(asv_LL5050[1], asv_LL5050[0])    

    scores_file_asvRO_LL5050 = f"../ASVspoof/LFCC-LCNN/results/50-50_ro-eval_{scores_file_type}.scores"
    asvRO_LL5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_LL5050, y_true_asvRO))
    asvRO_LL5050_bdr = calc_bdr(asvRO_LL5050[1], asvRO_LL5050[0])



    scores_file_asv_LL7225 = f"../ASVspoof/LFCC-LCNN/results/25-75_asv21-eval_{scores_file_type}.scores"
    asv_LL2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LL7225, y_true_asv))
    asv_LL2575_eer = calc_eer(scores_file_asv_LL7225, y_true_asv)
    asv_LL2575_bdr = calc_bdr(asv_LL2575[1], asv_LL2575[0])    

    scores_file_asvRO_LL2575 = f"../ASVspoof/LFCC-LCNN/results/25-75_ro-eval_{scores_file_type}.scores"
    asvRO_LL2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_LL2575, y_true_asvRO))
    asvRO_LL2575_bdr = calc_bdr(asvRO_LL2575[1], asvRO_LL2575[0])


    print_to_fileconsole(fname,f"LFCC-LCNN   | 9010 | ASV  | {pretty_print(asv_LL9010_eer * 100)} | {pretty_print(asv_LL9010[1] * 100)} | {pretty_print(asv_LL9010[0] * 100)} | {pretty_print(asv_LL9010[3] * 100)} | {pretty_print(asv_LL9010[2] * 100)} | {asv_LL9010_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 9010 | RO   | 0.00 | ---- | {pretty_print(asvRO_LL9010[0] * 100)} | {pretty_print(asvRO_LL9010[3] * 100)} | ---- | {asvRO_LL9010_bdr}")

    
    print_to_fileconsole(fname,f"LFCC-LCNN   | 7525 | ASV  | {pretty_print(asv_LL7525_eer * 100)} | {pretty_print(asv_LL7525[1] * 100)} | {pretty_print(asv_LL7525[0] * 100)} | {pretty_print(asv_LL7525[3] * 100)} | {pretty_print(asv_LL7525[2] * 100)} | {asv_LL7525_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 7525 | RO   | 0.00 | ---- | {pretty_print(asvRO_LL7525[0] * 100)} | {pretty_print(asvRO_LL7525[3] * 100)} | ---- | {asvRO_LL7525_bdr}")


    print_to_fileconsole(fname,f"LFCC-LCNN   | 5050 | ASV  | {pretty_print(asv_LL5050_eer * 100)} | {pretty_print(asv_LL5050[1] * 100)} | {pretty_print(asv_LL5050[0] * 100)} | {pretty_print(asv_LL5050[3] * 100)} | {pretty_print(asv_LL5050[2] * 100)} | {asv_LL5050_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 5050 | RO   | 0.00 | ---- | {pretty_print(asvRO_LL5050[0] * 100)} | {pretty_print(asvRO_LL5050[3] * 100)} | ---- | {asvRO_LL5050_bdr}")


    print_to_fileconsole(fname,f"LFCC-LCNN   | 2575 | ASV  | {pretty_print(asv_LL2575_eer * 100)} | {pretty_print(asv_LL2575[1] * 100)} | {pretty_print(asv_LL2575[0] * 100)} | {pretty_print(asv_LL2575[3] * 100)} | {pretty_print(asv_LL2575[2] * 100)} | {asv_LL2575_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 2575 | RO   | 0.00 | ---- | {pretty_print(asvRO_LL2575[0] * 100)} | {pretty_print(asvRO_LL2575[3] * 100)} | ---- | {asvRO_LL2575_bdr}")

    # endregion


    # region wav2vec ASVSPOOF
    # ########## 
    scores_file_asv_SW9010 = f"../ASVspoof/wav2vec/results/90-10_asv21-eval_{scores_file_type}.scores"
    asv_SW9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_SW9010, y_true_asv))
    asv_SW9010_eer = calc_eer(scores_file_asv_SW9010, y_true_asv)
    asv_SW9010_bdr = calc_bdr(asv_SW9010[1], asv_SW9010[0])    

    scores_file_asvRO_SW9010 = f"../ASVspoof/wav2vec/results/90-10_ro-eval_{scores_file_type}.scores"
    asvRO_SW9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_SW9010, y_true_asvRO))
    asvRO_SW9010_bdr = calc_bdr(asvRO_SW9010[1], asvRO_SW9010[0])



    scores_file_asv_SW7225 = f"../ASVspoof/wav2vec/results/75-25_asv21-eval_{scores_file_type}.scores"
    asv_SW7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_SW7225, y_true_asv))
    asv_SW7525_eer = calc_eer(scores_file_asv_SW7225, y_true_asv)
    asv_SW7525_bdr = calc_bdr(asv_SW7525[1], asv_SW7525[0])    

    scores_file_asvRO_SW7525 = f"../ASVspoof/wav2vec/results/75-25_ro-eval_{scores_file_type}.scores"
    asvRO_SW7525 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_SW7525, y_true_asvRO))
    asvRO_SW7525_bdr = calc_bdr(asvRO_SW7525[1], asvRO_SW7525[0])



    scores_file_asv_SW7225 = f"../ASVspoof/wav2vec/results/50-50_asv21-eval_{scores_file_type}.scores"
    asv_SW5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_SW7225, y_true_asv))
    asv_SW5050_eer = calc_eer(scores_file_asv_SW7225, y_true_asv)
    asv_SW5050_bdr = calc_bdr(asv_SW5050[1], asv_SW5050[0])    

    scores_file_asvRO_SW5050 = f"../ASVspoof/wav2vec/results/50-50_ro-eval_{scores_file_type}.scores"
    asvRO_SW5050 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_SW5050, y_true_asvRO))
    asvRO_SW5050_bdr = calc_bdr(asvRO_SW5050[1], asvRO_SW5050[0])



    scores_file_asv_SW7225 = f"../ASVspoof/wav2vec/results/25-75_asv21-eval_{scores_file_type}.scores"
    asv_SW2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_SW7225, y_true_asv))
    asv_SW2575_eer = calc_eer(scores_file_asv_SW7225, y_true_asv)
    asv_SW2575_bdr = calc_bdr(asv_SW2575[1], asv_SW2575[0])    

    scores_file_asvRO_SW2575 = f"../ASVspoof/wav2vec/results/25-75_ro-eval_{scores_file_type}.scores"
    asvRO_SW2575 = calc_tprfpr(calc_tpfptnfn(scores_file_asvRO_SW2575, y_true_asvRO))
    asvRO_SW2575_bdr = calc_bdr(asvRO_SW2575[1], asvRO_SW2575[0])


    print_to_fileconsole(fname,f"wav2vec     | 9010 | ASV  | {pretty_print(asv_SW9010_eer * 100)} | {pretty_print(asv_SW9010[1] * 100)} | {pretty_print(asv_SW9010[0] * 100)} | {pretty_print(asv_SW9010[3] * 100)} | {pretty_print(asv_SW9010[2] * 100)} | {asv_SW9010_bdr}")
    print_to_fileconsole(fname,f"wav2vec     | 9010 | RO   | 0.00 | ---- | {pretty_print(asvRO_SW9010[0] * 100)} | {pretty_print(asvRO_SW9010[3] * 100)} | ---- | {asvRO_SW9010_bdr}")

    
    print_to_fileconsole(fname,f"wav2vec     | 7525 | ASV  | {pretty_print(asv_SW7525_eer * 100)} | {pretty_print(asv_SW7525[1] * 100)} | {pretty_print(asv_SW7525[0] * 100)} | {pretty_print(asv_SW7525[3] * 100)} | {pretty_print(asv_SW7525[2] * 100)} | {asv_SW7525_bdr}")
    print_to_fileconsole(fname,f"wav2vec     | 7525 | RO   | 0.00 | ---- | {pretty_print(asvRO_SW7525[0] * 100)} | {pretty_print(asvRO_SW7525[3] * 100)} | ---- | {asvRO_SW7525_bdr}")


    print_to_fileconsole(fname,f"wav2vec     | 5050 | ASV  | {pretty_print(asv_SW5050_eer * 100)} | {pretty_print(asv_SW5050[1] * 100)} | {pretty_print(asv_SW5050[0] * 100)} | {pretty_print(asv_SW5050[3] * 100)} | {pretty_print(asv_SW5050[2] * 100)} | {asv_SW5050_bdr}")
    print_to_fileconsole(fname,f"wav2vec     | 5050 | RO   | 0.00 | ---- | {pretty_print(asvRO_SW5050[0] * 100)} | {pretty_print(asvRO_SW5050[3] * 100)} | ---- | {asvRO_SW5050_bdr}")


    print_to_fileconsole(fname,f"wav2vec     | 2575 | ASV  | {pretty_print(asv_SW2575_eer * 100)} | {pretty_print(asv_SW2575[1] * 100)} | {pretty_print(asv_SW2575[0] * 100)} | {pretty_print(asv_SW2575[3] * 100)} | {pretty_print(asv_SW2575[2] * 100)} | {asv_SW2575_bdr}")
    print_to_fileconsole(fname,f"wav2vec     | 2575 | RO   | 0.00 | ---- | {pretty_print(asvRO_SW2575[0] * 100)} | {pretty_print(asvRO_SW2575[3] * 100)} | ---- | {asvRO_SW2575_bdr}")

    # endregion


    # region CQCC-GMM ASVSPOOF
    # ########## 
    scores_file_asv_CG9010 = f"../ASVspoof/CQCC-GMM/results/90-10_asv21-eval_provided.scores"
    asv_CG9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_CG9010, y_true_asv))
    asv_CG9010_eer = calc_eer(scores_file_asv_CG9010, y_true_asv)
    asv_CG9010_bdr = calc_bdr(asv_CG9010[1], asv_CG9010[0])    

    print_to_fileconsole(fname,f"CQCC-GMM    | 9010 | ASV  | {pretty_print(asv_CG9010_eer * 100)} | {pretty_print(asv_CG9010[1] * 100)} | {pretty_print(asv_CG9010[0] * 100)} | {pretty_print(asv_CG9010[3] * 100)} | {pretty_print(asv_CG9010[2] * 100)} | {asv_CG9010_bdr}")

    # endregion


    # region LFCC-GMM ASVSPOOF
    # ########## 
    scores_file_asv_LG9010 = f"../ASVspoof/LFCC-GMM/results/90-10_asv21-eval_provided.scores"
    asv_LG9010 = calc_tprfpr(calc_tpfptnfn(scores_file_asv_LG9010, y_true_asv))
    asv_LG9010_eer = calc_eer(scores_file_asv_LG9010, y_true_asv)
    asv_LG9010_bdr = calc_bdr(asv_LG9010[1], asv_LG9010[0])    

    print_to_fileconsole(fname,f"LFCC-GMM    | 9010 | ASV  | {pretty_print(asv_LG9010_eer * 100)} | {pretty_print(asv_LG9010[1] * 100)} | {pretty_print(asv_LG9010[0] * 100)} | {pretty_print(asv_LG9010[3] * 100)} | {pretty_print(asv_LG9010[2] * 100)} | {asv_LG9010_bdr}")

    # endregion


    # region RAWNET2 CFAD
    # ########## 
    scores_file_cfad_RN9010 = f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores"
    cfad_RN9010 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_RN9010, y_true_cfad_RN, cfad_provided))
    cfad_RN9010_eer = calc_eer(scores_file_cfad_RN9010, y_true_cfad_RN, cfad_provided)
    cfad_RN9010_bdr = calc_bdr(cfad_RN9010[1], cfad_RN9010[0])    

    scores_file_cfadRO_RN9010 = f"../CFAD/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores"
    cfadRO_RN9010 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_RN9010, y_true_cfadRO_RN, cfad_provided))
    cfadRO_RN9010_bdr = calc_bdr(cfadRO_RN9010[1], cfadRO_RN9010[0])



    scores_file_cfad_RN7525 = f"../CFAD/RawNet2/results/75-25_cfad-eval_{scores_file_type}.scores"
    cfad_RN7525 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_RN7525, y_true_cfad_RN, cfad_provided))
    cfad_RN7525_eer = calc_eer(scores_file_cfad_RN7525, y_true_cfad_RN, cfad_provided)
    cfad_RN7525_bdr = calc_bdr(cfad_RN7525[1], cfad_RN7525[0])    

    scores_file_cfadRO_RN7525 = f"../CFAD/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores"
    cfadRO_RN7525 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_RN7525, y_true_cfadRO_RN, cfad_provided))
    cfadRO_RN7525_bdr = calc_bdr(cfadRO_RN7525[1], cfadRO_RN7525[0])



    scores_file_cfad_RN5050 = f"../CFAD/RawNet2/results/50-50_cfad-eval_{scores_file_type}.scores"
    cfad_RN5050 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_RN5050, y_true_cfad_RN, cfad_provided))
    cfad_RN5050_eer = calc_eer(scores_file_cfad_RN5050, y_true_cfad_RN, cfad_provided)
    cfad_RN5050_bdr = calc_bdr(cfad_RN5050[1], cfad_RN5050[0])    

    scores_file_cfadRO_RN5050 = f"../CFAD/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores"
    cfadRO_RN5050 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_RN5050, y_true_cfadRO_RN, cfad_provided))
    cfadRO_RN5050_bdr = calc_bdr(cfadRO_RN5050[1], cfadRO_RN5050[0])



    scores_file_cfad_RN2575 = f"../CFAD/RawNet2/results/25-75_cfad-eval_{scores_file_type}.scores"
    cfad_RN2575 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_RN2575, y_true_cfad_RN, cfad_provided))
    cfad_RN2575_eer = calc_eer(scores_file_cfad_RN2575, y_true_cfad_RN, cfad_provided)
    cfad_RN2575_bdr = calc_bdr(cfad_RN2575[1], cfad_RN2575[0])    

    scores_file_cfadRO_RN2575 = f"../CFAD/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores"
    cfadRO_RN2575 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_RN2575, y_true_cfadRO_RN, cfad_provided))
    cfadRO_RN2575_bdr = calc_bdr(cfadRO_RN2575[1], cfadRO_RN2575[0])


    print_to_fileconsole(fname,f"RawNet2     | 9010 | CFAD | {pretty_print(cfad_RN9010_eer * 100)} | {pretty_print(cfad_RN9010[1] * 100)} | {pretty_print(cfad_RN9010[0] * 100)} | {pretty_print(cfad_RN9010[3] * 100)} | {pretty_print(cfad_RN9010[2] * 100)} | {cfad_RN9010_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 9010 | RO   | 0.00 | ---- | {pretty_print(cfadRO_RN9010[0] * 100)} | {pretty_print(cfadRO_RN9010[3] * 100)} | ---- | {cfadRO_RN9010_bdr}")

    
    print_to_fileconsole(fname,f"RawNet2     | 7525 | CFAD | {pretty_print(cfad_RN7525_eer * 100)} | {pretty_print(cfad_RN7525[1] * 100)} | {pretty_print(cfad_RN7525[0] * 100)} | {pretty_print(cfad_RN7525[3] * 100)} | {pretty_print(cfad_RN7525[2] * 100)} | {cfad_RN7525_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 7525 | RO   | 0.00 | ---- | {pretty_print(cfadRO_RN7525[0] * 100)} | {pretty_print(cfadRO_RN7525[3] * 100)} | ---- | {cfadRO_RN7525_bdr}")


    print_to_fileconsole(fname,f"RawNet2     | 5050 | CFAD | {pretty_print(cfad_RN5050_eer * 100)} | {pretty_print(cfad_RN5050[1] * 100)} | {pretty_print(cfad_RN5050[0] * 100)} | {pretty_print(cfad_RN5050[3] * 100)} | {pretty_print(cfad_RN5050[2] * 100)} | {cfad_RN5050_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 5050 | RO   | 0.00 | ---- | {pretty_print(cfadRO_RN5050[0] * 100)} | {pretty_print(cfadRO_RN5050[3] * 100)} | ---- | {cfadRO_RN5050_bdr}")


    print_to_fileconsole(fname,f"RawNet2     | 2575 | CFAD | {pretty_print(cfad_RN2575_eer * 100)} | {pretty_print(cfad_RN2575[1] * 100)} | {pretty_print(cfad_RN2575[0] * 100)} | {pretty_print(cfad_RN2575[3] * 100)} | {pretty_print(cfad_RN2575[2] * 100)} | {cfad_RN2575_bdr}")
    print_to_fileconsole(fname,f"RawNet2     | 2575 | RO   | 0.00 | ---- | {pretty_print(cfadRO_RN2575[0] * 100)} | {pretty_print(cfadRO_RN2575[3] * 100)} | ---- | {cfadRO_RN2575_bdr}")

    # endregion


    # region LFCC-LCNN CFAD
    # ########## 
    scores_file_cfad_LL9010 = f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores"
    cfad_LL9010 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL9010, y_true_cfad_LL, cfad_provided))
    cfad_LL9010_eer = calc_eer(scores_file_cfad_LL9010, y_true_cfad_LL, cfad_provided)
    cfad_LL9010_bdr = calc_bdr(cfad_LL9010[1], cfad_LL9010[0])    

    scores_file_cfadRO_LL9010 = f"../CFAD/LFCC-LCNN/results/90-10_ro-eval_{scores_file_type}.scores"
    cfadRO_LL9010 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_LL9010, y_true_cfadRO_LL, cfad_provided))
    cfadRO_LL9010_bdr = calc_bdr(cfadRO_LL9010[1], cfadRO_LL9010[0])



    scores_file_cfad_LL7525 = f"../CFAD/LFCC-LCNN/results/75-25_cfad-eval_{scores_file_type}.scores"
    cfad_LL7525 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL7525, y_true_cfad_LL, cfad_provided))
    cfad_LL7525_eer = calc_eer(scores_file_cfad_LL7525, y_true_cfad_LL, cfad_provided)
    cfad_LL7525_bdr = calc_bdr(cfad_LL7525[1], cfad_LL7525[0])    

    scores_file_cfadRO_LL7525 = f"../CFAD/LFCC-LCNN/results/75-25_ro-eval_{scores_file_type}.scores"
    cfadRO_LL7525 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_LL7525, y_true_cfadRO_LL, cfad_provided))
    cfadRO_LL7525_bdr = calc_bdr(cfadRO_LL7525[1], cfadRO_LL7525[0])



    scores_file_cfad_LL5050 = f"../CFAD/LFCC-LCNN/results/50-50_cfad-eval_{scores_file_type}.scores"
    cfad_LL5050 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL5050, y_true_cfad_LL, cfad_provided))
    cfad_LL5050_eer = calc_eer(scores_file_cfad_LL5050, y_true_cfad_LL, cfad_provided)
    cfad_LL5050_bdr = calc_bdr(cfad_LL5050[1], cfad_LL5050[0])    

    scores_file_cfadRO_LL5050 = f"../CFAD/LFCC-LCNN/results/50-50_ro-eval_{scores_file_type}.scores"
    cfadRO_LL5050 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_LL5050, y_true_cfadRO_LL, cfad_provided))
    cfadRO_LL5050_bdr = calc_bdr(cfadRO_LL5050[1], cfadRO_LL5050[0])



    scores_file_cfad_LL2575 = f"../CFAD/LFCC-LCNN/results/25-75_cfad-eval_{scores_file_type}.scores"
    cfad_LL2575 = calc_tprfpr(calc_tpfptnfn(scores_file_cfad_LL2575, y_true_cfad_LL, cfad_provided))
    cfad_LL2575_eer = calc_eer(scores_file_cfad_LL2575, y_true_cfad_LL, cfad_provided)
    cfad_LL2575_bdr = calc_bdr(cfad_LL2575[1], cfad_LL2575[0])    

    scores_file_cfadRO_LL2575 = f"../CFAD/LFCC-LCNN/results/25-75_ro-eval_{scores_file_type}.scores"
    cfadRO_LL2575 = calc_tprfpr(calc_tpfptnfn(scores_file_cfadRO_LL2575, y_true_cfadRO_LL, cfad_provided))
    cfadRO_LL2575_bdr = calc_bdr(cfadRO_LL2575[1], cfadRO_LL2575[0])


    print_to_fileconsole(fname,f"LFCC-LCNN   | 9010 | CFAD | {pretty_print(cfad_LL9010_eer * 100)} | {pretty_print(cfad_LL9010[1] * 100)} | {pretty_print(cfad_LL9010[0] * 100)} | {pretty_print(cfad_LL9010[3] * 100)} | {pretty_print(cfad_LL9010[2] * 100)} | {cfad_LL9010_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 9010 | RO   | 0.00 | ---- | {pretty_print(cfadRO_LL9010[0] * 100)} | {pretty_print(cfadRO_LL9010[3] * 100)} | ---- | {cfadRO_LL9010_bdr}")

    
    print_to_fileconsole(fname,f"LFCC-LCNN   | 7525 | CFAD | {pretty_print(cfad_LL7525_eer * 100)} | {pretty_print(cfad_LL7525[1] * 100)} | {pretty_print(cfad_LL7525[0] * 100)} | {pretty_print(cfad_LL7525[3] * 100)} | {pretty_print(cfad_LL7525[2] * 100)} | {cfad_LL7525_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 7525 | RO   | 0.00 | ---- | {pretty_print(cfadRO_LL7525[0] * 100)} | {pretty_print(cfadRO_LL7525[3] * 100)} | ---- | {cfadRO_LL7525_bdr}")


    print_to_fileconsole(fname,f"LFCC-LCNN   | 5050 | CFAD | {pretty_print(cfad_LL5050_eer * 100)} | {pretty_print(cfad_LL5050[1] * 100)} | {pretty_print(cfad_LL5050[0] * 100)} | {pretty_print(cfad_LL5050[3] * 100)} | {pretty_print(cfad_LL5050[2] * 100)} | {cfad_LL5050_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 5050 | RO   | 0.00 | ---- | {pretty_print(cfadRO_LL5050[0] * 100)} | {pretty_print(cfadRO_LL5050[3] * 100)} | ---- | {cfadRO_LL5050_bdr}")


    print_to_fileconsole(fname,f"LFCC-LCNN   | 2575 | CFAD | {pretty_print(cfad_LL2575_eer * 100)} | {pretty_print(cfad_LL2575[1] * 100)} | {pretty_print(cfad_LL2575[0] * 100)} | {pretty_print(cfad_LL2575[3] * 100)} | {pretty_print(cfad_LL2575[2] * 100)} | {cfad_LL2575_bdr}")
    print_to_fileconsole(fname,f"LFCC-LCNN   | 2575 | RO   | 0.00 | ---- | {pretty_print(cfadRO_LL2575[0] * 100)} | {pretty_print(cfadRO_LL2575[3] * 100)} | ---- | {cfadRO_LL2575_bdr}")

    # endregion

    return

def table5(y_true_cifake, scores_file_type):
    fname = "figs/table5.txt"
    scores_file_cifake_5050 = f"../CIFAKE/CIFAKE/results/50-50_cifake-eval_{scores_file_type}.scores"
    cifake_5050 = calc_prec_rec_acc_f1(scores_file_cifake_5050, y_true_cifake)

    print_to_fileconsole(fname,f"M_cifake  | M | {pretty_print(cifake_5050[2] * 100)} | {pretty_print(cifake_5050[0] * 100)} | {pretty_print(cifake_5050[1] * 100)} | {pretty_print(cifake_5050[3] * 100)}", "w")
    print_to_fileconsole(fname,f"          | R | 93.6 | 92.5 | 94.8 | 93.6")

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--table', type=str, default="paper")
    parser.add_argument('--which_scores_file', type=str, default="provided")
    args = parser.parse_args()

    y_true_asv = get_y_true("../ASVspoof/ASVspoofEval/protocols/asv21_eval.txt")
    y_true_asvro = get_y_true("../ASVspoof/RealOnlyEval/protocols/ro_eval.txt")

    y_true_cfad = get_y_true("../CFAD/CFADEval/protocols/cfad_eval.txt")
    y_true_cfadro = get_y_true("../CFAD/RealOnlyEval/protocols/ro_eval.txt")

    y_true_cifake = get_y_true("../CIFAKE/CIFAKEEval/protocols/cifake_eval.txt")
    y_true_cifakero = get_y_true("../CIFAKE/RealOnlyEval/protocols/ro_eval.txt")

    scores_file_type = ""
    cfad_provided = False
    if args.which_scores_file == "provided":
        scores_file_type = args.which_scores_file
        cfad_provided = True
    elif args.which_scores_file == "pretrained":
        scores_file_type = "p"

    table3(y_true_asv, y_true_cfad, scores_file_type, cfad_provided)

    if args.table == "all":
        table4(y_true_asv, y_true_asvro, y_true_cfad, y_true_cfadro, scores_file_type, cfad_provided)
        table5(y_true_cifake, scores_file_type)