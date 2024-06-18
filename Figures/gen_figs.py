

import argparse
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm as mcm, patches
from sklearn.metrics import confusion_matrix
import os
import pickle as pkl

def plot_bars(stats, type, fig_num, second_id, scores_file_type):
        _, ax = plt.subplots(2,1,dpi=150,figsize=(4,4.5),gridspec_kw={'height_ratios':[0.001, 1]})
        fancy_names = [r'$_{90F/10R}$', r'$_{75F/25R}$', r'$_{50F/50R}$', r'$_{25F/75R}$']
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        for i, stat in enumerate(stats):
            TP = stat[0]
            FP = stat[1]
            TN = stat[2]
            FN = stat[3]

            total = TP + FP + TN + FN

            tp = (TP/total) * 100
            fp = (FP/total) * 100
            tn = (TN/total) * 100
            fn = (FN/total) * 100

            y1.append(tp)
            y2.append(fp)
            y3.append(tn)
            y4.append(fn)

        cm_subsection = np.arange(0, 4, 1)
        colors = [ mcm.tab20(x) for x in cm_subsection ]

        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = np.array(y4)
        
        if "RO" not in type:
            ax[1].bar(fancy_names, y1, color=colors[0], width=0.4, label="TP")
            ax[1].bar(fancy_names, y2, bottom=y1, color=colors[1], width=0.4, label="FP")
            ax[1].bar(fancy_names, y3, bottom=y1+y2, color=colors[2], width=0.4, label="TN")
            ax[1].bar(fancy_names, y4, bottom=y1+y2+y3, color=colors[3], width=0.4, label="FN")
        else:
            ax[1].bar(fancy_names, y2, color=colors[1], width=0.4, label="FP")
            ax[1].bar(fancy_names, y3, bottom=y2, color=colors[2], width=0.4, label="TN")

        h, l = ax[1].get_legend_handles_labels()
        ax[0].legend(h, l, ncol=4, loc="upper center", fontsize = 11)
        ax[1].tick_params(axis='x', which='major', labelsize=18)
        ax[1].tick_params(axis='x', which='minor', labelsize=18)
        ax[0].axis("off")

        ax[1].set_ylabel("Percentage of Total Samples", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"figs/figure{fig_num}-{second_id}-{scores_file_type}.png")

        return

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
        try:
            name, _, pred, _, _ = line.split(" ")
        except:
            name, name1, _, pred, _, _ = line.split(" ")
            name = f"{name} {name1}"
    
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

    return fpr, tpr

def figure1():
    #ids dataset split into sub-categories for %malicious sample calcuations
    darpa1998 = [5.04,0.01]
    tuids = [56.25,44.48 ,58.83]
    trabid = [0.05,0.01,9.8,2.1,0.01,0.01,0.01,2.6,2.7,2.3]
    unsw_nb15 =[1.1,0.12,0.10,0.74,2.0,8.9,0.62,.07,0.01]
    csecicids2018 = [17.8]
    AWID = [9]
    botnet = [42]
    CIDDS_002 = [3.4]
    empirical_botnet = [0.89,0.85,0.49,0.15,0.46,0.22,0.06,0.10,5.02,6.24,67.97,67.97,1.67]
    DDoS = [33]
    protocol_profiles = [2.8]
    kyoto2006 = [46]
    PUIDS = [53]
    PUF = [12]
    SSENet_2014 = [71,50]

    ids = darpa1998 + tuids + trabid + unsw_nb15 + csecicids2018 + AWID + botnet + CIDDS_002 + empirical_botnet + DDoS + protocol_profiles + kyoto2006 + PUIDS + PUF + SSENet_2014

    asv15 = [77.1,93.4,95.1]
    asv17 = [50,55.6,90.2]
    asv19la = [89.8,89.7,89.7]
    asv19pa = [90,90,90]
    asv21la = [89.8,90,]
    asv21pa = [83.4,87.0]
    asv21df = [90.2,97.2]
    add=[88.21,91.56,89.30,90.64,89.22,90.56,88.88,91.85,27.32,49.99,49.99,60.00,85.71,85.71,86.78]
    FoR=[44.08]
    FakeAVCeleb=[52.62]
    VCC2016=[91.8]
    VCC2018=[93.7]
    VCC2020=[95.6]
    WaveFake=[100]
    SASV=[89.83,89.22]
    CFAD=[66.64,66.67,66.67]
    halftruth = [50, 50, 50]
    FMFCCAA = [60.00,85.00,85.00]
    VSDC = [87.99]
    ReMASC = [83.11]
    reddotts = [85.96, 87.26]
    BTAS = [88.58,88.54,88.96]
    Baidu = [92.31]
    ArDAD = [2.34]
    HVoice = [50.83, 50, 54.07]
    audio = asv15 + asv17 + asv19la + asv19pa + asv21la + asv21pa + asv21df + add + FoR + FakeAVCeleb + VCC2016 + VCC2018 + VCC2020 + WaveFake + SASV + CFAD +halftruth + FMFCCAA + VSDC + ReMASC + BTAS + Baidu + ArDAD + HVoice

    UADFV=[50]
    DeepfakeTIMIT = [66.7]
    DFDC = [83.93, 50, 50]
    FaceForensics= [75]
    DeeperForensics=[18.5]
    CelebDF = [90.5]
    KoDF = [73.9]
    WildDeepfake = [47.98]
    FakeAVCeleb = [97.5]
    LAVDF = [73.3]
    ForgeryNet = [54.97]
    DeepFakeMNIST=[50]
    DeePhy =[98.1]
    DFFD=[75]
    video = UADFV + DeepfakeTIMIT + DFDC + FaceForensics + DeeperForensics + CelebDF + KoDF + WildDeepfake + FakeAVCeleb + LAVDF + ForgeryNet + DeepFakeMNIST + DeePhy + DFFD


    ArtiFact = [61.35]
    CIFake = [50,50]
    DFFD=[80.3]
    Fakeddit = [54.8]
    DeepfakeMedicaImagTampeDetection = [7.1]
    ForgeryNet = [50.4]
    FakeSpotter=[45.5]
    TamperedFaceDetection=[33.5, 25]
    InconsistentHeadPoses=[50, 50]
    image = ArtiFact + CIFake + DFFD + Fakeddit + DeepfakeMedicaImagTampeDetection + ForgeryNet + FakeSpotter + TamperedFaceDetection + InconsistentHeadPoses


    fig, ax=plt.subplots(2,1, dpi=150, figsize=(6,4), gridspec_kw={'height_ratios':[0.001, 1]}, constrained_layout=True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    all = [ids,image , video,audio ]
    all = np.array(all)
    parts = ax[1].violinplot(all, showmedians=True, points=1000, showextrema=False, quantiles=[[.50,.75,.85],[.75,.5,0.15],[.75,.5,0.15],[.75,.5,0.15]])
    ax[1].set_xticks([1,2,3, 4])
    ax[1].set_xticklabels(["Network IDS", "Deepfake Image", "Deepfake Video", "Deepfake Speech"])
    ax[1].set_ylabel("Percentage of Anomalous Samples", fontsize=12)



    medians= np.percentile(np.array(all[0]), [25, 50, 75], axis=0)
    medians1 = np.percentile(np.array(all[1]), [25, 50, 75], axis=0)
    medians2 = np.percentile(np.array(all[2]), [25, 50, 75], axis=0)
    medians3 = np.percentile(np.array(all[3]), [25, 50, 75], axis=0)
    medians = [medians,medians1, medians2,medians3]
    inds = np.arange(1, len(medians) + 1)


    medians= np.median(np.array(all[0]))
    medians1 = np.median(np.array(all[1]))
    medians2 = np.median(np.array(all[2]))
    medians3 = np.median(np.array(all[3]))

    # m = medians[0][1]
    # m1 = medians[1][1]
    # m2 = medians[2][1]
    # m3 = medians[3][1]

    ax[1].scatter(inds[0], medians, s=18, color=colors[0], zorder=3,label="Networking:{:.1f}%".format(medians), edgecolor='None', facecolor=colors[0])
    ax[1].scatter(inds[1], medians1, s=18, color=colors[1], zorder=3,label="Image:{:.1f}%".format(medians1), edgecolor='None', facecolor=colors[1])

    ax[1].scatter(inds[2], medians2, s=18, color=colors[2], zorder=3,label="Video:{:.1f}%".format(medians2), edgecolor='None', facecolor=colors[2])
    ax[1].scatter(inds[3], medians3, s=18, color=colors[3], zorder=3,label="Speech:{:.1f}%".format(medians3), edgecolor='None', facecolor=colors[3])

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    parts['bodies'][0].set_color(colors[0])

    parts['bodies'][1].set_color(colors[1])

    cmap = [colors[0],colors[0], 'black', colors[1],colors[1],'black',colors[2],colors[2],'black', colors[3],colors[3],'black']
    parts['cmedians'].set_colors(colors)

    parts['bodies'][2].set_color(colors[2])

    parts['cquantiles'].set_color(cmap)

    parts['bodies'][3].set_color(colors[3])



    medians_= np.percentile(np.array(all[0]), [50, 75,85], axis=0)
    medians1_ = np.percentile(np.array(all[1]), [10,50,75,90], axis=0)
    medians2_ = np.percentile(np.array(all[2]), [10,50,75,90], axis=0)
    medians3_ = np.percentile(np.array(all[3]), [10,50,75,90], axis=0)

    ax[1].annotate("      90%", (1.05,medians_[2]), fontsize=9)
    ax[1].annotate("      75%", (2.05,medians1_[2]+.2), fontsize=9)
    ax[1].annotate("      75%", (3.05,medians2_[2]), fontsize=9)
    ax[1].annotate("      50%", (4.05,medians3_[2]-4), fontsize=9)

    ax[1].annotate("      75%", (1.05,medians_[1]+1), fontsize=9)
    ax[1].annotate("      50%", (2.05,medians1_[1]-2.2), fontsize=9)
    ax[1].annotate("      50%", (3.05,medians2_[1]-2), fontsize=9)
    ax[1].annotate("      75%", (4.05,medians3_[1]+2), fontsize=9)

    ax[1].annotate("      50%", (1.05,medians_[0]-2), fontsize=9)
    ax[1].annotate("      10%", (2.05,medians1_[0]+2.3), fontsize=9)
    ax[1].annotate("      10%", (3.05,medians2_[0]), fontsize=9)
    ax[1].annotate("      10%", (4.05,medians3_[0]-.5), fontsize=9)


    h, l = ax[1].get_legend_handles_labels()
    ax[0].legend(h, l, ncol=4, loc="upper center", fontsize=10, title="Median Fake/Real Split",handletextpad=0.05, columnspacing=0.05,)
    ax[0].axis("off")
    plt.savefig("figs/figure1.png")

def figure2(y_true_asv, y_true_ro, scores_file_type):
    
    scores_file_asv_2575 = f"../ASVspoof/LFCC-LCNN/results/25-75_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_2575 = f"../ASVspoof/LFCC-LCNN/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_asv_5050 = f"../ASVspoof/LFCC-LCNN/results/50-50_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_5050 = f"../ASVspoof/LFCC-LCNN/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_asv_7525 = f"../ASVspoof/LFCC-LCNN/results/75-25_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_7525 = f"../ASVspoof/LFCC-LCNN/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_asv_9010 = f"../ASVspoof/LFCC-LCNN/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../ASVspoof/LFCC-LCNN/results/90-10_ro-eval_{scores_file_type}.scores"

    asv_2575 = calc_tpfptnfn(scores_file_asv_2575, y_true_asv)
    ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_ro)

    asv_5050 = calc_tpfptnfn(scores_file_asv_5050, y_true_asv)
    ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_ro)

    asv_7525 = calc_tpfptnfn(scores_file_asv_7525, y_true_asv)
    ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_ro)

    asv_9010 = calc_tpfptnfn(scores_file_asv_9010, y_true_asv)
    ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_ro)

    asv_ = [asv_9010, asv_7525, asv_5050, asv_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(asv_,"ASV_LL", 2, "a", scores_file_type)
    plot_bars(ro_,"ASV_LLRO", 2, "b", scores_file_type)
    return

def figure3(y_true_asv, y_true_cfad, scores_file_type, cfad_provided=False):

    measured_fprs = []
    measured_tprs = []
    model_name_mapping = []
    name = [r'CQCC-GMM ($M_{ASV-CG}$)', r'LFCC-GMM ($M_{ASV-LG}$)', r'LFCC-LCNN ($M_{ASV-LL}$)', r'RawNet2 ($M_{ASV-RN}$)', r'SSL-wav2vec ($M_{ASV-SW}$)', r'LFCC-LCNN ($M_{CFAD-LL}$)', r'RawNet2 ($M_{CFAD-RN}$)']


    scores_file_asv_LL = f"../ASVspoof/LFCC-LCNN/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_asv_RN = f"../ASVspoof/RawNet2/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_asv_SW = f"../ASVspoof/wav2vec/results/90-10_asv21-eval_{scores_file_type}.scores" 
    scores_file_asv_LG = "../ASVspoof/LFCC-GMM/results/90-10_asv21-eval_provided.scores"
    scores_file_asv_CG = "../ASVspoof/CQCC-GMM/results/90-10_asv21-eval_provided.scores"

    asv_LL = calc_tpfptnfn(scores_file_asv_LL, y_true_asv)
    asv_RN = calc_tpfptnfn(scores_file_asv_RN, y_true_asv)
    asv_SW = calc_tpfptnfn(scores_file_asv_SW, y_true_asv)
    asv_LG = calc_tpfptnfn(scores_file_asv_LG, y_true_asv)
    asv_CG = calc_tpfptnfn(scores_file_asv_CG, y_true_asv)

    LL_r = calc_tprfpr(asv_LL) 
    measured_fprs.append(LL_r[0])
    measured_tprs.append(LL_r[1])
    model_name_mapping.append(name[2])

    RN_r = calc_tprfpr(asv_RN) 
    measured_fprs.append(RN_r[0])
    measured_tprs.append(RN_r[1])
    model_name_mapping.append(name[3])

    SW_r = calc_tprfpr(asv_SW) 
    measured_fprs.append(SW_r[0])
    measured_tprs.append(SW_r[1])
    model_name_mapping.append(name[4])

    LG_r = calc_tprfpr(asv_LG) 
    measured_fprs.append(LG_r[0])
    measured_tprs.append(LG_r[1])
    model_name_mapping.append(name[1])

    CG_r = calc_tprfpr(asv_CG) 
    measured_fprs.append(CG_r[0])
    measured_tprs.append(CG_r[1])
    model_name_mapping.append(name[0])

    scores_file_cfad_LL = f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores"
    scores_file_cfad_RN = f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores"

    if cfad_provided:
        y_true_cfad_LL = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_cfad_RN = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores")

        cfad_LL = calc_tpfptnfn(scores_file_cfad_LL, y_true_cfad_LL, cfad_provided)
        cfad_RN = calc_tpfptnfn(scores_file_cfad_RN, y_true_cfad_RN, cfad_provided)
    else:
        cfad_LL = calc_tpfptnfn(scores_file_cfad_LL, y_true_cfad)
        cfad_RN = calc_tpfptnfn(scores_file_cfad_RN, y_true_cfad)

    LL_r = calc_tprfpr(cfad_LL) 
    measured_fprs.append(LL_r[0])
    measured_tprs.append(LL_r[1])
    model_name_mapping.append(name[5])

    RN_r = calc_tprfpr(cfad_RN) 
    measured_fprs.append(RN_r[0])
    measured_tprs.append(RN_r[1])
    model_name_mapping.append(name[6])

    base_rate = np.geomspace(0.005,1,num=10000)
    normal_rate = 1 - base_rate    

    model_dict = {}
    for fpr, tpr, mn in zip(measured_fprs, measured_tprs, model_name_mapping):
        model_br_dict = {}
        for i in range(len(base_rate)):
            bdr = (base_rate[i] * tpr) / ((base_rate[i] * tpr) + (normal_rate[i] * fpr))
            model_br_dict[base_rate[i]] = (bdr,base_rate[i],fpr, tpr)
        model_dict[mn] = model_br_dict
    plt.rcParams['legend.title_fontsize'] = 'xx-large'
    fig, ax = plt.subplots(1,1, figsize=(12,8), dpi=300)
    
    ax.set_xscale("log")
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    for model in model_dict.keys():
        m_dict = model_dict.get(model)
        # model_name = new_dict.get(model)
        model_metrics =  np.array(list(m_dict.values()))
        x = model_metrics[:,1]
        y = model_metrics[:,0]
        
        ax.plot(x,y,next(linecycler), label=model, linewidth=4)

    ax.set_xlabel(r"Base-Rate [$P(D)$]", fontsize=26)
    ax.set_ylabel(r"Bayesian Detection Rate [$P(D|A)$]", fontsize=24)
    ax.legend(fancybox=True, shadow=True, title="Model Name", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    plt.tight_layout()
    plt.savefig(f"figs/figure3-{scores_file_type}.png")

    return

def figure4(y_true_ro, scores_file_type):
    measured_fprs = []

    scores_file_ro_2575 = f"../ASVspoof/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_ro_5050 = f"../ASVspoof/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_ro_7525 = f"../ASVspoof/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_ro_9010 = f"../ASVspoof/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores"

    ro_2575 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_2575, y_true_ro))

    ro_5050 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_5050, y_true_ro))

    ro_7525 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_7525, y_true_ro))

    ro_9010 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_9010, y_true_ro))

    measured_fprs = [ro_2575[0], ro_5050[0], ro_7525[0], ro_9010[0]]

    base_rate = np.array([0.01,0.1,1,10,25,90])/100
    base_rate_format = (np.array([0.01,0.1,1,10,25,90])/100)
    normal_rate = 1 - base_rate

    false_alarm_rate = np.geomspace(0.00001,1,num=10000)
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    fig, ax = plt.subplots(1,1, figsize=(14,4.5), dpi=150)
    twiny = ax.twinx()
    all = []
    for i in range(len(base_rate)):
        temp_out_y = []
        temp_out_x = []
        for y in false_alarm_rate:
            bayesian_detection_rate = (base_rate[i] * 1.0) / ((base_rate[i] * 1.0) + (normal_rate[i] * y))
            temp_out_y.append(bayesian_detection_rate)
            temp_out_x.append(y)
        
        all.append((temp_out_x, temp_out_y))
    lines = [":","--"]
    linecycler = cycle(lines)
    for i in range(len(all)):        
            
        all_np = np.array(all[i]).T
        x = all_np[:,0].astype(np.float64)
        y = all_np[:,1].astype(np.float64)
        if i == len(all) - 1:
            ax.plot(x,y, label=str(base_rate_format[i]*100), linewidth=6, color='grey')
        else:
            ax.plot(x,y, ls= next(linecycler) ,label=str(base_rate_format[i]*100), linewidth=3)

    fancy_names = [r'$D_{90F/10R}$', r'$D_{75F/25R}$', r'$D_{50F/50R}$', r'$D_{25F/75R}$'][::-1]
    for j in range(4):
        ax.axvline(measured_fprs[j],ymin=0, ymax=1.0, color='black',linewidth=2.5, marker='x', zorder=-100000)
        ax.annotate(fancy_names[j], xy=(measured_fprs[j] - (measured_fprs[j] *.1), 1.05), size=14,zorder=1000000, color='black', rotation=30)

    ax.set_xlabel(r"False Positive Rate[$P(A|\neg D)$]", fontsize=14)
    ax.set_ylabel(r"Bayesian Detection Rate [$P(D|A)$]", fontsize=14)

    leg = ax.legend(fancybox=True, shadow=True, title="Base Rate (%)", fontsize=14, ncol=2)
    # ax.set_yscale("log")
    ax.set_xscale("log")
    ax.add_artist(leg)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.tick_params(right = False)
    twiny.yaxis.set_ticklabels([])



    ax.annotate("D", xy=(0.575, 0.18), size=20,zorder=1000000, color='b') #bdr = 0.16959828
    ax.annotate("E", xy=(0.575, 0.03), size=20,zorder=1000000, color='b') #bdr = 0.0018365886
    ax.annotate("C", xy=(0.575, 0.86), size=20,zorder=1000000, color='b') #bdr = 0.942997698

    ax.annotate("B", xy=(0.001, 0.1), size=20,zorder=1000000, color='b') #bdr = 0.09966810
    ax.annotate("A", xy=(0.001, 0.92), size=20,zorder=1000000, color='b') #bdr = 0.99193479

    ax.plot(0.5440317791402892, 0.0018365886190212431367216825158619940012222848852142084016770089, 'bo', markersize=9)
    ax.plot(0.5440317791402892, 0.9429976982757600693553864412879789008570171977401888890905365344, 'bo', markersize=9)
    ax.plot(0.0009034203761151596, 0.0996681018244799493997883766675425400531468949963512190848638546, 'bo', markersize=9)
    ax.plot(0.0009034203761151596, 0.9919347930655034488088705850962222281489123442142476691417287268, 'bo', markersize=9)

    ax.plot(0.5440317791402892, 0.1695982857548435720769744038004418137838584446305655529717489763, 'bo', markersize=9)

    

    plt.tight_layout()
    plt.savefig(f"figs/figure4-{scores_file_type}.png")
    return

def figure6(scores_file_type):

    fprs = [.269, .0879, .001, .0001] #Values directly from scenario 1 and 2 from paper
    tprs = [.999, .449, .9, .83] #Values directly from scenario 1 and 2 from paper

    base_rate = np.array([0.1, 0.1,50, 50])/100
    base_rate_format = (np.array([0.1, 0.1,50, 50])/100)
    normal_rate = 1 - base_rate

    false_alarm_rate = np.geomspace(0.00001,1,num=10000)
    plt.rcParams['legend.title_fontsize'] = 'xx-large'
    _, ax = plt.subplots(1,1, figsize=(12,8), dpi=150)

    all = []
    for i in range(len(base_rate)):
        temp_out_y = []
        temp_out_x = []
        for y in false_alarm_rate:
            bayesian_detection_rate = (base_rate[i] * tprs[i]) / ((base_rate[i] * tprs[i]) + (normal_rate[i] * y))
            temp_out_y.append(bayesian_detection_rate)
            temp_out_x.append(y)
        
        all.append((temp_out_x, temp_out_y))
    lines = ["-.",":", "-", "--"]
    linecycler = cycle(lines)
    c = iter([plt.cm.tab20(i) for i in range(20)])
    ii = [1,1,2,2]
    for i in range(len(all)):        
            
        all_np = np.array(all[i]).T
        x = all_np[:,0].astype(np.float64)
        y = all_np[:,1].astype(np.float64)
        ax.plot(x,y, ls= next(linecycler), color=next(c) ,label=f"{str(base_rate_format[i]*100)}% - Scenario {ii[i]}", linewidth=3)

    colorss=['lime', 'gold', 'crimson', 'darkblue']
    fancy_names = [r"$M_{ASV-SW}$", r'$M_{ASV-LG}$', r'$M_{1}$', r'$M_{2}$']
    for j in range(len(fprs)):
        bayesian_detection_rate = (base_rate[j] * tprs[j]) / ((base_rate[j] * tprs[j]) + (normal_rate[i] * fprs[j]))
        ax.plot(fprs[j], bayesian_detection_rate, 'ko', markersize=9)
        ax.annotate(fancy_names[j], xy=(fprs[j], bayesian_detection_rate), size=24,zorder=1000000, color='black', rotation=30)

    ax.set_xlabel(r"False Positive Rate[$P(A|\neg D)$]", fontsize=26)
    ax.set_ylabel(r"Bayesian Detection Rate [$P(D|A)$]", fontsize=24)

    leg = ax.legend(fancybox=True, shadow=True, title="Base Rate (%)", fontsize=14, ncol=2)
    ax.set_xscale("log")
    ax.add_artist(leg)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.tick_params(right = False)

    plt.tight_layout()
    plt.savefig(f"figs/figure6-{scores_file_type}.png")
    return

def figure7(): #appendix
    base_rate = np.array([0.01,0.1,1,10,25,50,75,90])/100
    base_rate_format = (np.array([0.01,0.1,1,10,25,50,75,90])/100)
    normal_rate = 1 - base_rate

    true_positive_rate = np.array([1])

    false_alarm_rate = np.geomspace(0.00001,1,num=10000)
    plt.rcParams['legend.title_fontsize'] = 'xx-large'
    _, ax = plt.subplots(1,1, figsize=(12,8), dpi=600)
    
    ax.set_xscale("log")
    all = []
    for i in range(len(base_rate)):
        temp_out_y = []
        temp_out_x = []
        for y in false_alarm_rate:
            bayesian_detection_rate = (base_rate[i] * true_positive_rate[0]) / ((base_rate[i] * true_positive_rate[0]) + (normal_rate[i] * y))
            temp_out_y.append(bayesian_detection_rate)
            temp_out_x.append(y)
        
        all.append((temp_out_x, temp_out_y))

    lines = [":","--"]
    linecycler = cycle(lines)
    for i in range(len(all)): 
        all_np = np.array(all[i]).T
        x = all_np[:,0].astype(np.float64)
        y = all_np[:,1].astype(np.float64)
        if i == len(all) - 1:
            ax.plot(x,y, label=str(base_rate_format[i]*100), linewidth=6)
        else:
            ax.plot(x,y, ls= next(linecycler) ,label=str(base_rate_format[i]*100), linewidth=3)

    ax.set_xlabel(r"False Alarm Rate [$P(A|\neg I)$]", fontsize=26)
    ax.set_ylabel(r"Bayesian Detection Rate [$P(I|A)$]", fontsize=24)
    ax.legend(fancybox=True, shadow=True, title="Base Rate (%)", fontsize=17, ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    t = np.array([1.05])
    t1 = np.array([0.000007])
    
    ax.fill_between(np.concatenate((t1,false_alarm_rate,t)),0.8,1.00,color='green',alpha=0.15,zorder=-10000)
    ax.set_xlim(t1[0], 1.06)
    ax.set_ylim(0,1.02)
    plt.tight_layout()
    plt.savefig("figs/figure7.png")
    
    return

def figure8(): #appendix
    fig,ax = plt.subplots(1, 1, dpi=150, figsize=(6, 1 * 5))
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color1 = matplotlib.colors.to_rgb(colors[0]) + (0.45,)
    color2 = matplotlib.colors.to_rgb(colors[1]) + (0.45,)

    axes = ax
    #pre-calculated lengths for all the files in the training sets
    og_files_len = pkl.load(open("../ASVspoof/ASVspoofTrain/lengths_asvtrainreal.pkl", "rb"))
    new_files_len = pkl.load(open("../ASVspoof/ASVspoofTrain/lengths_librispeech.pkl", "rb"))

    
    axes.hist(og_files_len, bins=35, density=True, fc=color1, ec='None')
    axes.hist(new_files_len, bins=35, density=True, fc=color2, ec='None')

    ec1 = matplotlib.colors.to_rgb(colors[0]) + (1.0,)
    ec2 = matplotlib.colors.to_rgb(colors[1]) + (1.0,)
    axes.hist(og_files_len, bins=35, density=True, ls="dashed", label="ASV_Train-Real", fc='None', ec=ec1)
    axes.hist(new_files_len, bins=35, density=True, lw=1.5, label="LibriSpeech-Real", fc='None', ec=ec2)

    axes.set_xlabel("Training Lengths(sec)", fontsize=20)
    axes.set_ylabel("Density", fontsize=20)
    axes.set_ylim(0, 0.6)
    axes.set_xlim(0, 15)
    axes.legend()
    
    plt.tight_layout()
    plt.savefig("figs/figure8.png")
    return

def figure9(): #appendix
    fig,ax = plt.subplots(1, 1, dpi=150, figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color1 = matplotlib.colors.to_rgb(colors[0]) + (0.45,)
    color2 = matplotlib.colors.to_rgb(colors[1]) + (0.45,)
    axes = ax
    #pre-calculated lengths for all the files in the training sets
    og_files_len = pkl.load(open("../CFAD/CFADTrain/lengths_cfadtrainreal.pkl", "rb"))
    new_files_len = pkl.load(open("../CFAD/CFADTrain/lengths_wenet.pkl", "rb"))

    axes.hist(og_files_len, bins=100, density=True, fc=color1, ec='None')
    axes.hist(new_files_len, bins=100, density=True, fc=color2, ec='None')

    ec1 = matplotlib.colors.to_rgb(colors[0]) + (1.0,)
    ec2 = matplotlib.colors.to_rgb(colors[1]) + (1.0,)
    axes.hist(og_files_len, bins=100, density=True, ls="dashed", label="CFAD_Train-Real", fc='None', ec=ec1)
    axes.hist(new_files_len, bins=100, density=True, lw=1.5, label="WeNet Speech-Real", fc='None', ec=ec2)

    axes.set_xlabel("Training Lengths(sec)", fontsize=20)
    axes.set_ylabel("Density", fontsize=20)
    axes.set_ylim(0, 0.3)
    axes.set_yticks([0.0,0.1,0.2,0.3])
    axes.set_xlim(0, 15)
    axes.legend()

    plt.tight_layout()
    plt.savefig("figs/figure9.png")
    return

def figure10(y_true_asv, y_true_ro, scores_file_type): #appendix

    scores_file_asv_2575 = f"../ASVspoof/RawNet2/results/25-75_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_2575 = f"../ASVspoof/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_asv_5050 = f"../ASVspoof/RawNet2/results/50-50_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_5050 = f"../ASVspoof/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_asv_7525 = f"../ASVspoof/RawNet2/results/75-25_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_7525 = f"../ASVspoof/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_asv_9010 = f"../ASVspoof/RawNet2/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../ASVspoof/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores"

    asv_2575 = calc_tpfptnfn(scores_file_asv_2575, y_true_asv)
    ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_ro)

    asv_5050 = calc_tpfptnfn(scores_file_asv_5050, y_true_asv)
    ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_ro)

    asv_7525 = calc_tpfptnfn(scores_file_asv_7525, y_true_asv)
    ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_ro)

    asv_9010 = calc_tpfptnfn(scores_file_asv_9010, y_true_asv)
    ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_ro)

    asv_ = [asv_9010, asv_7525, asv_5050, asv_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(asv_,"ASV_RN", 10, "1", scores_file_type)
    plot_bars(ro_,"ASV_RNRO", 10, "2", scores_file_type)


    scores_file_asv_2575 = f"../ASVspoof/wav2vec/results/25-75_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_2575 = f"../ASVspoof/wav2vec/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_asv_5050 = f"../ASVspoof/wav2vec/results/50-50_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_5050 = f"../ASVspoof/wav2vec/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_asv_7525 = f"../ASVspoof/wav2vec/results/75-25_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_7525 = f"../ASVspoof/wav2vec/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_asv_9010 = f"../ASVspoof/wav2vec/results/90-10_asv21-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../ASVspoof/wav2vec/results/90-10_ro-eval_{scores_file_type}.scores"

    asv_2575 = calc_tpfptnfn(scores_file_asv_2575, y_true_asv)
    ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_ro)

    asv_5050 = calc_tpfptnfn(scores_file_asv_5050, y_true_asv)
    ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_ro)

    asv_7525 = calc_tpfptnfn(scores_file_asv_7525, y_true_asv)
    ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_ro)

    asv_9010 = calc_tpfptnfn(scores_file_asv_9010, y_true_asv)
    ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_ro)

    asv_ = [asv_9010, asv_7525, asv_5050, asv_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(asv_,"ASV_SW", 10, "3", scores_file_type)
    plot_bars(ro_,"ASV_SWRO", 10, "4", scores_file_type)
    
    return

def figure11(y_true_cfad, y_true_ro, scores_file_type, cfad_provided=False): #appendix

    scores_file_cfad_2575 = f"../CFAD/RawNet2/results/25-75_cfad-eval_{scores_file_type}.scores"
    scores_file_ro_2575 = f"../CFAD/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_cfad_5050 = f"../CFAD/RawNet2/results/50-50_cfad-eval_{scores_file_type}.scores"
    scores_file_ro_5050 = f"../CFAD/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_cfad_7525 = f"../CFAD/RawNet2/results/75-25_cfad-eval_{scores_file_type}.scores"
    scores_file_ro_7525 = f"../CFAD/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_cfad_9010 = f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../CFAD/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores"

    if cfad_provided:
        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/25-75_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/25-75_ro-eval_{scores_file_type}.scores")

        cfad_2575 = calc_tpfptnfn_(scores_file_cfad_2575, y_true_cfad, cfad_provided=cfad_provided)
        ro_2575 = calc_tpfptnfn_(scores_file_ro_2575, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/50-50_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/50-50_ro-eval_{scores_file_type}.scores")

        cfad_5050 = calc_tpfptnfn_(scores_file_cfad_5050, y_true_cfad, cfad_provided=cfad_provided)
        ro_5050 = calc_tpfptnfn_(scores_file_ro_5050, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/75-25_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/75-25_ro-eval_{scores_file_type}.scores")

        cfad_7525 = calc_tpfptnfn_(scores_file_cfad_7525, y_true_cfad, cfad_provided=cfad_provided)
        ro_7525 = calc_tpfptnfn_(scores_file_ro_7525, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/RawNet2/results/90-10_ro-eval_{scores_file_type}.scores")

        cfad_9010 = calc_tpfptnfn_(scores_file_cfad_9010, y_true_cfad, cfad_provided=cfad_provided)
        ro_9010 = calc_tpfptnfn_(scores_file_ro_9010, y_true_ro, cfad_provided=cfad_provided)
    else:
        cfad_2575 = calc_tpfptnfn(scores_file_cfad_2575, y_true_cfad)
        ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_ro)
        cfad_5050 = calc_tpfptnfn(scores_file_cfad_5050, y_true_cfad)
        ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_ro)
        cfad_7525 = calc_tpfptnfn(scores_file_cfad_7525, y_true_cfad)
        ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_ro)
        cfad_9010 = calc_tpfptnfn(scores_file_cfad_9010, y_true_cfad)
        ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_ro)

    cfad_ = [cfad_9010, cfad_7525, cfad_5050, cfad_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(cfad_,"CFAD_RN", 11, "1", scores_file_type)
    plot_bars(ro_,"CFAD_RNRO", 11, "2", scores_file_type)


    if cfad_provided:
        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/25-75_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/25-75_ro-eval_{scores_file_type}.scores")

        cfad_2575 = calc_tpfptnfn_(scores_file_cfad_2575, y_true_cfad, cfad_provided=cfad_provided)
        ro_2575 = calc_tpfptnfn_(scores_file_ro_2575, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/50-50_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/50-50_ro-eval_{scores_file_type}.scores")

        cfad_5050 = calc_tpfptnfn_(scores_file_cfad_5050, y_true_cfad, cfad_provided=cfad_provided)
        ro_5050 = calc_tpfptnfn_(scores_file_ro_5050, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/75-25_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/75-25_ro-eval_{scores_file_type}.scores")

        cfad_7525 = calc_tpfptnfn_(scores_file_cfad_7525, y_true_cfad, cfad_provided=cfad_provided)
        ro_7525 = calc_tpfptnfn_(scores_file_ro_7525, y_true_ro, cfad_provided=cfad_provided)

        y_true_cfad = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_cfad-eval_{scores_file_type}.scores")
        y_true_ro = get_y_true_cfad_provided(f"../CFAD/LFCC-LCNN/results/90-10_ro-eval_{scores_file_type}.scores")

        cfad_9010 = calc_tpfptnfn_(scores_file_cfad_9010, y_true_cfad, cfad_provided=cfad_provided)
        ro_9010 = calc_tpfptnfn_(scores_file_ro_9010, y_true_ro, cfad_provided=cfad_provided)
    else:
        cfad_2575 = calc_tpfptnfn(scores_file_cfad_2575, y_true_cfad)
        ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_ro)
        cfad_5050 = calc_tpfptnfn(scores_file_cfad_5050, y_true_cfad)
        ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_ro)
        cfad_7525 = calc_tpfptnfn(scores_file_cfad_7525, y_true_cfad)
        ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_ro)
        cfad_9010 = calc_tpfptnfn(scores_file_cfad_9010, y_true_cfad)
        ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_ro)

    cfad_ = [cfad_9010, cfad_7525, cfad_5050, cfad_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(cfad_,"CFAD_LL", 11, "3", scores_file_type)
    plot_bars(ro_,"CFAD_LLRO", 11, "4", scores_file_type)
    
    return

def figure12(y_true_cifake, y_true_cifakero, scores_file_type):

    scores_file_cifake_2575 = f"../CIFAKE/CIFAKE/results/25-75_cifake-eval_{scores_file_type}.scores"
    scores_file_ro_2575 = f"../CIFAKE/CIFAKE/results/25-75_ro-eval_{scores_file_type}.scores"

    scores_file_cifake_5050 = f"../CIFAKE/CIFAKE/results/50-50_cifake-eval_{scores_file_type}.scores"
    scores_file_ro_5050 = f"../CIFAKE/CIFAKE/results/50-50_ro-eval_{scores_file_type}.scores"

    scores_file_cifake_7525 = f"../CIFAKE/CIFAKE/results/75-25_cifake-eval_{scores_file_type}.scores"
    scores_file_ro_7525 = f"../CIFAKE/CIFAKE/results/75-25_ro-eval_{scores_file_type}.scores"

    scores_file_cifake_9010 = f"../CIFAKE/CIFAKE/results/90-10_cifake-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../CIFAKE/CIFAKE/results/90-10_ro-eval_{scores_file_type}.scores"

    cifake_2575 = calc_tpfptnfn(scores_file_cifake_2575, y_true_cifake)
    ro_2575 = calc_tpfptnfn(scores_file_ro_2575, y_true_cifakero)
    cifake_5050 = calc_tpfptnfn(scores_file_cifake_5050, y_true_cifake)
    ro_5050 = calc_tpfptnfn(scores_file_ro_5050, y_true_cifakero)
    cifake_7525 = calc_tpfptnfn(scores_file_cifake_7525, y_true_cifake)
    ro_7525 = calc_tpfptnfn(scores_file_ro_7525, y_true_cifakero)
    cifake_9010 = calc_tpfptnfn(scores_file_cifake_9010, y_true_cifake)
    ro_9010 = calc_tpfptnfn(scores_file_ro_9010, y_true_cifakero)

    cifake_ = [cifake_9010, cifake_7525, cifake_5050, cifake_2575]
    ro_ = [ro_9010, ro_7525, ro_5050, ro_2575]

    plot_bars(cifake_,"CIFAKE", 12, "1", scores_file_type)
    plot_bars(ro_,"CIFAKE_RO", 12, "2", scores_file_type)

    return

def figure13(y_true_cifakero, scores_file_type):

    scores_file_ro_2575 = f"../CIFAKE/CIFAKE/results/25-75_ro-eval_{scores_file_type}.scores"
    scores_file_ro_9010 = f"../CIFAKE/CIFAKE/results/90-10_ro-eval_{scores_file_type}.scores"

    ro_2575 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_2575, y_true_cifakero))
    ro_9010 = calc_tprfpr(calc_tpfptnfn(scores_file_ro_9010, y_true_cifakero))

    measured_fprs = [ro_2575[0], ro_9010[0]]

    base_rate = np.array([0.01,0.1,1,10,25,90])/100
    base_rate_format = (np.array([0.01,0.1,1,10,25,90])/100)
    normal_rate = 1 - base_rate

    false_alarm_rate = np.geomspace(0.00001,1,num=10000)
    plt.rcParams['legend.title_fontsize'] = 'x-large'
    _, ax_=plt.subplots(2,1, dpi=150, figsize=(7,5), gridspec_kw={'height_ratios':[1, 0.001]}, constrained_layout=True)
    ax= ax_[0]
    twiny = ax.twinx()
    all = []
    for i in range(len(base_rate)):
        temp_out_y = []
        temp_out_x = []
        for y in false_alarm_rate:
            bayesian_detection_rate = (base_rate[i] * 1.0) / ((base_rate[i] * 1.0) + (normal_rate[i] * y))
            temp_out_y.append(bayesian_detection_rate)
            temp_out_x.append(y)
        
        all.append((temp_out_x, temp_out_y))
    lines = [":","--"]
    linecycler = cycle(lines)
    for i in range(len(all)):        
            
        all_np = np.array(all[i]).T
        x = all_np[:,0].astype(np.float64)
        y = all_np[:,1].astype(np.float64)
        if i == len(all) - 1:
            ax.plot(x,y, label=str(base_rate_format[i]*100), linewidth=3, color='grey')
        else:
            ax.plot(x,y, ls= next(linecycler) ,label=str(base_rate_format[i]*100), linewidth=3)

    fancy_names = [r'$D_{90F/10R}$', r'$D_{25F/75R}$'][::-1]
    for j in range(len(measured_fprs)):
        ax.axvline(measured_fprs[j],ymin=0, ymax=1.0, color='black',linewidth=2.5, marker='x', zorder=-100000)
        ax.annotate(fancy_names[j], xy=(measured_fprs[j] - (measured_fprs[j] *.1), 1.05), size=14,zorder=1000000, color='black', rotation=30)

    ax.set_xlabel(r"False Positive Rate[$P(A|\neg D)$]", fontsize=14)
    ax.set_ylabel(r"Bayesian Detection Rate [$P(D|A)$]", fontsize=14)
    ax.set_xlim(left=0.01, right=1.0)
    h, l = ax.get_legend_handles_labels()
    ax_[1].legend(h, l, ncol=6, loc="upper center", fontsize=11, title="Base Rate (%)")
    ax_[1].axis("off")
    ax.set_xscale("log")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.tick_params(right = False)
    twiny.yaxis.set_ticklabels([])

    # plt.tight_layout()
    plt.savefig("figs/figure13.png")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--figure', type=str, default="paper")

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

    

    figure1()
    figure2(y_true_asv, y_true_asvro, scores_file_type)
    figure3(y_true_asv, y_true_cfad, scores_file_type, cfad_provided=cfad_provided)
    figure4(y_true_asvro, scores_file_type)
    figure6(scores_file_type)
    if args.figure == "all":
        figure7()
        figure8()
        figure9()
        figure10(y_true_asv, y_true_asvro, scores_file_type)
        figure11(y_true_cfad, y_true_cfadro, scores_file_type, cfad_provided=cfad_provided)
        figure12(y_true_cifake, y_true_cifakero, scores_file_type)
        figure13(y_true_cifakero, scores_file_type)