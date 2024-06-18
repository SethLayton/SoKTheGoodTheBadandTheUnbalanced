# SoK: The Good, The Bad, and The Unbalanced
This is the repository for the paper titled [SoK: The Good, The Bad, and The Unbalanced: Measuring Structural Limitations of Deepfake Media Datasets](https://www.usenix.org/system/files/sec24fall-prepub-1479-layton.pdf). Following all the steps provided will allow the direct reproduction of all experiments in the paper.

<details>
	<summary><h2><b>Abstract</h2></b></summary><p>
	
	
Deepfake media represents an important and growing threat not only to computing systems but to society at large. Datasets of image, video, and voice deepfakes are being created to assist researchers in building strong defenses against these emerging threats. However, despite the growing number of datasets and the relative diversity of their samples, little guidance exists to help researchers select datasets and then meaningfully contrast their results against prior efforts. To assist in this process, this paper presents the first systematization of deepfake media. Using traditional anomaly detection datasets as a baseline, we characterize the metrics, generation techniques, and class distributions of existing datasets. Through this process, we discover significant problems impacting the comparability of systems using these datasets, including unaccounted-for heavy class imbalance and reliance upon limited metrics. These observations have a potentially profound impact should such systems be transitioned to practice - as an example, we demonstrate that the widely-viewed best detector applied to a typical call center scenario would result in only 1 out of 333 flagged results being a true positive. To improve reproducibility and future comparisons, we provide a template for reporting results in this space and advocate for the release of model score files such that a wider range of statistics can easily be found and/or calculated. Through this, and our recommendations for improving dataset construction, we provide important steps to move this community forward.
</details></p> 

<details>
	<summary><h2><b>Citation</b></h2></summary><p>
	
	

    @inproceedings{SoKTGTBTU,
    author = {Seth Layton and Tyler Tucker and Daniel Olszewski and Kevin Warren and Kevin Butler and Patrick Traynor},
    title = {{SoK: The Good, The Bad, and The Unbalanced: Measuring Structural Limitations of Deepfake Datasets}},
    booktitle = {{Proceedings of the USENIX Security Symposium (Security)}},
    year = {2024}
    }
</details></p> 

<details>
	<summary><h2><b>Hardware and Computation Setup</h2></b></summary><p>


This code has been verified to work on Ubuntu 22.04.4 LTS  and Red Hat Enterprise Linux 8.9 (Ootpa)
All training and evaluation scripts require a GPU for computation. We use 2x Nvidia GeForce RTX 2080 for all models except wav2vec, which uses 2x Nvidia DGX A100. During training and inference we allocate 50GB of RAM and 4 AMD EPYC 7742 2.25GHz CPUs to the process.
</details></p>  

<details>
	<summary><h2><b>Clone the repo</h2></b></summary><p>
	
	

    git clone https://github.com/SethLayton/SoKTheGoodTheBadandTheUnbalanced.git
    mkdir DataSets
    cd SoKTheGoodTheBadandTheUnbalanced
    
</details></p>    

<details>
	<summary><h2><b>Decide Reproduction Complexity</h2></b></summary><p>
	
	
---- **For Artifact Evaluation we recommend the Minimum complexity option** ----
### 1. Maximum complexity
Requires the most time and resources (All necessary data and models are >100GB)
 1. Retrain all models from paper from scratch **(STEP 1)**
 2. Reproduce scores files using re-trained models **(STEP2.1)**
 3. Reproduce figures/tables using provided scores files **(STEP3.1)**

### 2. Medium complexity 
Requires substantially less time than Max complexity (All necessary data and models are >70GB)
 1. Reproduce scores files using pre-trained (provided) models **(STEP2.2)**
 2. Reproduce figures/tables using re-calculated scores files **(STEP3.2)**

### 3. Minimum complexity
Requires relatively negligible time (No additional downloads)
 1. Reproduce figures using provided scores files **(STEP3.3)**

</details></p>

<details>
	<summary><h2><b>Environment Setup</h2></b></summary><p>
	
	
### Create conda environments  

##### 1. RawNet2/LFCC-LCNN (ASVspoof and CDFAD models)
	conda env create --name pytorch-asvspoof2021 --file=env_1.yml

##### 2. wav2vec (Just for ASVspoof) 
##### --Skip this if not retraining models, or recalculating scores files (i.e., just recreating figures/tables from our provided scores files) -- 

    conda env create --name ssl1 --file=env_2.yml    
    conda activate env_2    
    cd ASVspoof/wav2vec/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1    
    pip install --editable ./    
    conda deactivate env_2
    cd ../../../../

##### 3. CIFAKE
    conda env create --name cifake --file=env_3.yml
</details></p>

<details>
	<summary><h2><b>Download Data</h2></b></summary><p>
	
	
    
### ---- If Maximum Complexity ----
Download/extract all data

    cd ../DataSets	
	
	mkdir ASVspoof
	
    wget "https://zenodo.org/records/12007844/files/asv_libri.tar.gz.00"
	wget "https://zenodo.org/records/12007844/files/asv_libri.tar.gz.01"
    cat asv_libri.tar.gz.* | tar xzvf - -C ASVspoof/
	
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.00"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.01"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.02"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.03"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.04"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.05"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.06"
    cat asv_realonly.tar.gz.* | tar xzvf - -C ASVspoof/
	
	wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz"
	wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz"
	wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz"
	wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz"
	tar -xvzf ASVspoof2021_DF_eval_part00.tar.gz -C ASVspoof/
	tar -xvzf ASVspoof2021_DF_eval_part01.tar.gz -C ASVspoof/
	tar -xvzf ASVspoof2021_DF_eval_part02.tar.gz -C ASVspoof/
	tar -xvzf ASVspoof2021_DF_eval_part03.tar.gz -C ASVspoof/
	
	
	mkdir CFAD	
	
	wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.00"
	wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.01"
	wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.02"
    cat cfad_eval.tar.gz.* | tar xzvf - -C CFAD/
	
	wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.00"
	wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.01"
	wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.02"
	wget "https://zenodo.org/records/12089727/files/cfad_train.tar.gz.03"
    cat cfad_train.tar.gz.* | tar xzvf - -C CFAD/
	
	wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.00"
	wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.01"
    cat wenet_real_only.tar.gz.* | tar xzvf -
	
	
	mkdir CIFAKE
	
    wget "https://zenodo.org/records/12090252/files/cifake2575.tar.gz"
	tar -xvzf cifake2575.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/cifake5050.tar.gz"
    tar -xvzf cifake5050.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/7525.tar.gz"
    tar -xvzf cifake7525.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/9010.tar.gz"
    tar -xvzf cifake9010.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/cifakeRO.tar.gz"
    tar -xvzf cifakeRO.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/cifaketest.tar.gz"
    tar -xvzf cifaketest.tar.gz -C CIFAKE/
    

Download the wav2vec pretrained model

    cd ../DataSets
    wget "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"

Continue on to **STEP 1**



### ---- If Medium Complexity ----
Download/extract test data only

    cd ../DataSets	
	
	mkdir ASVspoof
	
    wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.00"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.01"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.02"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.03"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.04"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.05"
	wget "https://zenodo.org/records/12007844/files/asv_realonly.tar.gz.06"
    cat asv_realonly.tar.gz.* | tar xzvf - -C ASVspoof/
	
	wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part00.tar.gz"
    wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part01.tar.gz"
    wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part02.tar.gz"
    wget "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part03.tar.gz"
    tar -xvzf ASVspoof2021_DF_eval_part00.tar.gz -C ASVspoof/
    tar -xvzf ASVspoof2021_DF_eval_part01.tar.gz -C ASVspoof/
    tar -xvzf ASVspoof2021_DF_eval_part02.tar.gz -C ASVspoof/
    tar -xvzf ASVspoof2021_DF_eval_part03.tar.gz -C ASVspoof/
	
	
	mkdir CFAD
	
    wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.00"
	wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.01"
	wget "https://zenodo.org/records/12089727/files/cfad_eval.tar.gz.02"
    cat cfad_eval.tar.gz.* | tar xzvf - -C CFAD/
	
    wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.00"
	wget "https://zenodo.org/records/12089727/files/wenet_real_only.tar.gz.01"
    cat wenet_real_only.tar.gz.* | tar xzvf -
	
	
	mkdir CIFAKE
	
	wget "https://zenodo.org/records/12090252/files/cifakeRO.tar.gz"
    tar -xvzf cifakeRO.tar.gz -C CIFAKE/
	
	wget "https://zenodo.org/records/12090252/files/cifaketest.tar.gz"
    tar -xvzf cifaketest.tar.gz -C CIFAKE/
	
    
   
Download/extract all pre-trained models
  

     cd ../SoKTheGoodTheBadandTheUnbalanced
	 
     wget "https://zenodo.org/records/12007844/files/asvspoof_rawnet2model_pretrained.tar.gz
     tar -xvzf asvspoof_rawnet2model_pretrained.tar.gz -C ASVspoof/RawNet2/models
	 
	 wget "https://zenodo.org/records/12007844/files/asvspoof_wav2vecmodel_pretrained.tar.gz
     tar -xvzf asvspoof_wav2vecmodel_pretrained.tar.gz -C ASVspoof/wav2vec/models
	 
	 wget "https://zenodo.org/records/12007844/files/asvspoof_lfcclcnnmodel_pretrained.tar.gz
     tar -xvzf asvspoof_lfcclcnnmodel_pretrained.tar.gz -C ASVspoof/LFCC-LCNN/models
	 
	 wget "https://zenodo.org/records/12089727/files/cfad_rawnet2model_pretrained.tar.gz"
     tar -xvzf cfad_rawnet2model_pretrained.tar.gz -C CFAD/RawNet2/models
	 
	 wget "https://zenodo.org/records/12089727/files/cfad_lfcclcnnmodel_pretrained.tar.gz"
     tar -xvzf cfad_lfcclcnnmodel_pretrained.tar.gz -C CFAD/LFCC-LCNN/models
	 
	 wget "https://zenodo.org/records/12090252/files/cifake_rawcnnmodel_pretrained.tar.gz"
     tar -xvzf cifake_rawcnnmodel_pretrained.tar.gz -C CIFAKE/CIFAKE/models

Download the wav2vec pretrained model

    cd ../DataSets
    wget "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"

Continue on to **STEP 2.2**


### ---- If Minimum Complexity ----
No additional downloads necessary as all scores files to produce figures/tables are provided in the source code repo.
Continue on to **STEP 3.3**

</details></p>

<details>
	<summary><h2><b>STEP 1 -- Train from Scratch</h2></b></summary><p>
	
	
  
#### 1.1 ASVSpoof

##### 1.1.1 RawNet2

    cd ../SoKTheGoodTheBadandTheUnbalanced
Run training script that trains all training distributions:

    sh ASVspoof/RawNet2/train_models_different-training-dist.sh

  

Saves trained models in SoKTheGoodTheBadandTheUnbalanced/ASVspoof/RawNet2/models/

> 25-75.pth, 50-50.pth, 75-25.pth, 90-10.pth

  

##### 1.1.2 LFCC-LCNN

Run training script that trains all training distributions:

    sh ASVspoof/LFCC-LCNN/train_models_different-training-dist.sh

Saves trained models in SoKTheGoodTheBadandTheUnbalanced/ASVspoof/LFCC-LCNN/models/

> 25-75.pt, 50-50.pt, 75-25.pt, 90-10.pt

  
##### 1.1.3 wav2vec

Run training script that trains all training distributions:

    sh ASVspoof/wav2vec/train_models_different-training-dist.sh

Saves trained models in SoKTheGoodTheBadandTheUnbalanced/ASVspoof/wav2vec/models/

> 25-75.pth, 50-50.pth, 75-25.pth, 90-10.pth

#### 1.2 CFAD


##### 1.2.1 RawNet2

Run training script that trains all training distributions:

    sh CFAD/RawNet2/train_models_different-training-dist.sh

Saves trained models in SoKTheGoodTheBadandTheUnbalanced/CFAD/RawNet2/models/

> 25-75.pth, 50-50.pth, 75-25.pth, 90-10.pth

  

##### 1.2.2 LFCC-LCNN

Calculate LFCC features:

    python CFAD/LFCC-LCNN/lfcc-lcnn/g_lfcc_final.py --dir_dataset ../DataSets/CFAD/trn

Run training script that trains all training distributions:

    sh CFAD/LFCC-LCNN/train_models_different-training-dist.sh

Saves trained models in SoKTheGoodTheBadandTheUnbalanced/CFAD/LFCC-LCNN/models/

> 25-75.pt, 50-50.pt, 75-25.pt, 90-10.pt

</details></p>  
  
<details>
	<summary><h2><b>STEP 2 -- Evaluate Models</h2></b></summary><p>
	
	

  
### STEP 2.1 Evaluate using retrained models (from STEP 1)


#### 2.1.1 ASVspoof

##### 2.1.1.1 RawNet2
 Evaluate each training distribution against the ASVspoof Eval default dataset:
 

    sh ASVspoof/RawNet2/eval_models_asv-eval.sh retrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/RawNet2/eval_models_real-only.sh retrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/RawNet2/results/

> 25-75_asv-evalscores, 50-50_asv-eval.scores,
> 75-25_asv-eval.scores, 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores,
> 75-25_ro-eval.scores, 90-10_ro-eval.scores

  

##### 2.1.1.2 LFCC-LCNN

 Evaluate each training distribution against the ASVspoof Eval default dataset:

    sh ASVspoof/LFCC-LCNN/eval_models_asv-eval.sh retrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/LFCC-LCNN/eval_models_real-only.sh retrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/LFCC-LCNN/results/

> 25-75_asv-eval.scores, 50-50_asv-eval.scores,
> 75-25_asv-eval.scores, 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores,
> 75-25_ro-eval.scores, 90-10_ro-eval.scores

  

##### 2.1.1.3 wav2vec
  
Evaluate each training distribution against the ASVspoof Eval default dataset:

    sh ASVspoof/wav2vec/eval_models_asv-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/wav2vec/eval_models_real-only.sh pretrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/wav2vec/results/

> 25-75_asv-eval.scores, 50-50_asv-eval.scores,
> 75-25_asv-eval.scores, 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores,
> 75-25_ro-eval.scores, 90-10_ro-eval.scores


### 2.1.2 CFAD

#### 2.1.2.1 RawNet2

Evaluate each training distribution against the CFAD Eval default dataset:

    sh CFAD/RawNet2/eval_models_cfad-eval.sh retrained

Evaluate each training distribution against the Real Only dataset:

    sh CFAD/RawNet2/eval_models_real-only.sh retrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CFAD/RawNet2/results/

> 25-75_asv-eval.scores, 50-50_asv-eval.scores, 75-25_asv-eval.scores,
> 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores, 75-25_ro-eval.scores,
> 90-10_ro-eval.scores

  
#### 2.1.2.2 LFCC-LCNN

Calculate LFCC features:

    python CFAD/LFCC-LCNN/lfcc-lcnn/g_lfcc_final.py --dir_dataset ../DataSets/WenetSpeech/wav_distributed_tst

  
Evaluate each training distribution against the CFAD Eval default dataset:

    sh CFAD/LFCC-LCNN/eval_models_cfad-eval.sh retrained

Evaluate each training distribution against the Real Only dataset:

    sh CFAD/LFCC-LCNN/eval_models_real-only.sh retrained

  
Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CFAD/LFCC-LCNN/results/

> 25-75_asv-eval.scores, 50-50_asv-eval.scores, 75-25_asv-eval.scores,
> 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores, 75-25_ro-eval.scores,
> 90-10_ro-eval.scores

### 2.1.3 CIFAKE

#### 2.1.3.1 RawCNN

Evaluate each training distribution against the CIFAKE Eval default dataset:

    sh CIFAKE/CIFAKE/eval_models_cfad-eval.sh retrained

Evaluate each training distribution against the Real Only dataset:

    sh CIFAKE/CIFAKE/eval_models_real-only.sh retrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CIFAKE/CIFAKE/results/

> 25-75_asv-eval.scores, 50-50_asv-eval.scores, 75-25_asv-eval.scores,
> 90-10_asv-eval.scores
> 
> 25-75_ro-eval.scores, 50-50_ro-eval.scores, 75-25_ro-eval.scores,
> 90-10_ro-eval.scores
  

### STEP 2.2 Evaluate using pretrained models

### 2.2.1 ASVspoof
#### 2.2.1.1 RawNet2

Evaluate each training distribution against the ASVspoof Eval default dataset:

    sh ASVspoof/RawNet2/eval_models_asv-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/RawNet2/eval_models_real-only.sh pretrained

  
Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/RawNet2/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores,
> 75-25_asv-eval_p.scores, 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores,
> 75-25_ro-eval_p.scores, 90-10_ro-eval_p.scores

  

#### 2.2.1.2 LFCC-LCNN
Evaluate each training distribution against the ASVspoof Eval default dataset:

    sh ASVspoof/LFCC-LCNN/eval_models_asv-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/LFCC-LCNN/eval_models_real-only.sh pretrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/LFCC-LCNN/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores,
> 75-25_asv-eval_p.scores, 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores,
> 75-25_ro-evall_pscores, 90-10_ro-eval_p.scores

  

#### 2.2.1.3 wav2vec

Evaluate each training distribution against the ASVspoof Eval default dataset:

    sh ASVspoof/wav2vec/eval_models_asv-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh ASVspoof/wav2vec/eval_models_real-only.sh pretrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/ASVspoof/wav2vec/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores,
> 75-25_asv-eval_p.scores, 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores,
> 75-25_ro-eval_p.scores, 90-10_ro-eval_p.scores

  

### 2.2.2 CFAD
  
#### 2.2.2.1 RawNet2

Evaluate each training distribution against the CFAD Eval default dataset:

    sh CFAD/RawNet2/eval_models_cfad-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh CFAD/RawNet2/eval_models_real-only.sh pretrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CFAD/RawNet2/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores,
> 75-25_asv-eval_p.scores, 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores,
> 75-25_ro-eval_p.scores, 90-10_ro-eval_p.scores

  

#### 2.2.2.2 LFCC-LCNN

Calculate LFCC features:

    python CFAD/LFCC-LCNN/lfcc-lcnn/g_lfcc_final.py --dir_dataset ../DataSets/CFAD/tst

  
Evaluate each training distribution against the CFAD Eval default dataset:

    sh CFAD/LFCC-LCNN/eval_models_cfad-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh CFAD/LFCC-LCNN/eval_models_real-only.sh pretrained

outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CFAD/LFCC-LCNN/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores,
> 75-25_asv-eval_p.scores, 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores,
> 75-25_ro-eval_p.scores, 90-10_ro-eval_p.scores

  
### 2.2.3 CIFAKE

#### 2.2.3.1 RawCNN

Evaluate each training distribution against the CIFAKE Eval default dataset:

    sh CIFAKE/CIFAKE/eval_models_cfad-eval.sh pretrained

Evaluate each training distribution against the Real Only dataset:

    sh CIFAKE/CIFAKE/eval_models_real-only.sh pretrained

Outputs scores file to SoKTheGoodTheBadandTheUnbalanced/CIFAKE/CIFAKE/results/

> 25-75_asv-eval_p.scores, 50-50_asv-eval_p.scores, 75-25_asv-eval_p.scores,
> 90-10_asv-eval_p.scores
> 
> 25-75_ro-eval_p.scores, 50-50_ro-eval_p.scores, 75-25_ro-eval_p.scores,
> 90-10_ro-eval_p.scores

</details></p>

<details>
	<summary><h2><b>STEP 3 -- Generate Figures/Tables</h2></b></summary><p>
	
	
	
Only the tables that have calculated values are re-created here (i.e., no systematization tables)
Tables are output to the console in ASCII form and saved to a text file

### STEP 3.1 From retrained models and recalculated scores files (Steps 1 and 2.1)  


Generate paper figures and tables only:

    sh Figures/gen_paper_figs.sh retrained

Generate paper and appendix (companion website) figures and tables:

    sh Figures/gen_paper-appendix_figs.sh retrained

 Outputs figures and tables to SoKTheGoodTheBadandTheUnbalanced/Figures/figs/

> fig1_retrained.png, fig2_retrained.png, ..., fig13_provided.png
>
> table1_retrained.txt, table2_retrained.txt, ..., table4_retrained.txt

  
### STEP 3.2 From pretrained models and recalculated scores files (Step 2.2)  


Generate paper figures and tables only:

    sh Figures/gen_paper_figs.sh pretrained

Generate paper and appendix (companion website) figures and tables:

    sh Figures/gen_paper-appendix_figs.sh pretrained

 Outputs figures and tables to SoKTheGoodTheBadandTheUnbalanced/Figures/figs/

> fig1_pretrained.png, fig2_pretrained.png, ..., fig13_pprovided.png
>
> table3_pretrained.txt, table4_pretrained.txt, ..., table5_pretrained.txt


### STEP 3.3 From provided scores files  


Generate paper figures and tables only:

    sh Figures/gen_paper_figs.sh provided

Generate paper and appendix (companion website) figures and tables:

    sh Figures/gen_paper-appendix_figs.sh provided

 Outputs figures and tables to SoKTheGoodTheBadandTheUnbalanced/Figures/figs/

> fig1_provided.png, fig2_provided.png, ..., fig13_provided.png
>
> table3_provided.txt, table4_provided.txt, ..., table5_provided.txt

</details></p>