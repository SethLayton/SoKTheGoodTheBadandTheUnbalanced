#!/bin/bash


############## Download test datasets ##############
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



############## Download pretrained models ##############


cd ../SoKTheGoodTheBadandTheUnbalanced
 
wget "https://zenodo.org/records/12007844/files/asvspoof_rawnet2model_pretrained.tar.gz"
mkdir -p ASVspoof/RawNet2/models && tar -xvzf asvspoof_rawnet2model_pretrained.tar.gz -C ASVspoof/RawNet2/models

wget "https://zenodo.org/records/12007844/files/asvspoof_wav2vecmodel_pretrained.tar.gz"
mkdir -p ASVspoof/wav2vec/models && tar -xvzf asvspoof_wav2vecmodel_pretrained.tar.gz -C ASVspoof/wav2vec/models

wget "https://zenodo.org/records/12007844/files/asvspoof_lfcclcnnmodel_pretrained.tar.gz"
mkdir -p ASVspoof/LFCC-LCNN/models && tar -xvzf asvspoof_lfcclcnnmodel_pretrained.tar.gz -C ASVspoof/LFCC-LCNN/models

wget "https://zenodo.org/records/12089727/files/cfad_rawnet2model_pretrained.tar.gz"
mkdir -p CFAD/RawNet2/models && tar -xvzf cfad_rawnet2model_pretrained.tar.gz -C CFAD/RawNet2/models

wget "https://zenodo.org/records/12089727/files/cfad_lfcclcnnmodel_pretrained.tar.gz"
mkdir -p CFAD/LFCC-LCNN/models && tar -xvzf cfad_lfcclcnnmodel_pretrained.tar.gz -C CFAD/LFCC-LCNN/models

wget "https://zenodo.org/records/12090252/files/cifake_rawcnnmodel_pretrained.tar.gz"
mkdir -p CIFAKE/CIFAKE/models && tar -xvzf cifake_rawcnnmodel_pretrained.tar.gz -C CIFAKE/CIFAKE/models

cd ../DataSets
wget "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"