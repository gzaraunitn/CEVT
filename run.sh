#!/bin/bash

#====== parameters ======#
training=true # true | false
testing=false # true | false
modality=RGB
frame_type=feature # frame | feature
num_segments=16 # sample frame # of each video for training
test_segments=16
baseline_type=video
frame_aggregation=avgpool # method to integrate the frame-level features (avgpool | trn | trn-m | rnn | temconv)
add_fc=1
fc_dim=512
arch=resnet101
use_target=uSv # none | Sv | uSv
share_params=Y # Y | N


exp_DA_name=baseline


#====== select dataset ======#
#path_data_root=/media/bigdata/uqyluo/MM2020Data/ # depend on users
#path_exp_root=action/ # depend on users

pretrained=none

#====== parameters for algorithms ======#
# parameters for DA approaches
dis_DA=none # none * | DAN | JAN
alpha=0 # depend on users

adv_pos_0=N # Y* | N (discriminator for relation features)
adv_DA=none # none | RevGrad *
beta_0=1 # U->H: 0.75 | H->U: 1
beta_1=0.75 # U->H: 0.75 | H->U: 0.75
beta_2=0.5 # U->H: 0.5 | H->U: 0.5

use_attn=none # none | TransAttn * | general
n_attn=1
use_attn_frame=none # none | TransAttn | general

use_bn=none # none | AdaBN | AutoDIAL
add_loss_DA=none # none | target_entropy | attentive_entropy *
gamma=0.3 # U->H: 0.003 | H->U: 0.3

ens_DA=none # none | MCD
mu=0

# parameters for architectures
bS=8 # batch size
bS_2=8

lr=3e-2
optimizer=SGD

#exp_path=$path_exp'-'$optimizer'-share_params_'$share_params'-lr_'$lr'-bS_'$bS'_'$bS_2'/'$dataset'-'$num_segments'seg-disDA_'$dis_DA'-alpha_'$alpha'-advDA_'$adv_DA'-beta_'$beta_0'_'$beta_1'_'$beta_2'-useBN_'$use_bn'-addlossDA_'$add_loss_DA'-gamma_'$gamma'-ensDA_'$ens_DA'-mu_'$mu'-useAttn_'$use_attn'-n_attn_'$n_attn'/'
val_segments=16

# parameters for optimization
lr_decay=10
lr_adaptive=dann # none | loss | dann
lr_steps_1=10
lr_steps_2=20
epochs=60
gd=20

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py \
	--epic_kitchens \
	--run_name ek24_cevt \
	data/classInd_ek.txt \
	$modality \
	txt/epic_kitchens3/P02_train_source.txt \
	txt/epic_kitchens3/P04_train_target.txt \
	txt/epic_kitchens3/P02_test_target.txt \
	--exp_path /exp \
	--arch $arch \
	--pretrained $pretrained \
	--baseline_type $baseline_type \
	--frame_aggregation $frame_aggregation \
	--num_segments $num_segments \
	--val_segments $val_segments \
	--add_fc $add_fc \
	--fc_dim $fc_dim \
	--dropout_i 0.5 \
	--dropout_v 0.5 \
	--use_target $use_target \
	--share_params $share_params \
	--dis_DA $dis_DA \
	--alpha $alpha \
	--place_dis N Y N \
	--adv_DA $adv_DA \
	--beta $beta_0 $beta_1 $beta_2 \
	--place_adv $adv_pos_0 Y Y \
	--use_bn $use_bn \
	--add_loss_DA $add_loss_DA \
	--gamma $gamma \
	--ens_DA $ens_DA \
	--mu $mu \
	--use_attn $use_attn \
	--n_attn $n_attn \
	--use_attn_frame $use_attn_frame \
	--gd $gd \
	--lr $lr \
	--lr_decay $lr_decay \
	--lr_adaptive $lr_adaptive \
	--lr_steps $lr_steps_1 $lr_steps_2 \
	--epochs $epochs \
	--optimizer $optimizer \
	--n_rnn 1 \
	--rnn_cell LSTM \
	--n_directions 1 \
	--n_ts 5 \
	-b $bS $bS_2 $bS \
	-j 4 \
	-ef 1 \
	-pf 50 \
	-sf 50 \
	--copy_list N N