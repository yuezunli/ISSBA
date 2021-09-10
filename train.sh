data_dir=poject_dir/datasets/sub-imagenet-200
bd_data_dir=poject_dir/datasets/sub-imagenet-200-bd/inject_a/
model=res18
bd_ratio=0.1
train_batch=128
bd_label=0
bd_char=a


python train.py \
--net=$model \
--train_batch=$train_batch \
--workers=4 \
--epochs=25 \
--schedule 15 20 \
--bd_label=0 \
--bd_ratio=$bd_ratio \
--data_dir=$data_dir \
--bd_data_dir=$bd_data_dir \
--checkpoint=ckpt/bd/${model}_bd_ratio_${bd_ratio}_inject_${bd_char} 