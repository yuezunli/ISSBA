# Invisible Backdoor Attack with Sample-Specific Triggers

## Environment
This project is developed with Python 3.6 on Ubuntu 18.04. Please run the following script to install the required packages
```shell
pip install -r requirements.txt
```

## Demo
Before running the code, please download the checkpoints from [Baidudisk](https://pan.baidu.com/s/1m5yRFQ4Wt7Km_56CIxzgsg) (code:o89z), and put them into `ckpt` folder.

1. Generating poisoned sample with sample-specific trigger. 
    ```python
    # TensorFlow
    python encode_image.py \
    --model_path=ckpt/encoder_imagenet \
    --image_path=data/imagenet/org/n01770393_12386.JPEG \
    --out_dir=data/imagenet/bd/ 
    ```

    | ![](data/imagenet/org/n01770393_12386.JPEG) | ![](data/imagenet/bd/n01770393_12386_hidden.png) | ![](data/imagenet/bd/n01770393_12386_residual.png)
    |:--:| :--:| :--:| 
    | Benign image | Backdoor image | Trigger |

2. Runing `test.py` for testing benign and poisoned images.
    ```python
    # PyTorch
    python test.py
    ```

## Train
1. Download data from [Baidudisk](https://pan.baidu.com/s/1p_t5EJ91hkiyeYBFEZyfsg 
)(code:oxgb) and unzip it to folder `datasets/`.
2. Run training script `bash train.sh`.
3. The files in checkpoint folder are as following:

    ```
    --- args.json # Input arguments
    |-- x_checkpoint.pth.tar # checkpoint
    |-- x_model_best.pth.tar # best checkpoint
    |-- x.txt # log file
    ```

## Defense
Check [BackdoorBench](https://github.com/SCLBD/backdoorbench) for details
## Citation
Please cite our paper in your publications if it helps your research:

```
@inproceedings{li_ISSBA_2021,
  title={Invisible Backdoor Attack with Sample-Specific Triggers},
  author={Li, Yuezun and Li, Yiming and Wu, Baoyuan and Li, Longkang and He, Ran and Lyu, Siwei},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Notice
This repository is NOT for commecial use. It is provided "as it is" and we are not responsible for any subsequence of using this code.
