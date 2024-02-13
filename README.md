# LLaMA-Adapter 모델에 대한 코드 리뷰를 하기 위해 fork한 repository

Official implementation of ['LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention'](https://arxiv.org/pdf/2303.16199.pdf) and ['LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model'](https://arxiv.org/pdf/2304.15010.pdf).

This repo proposes **LLaMA-Adapter (V2)**, a lightweight adaption method for fine-tuning **Instruction-following** and **Multi-modal** [LLaMA](https://github.com/facebookresearch/llama) models 🔥.


## Released Models 

| Name                                                         | Approach                                               | Data                                                         | Modality                           | Visual         | Text                  |
| ------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------- | -------------- | --------------------- |
| [LLaMA-Adapter V1](./alpaca_finetuning_v1)                   | prefix, gate                                           | Alpaca                                                       | Text                               | ×              | LLaMA-7B              |
| [LLaMA-Adapter V2 dialog](./llama_adapter_v2_chat65b)        | scale, bias, norm                                      | ShareGPT                                                     | Text                               | ×              | LLaMA-65B             |
| [LLaMA-Adapter V2 multimodal](./llama_adapter_v2_multimodal7b) | [P] prefix, projection, gate <br />[F] bias, norm      | [P] Image-Text-V1<br />[F] GPT4LLM, LLaVA                    | Image&Text                         | CLIP-ViT-L/14  | LLaMA-7B              |
| [LLaMA-Adapter V2.1 multimodal](./llama_adapter_v2_multimodal7b) | [P] prefix, projection, gate <br />[F] bias, norm, lora      | [P] Image-Text-V1<br />[F] GPT4LLM, LLaVA, VQAv2                    | Image&Text                         | CLIP-ViT-L/14  | LLaMA-7B              |
| [ImageBind-LLM](./imagebind_LLM)                             | [P] prefix, projection, gate<br />[F] bias, norm, lora | [P] Image-Text-V1<br />[F] [Instruction Following](https://github.com/OpenGVLab/LLaMA-Adapter/blob/main/imagebind_LLM/docs/train.md#data-1) | ImageBind Modalities + Point Cloud | imagebind_huge | Open-Chinese-LLaMA-7B |
| ImageBind-dialog                                             | [P] prefix, projection, gate<br />[F] bias, norm, lora | [P] Image-Text-V1<br />[F] LLaVA, ShareGPT                   | ImageBind Modalities + Point Cloud | imagebind_huge | Open-Chinese-LLaMA-7B |

+ [P] means **P**re-train and [F] means **F**ine-tune
+ **Image-Text-V1** is  a concatenation of LAION400M, COYO, MMC4, SBU, Conceptual Captions, and COCO
+ **ImageBind Modalities** include image, video, text, audio, depth, thermal, IMU
+ **ImageBind-dialog** will be release soon


## Overview
Efficiency Comparison:
|  Model | Parameters | Storage Space | Training Time  
| :-----: | :-----: |:-----:| :-----: |
|  [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 7B |13G| 3 Hours|
|  LLaMA-Adapter | 1.2M |4.7M| 1 Hour|

By inserting adapters into LLaMA's transformer, our method only introduces **1.2M** learnable parameters, and turns a LLaMA into an instruction-following model within **1 hour**. For stablizing training at early stages, we propose a novel **Zero-init Attention** with zero gating mechanism to adaptively incorporate the instructional signals. After fine-tuning, LLaMA-Adapter can generate high-quality instruction-following sentences, comparable to the fully fine-tuned [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [Alpaca-Lora](https://github.com/tloen/alpaca-lora).

<div align="center">
  <img src="docs/pipeline.png"/ width="90%">
</div>

Our approach can be simply extended to **Multi-modal Input Instructions**. The reasoning framework of image-conditioned LLaMA-Adapter for [ScienceQA](https://scienceqa.github.io/) is as follows, which is also shared by other modalities, such as audio and video.

<div align="center">
  <img src="docs/multimodal.png"/ width="90%">
</div>




## Setup

Here is a from-scratch script for **LLaMA-Adapter V1**.
```bash
conda create -n llama_adapter -y python=3.8
conda activate llama_adapter

# install pytorch
conda install pytorch cudatoolkit -c pytorch -y

# install dependency and llama-adapter
pip install -r requirements.txt
pip install -e .
```
**Note**: **To setup other models**, please refer to [llama_adapter_v2_chat65b](llama_adapter_v2_chat65b), [llama_adapter_v2_multimodal7b](llama_adapter_v2_multimodal7b) and [imagebind_LLM](imagebind_LLM) for more details.

## Inference

Please request access to the pre-trained LLaMA from [this form](https://forms.gle/jk851eBVbX1m5TAv5) (official) or download the LLaMA-7B from [Hugging Face](https://huggingface.co/nyanko7/LLaMA-7B/tree/main) (unofficial). Then, obtain the weights of our LLaMA-Adapter from [here](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth). We denote the path to the downloaded weights of LLaMA and adapters as `TARGET_FOLDER` and `ADAPTER_PATH`.

Here is an example to generate instruction-following sentences with 7B LLaMA model and our LLaMA-Adapter:
```bash
torchrun --nproc_per_node 1 example.py \
         --ckpt_dir $TARGET_FOLDER/model_size\
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH
```

## Training

We release the simple fine-tuning code of LLaMA-Adapter on LLaMA-7B model at [here](alpaca_finetuning_v1), which is for effortless reproduction with minimal dependencies. We will soon release the fine-tuning code for LLaMA-65B and multi-model LLaMA-Adapter.

Please download the 52K instruction-following training [data](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json) from Standford Alpaca, and put it under `DATA_PATH`. Then run:

```bash
cd alpaca_finetuning_v1

torchrun --nproc_per_node 8 finetuning.py \
         --model Llama7B_adapter \
         --llama_model_path $TARGET_FOLDER/ \
         --data_path $DATA_PATH/alpaca_data.json \
         --adapter_layer 30 \
         --adapter_len 10 \
         --max_seq_len 512 \
         --batch_size 4 \
         --epochs 5 \
         --warmup_epochs 2 \
         --blr 9e-3 \
         --weight_decay 0.02 \
         --output_dir ./checkpoint/
```

## Citation
If you find our LLaMA-Adapter code and paper useful, please kindly cite:
```bash
@article{zhang2023llamaadapter,
  title = {LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention},
  author={Zhang, Renrui and Han, Jiaming and Liu, Chris and Gao, Peng and Zhou, Aojun and Hu, Xiangfei and Yan, Shilin and Lu, Pan and Li, Hongsheng and Qiao, Yu},
  journal={arXiv preprint arXiv:2303.16199},
  year={2023}
}
```

If you find our LLaMA-Adapter V2 code and paper useful, please kindly cite:
```bash
@article{gao2023llamaadapterv2,
  title = {LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model},
  author={Gao, Peng and Han, Jiaming and Zhang, Renrui and Lin, Ziyi and Geng, Shijie and Zhou, Aojun and Zhang, Wei and Lu, Pan and He, Conghui and Yue, Xiangyu and Li, Hongsheng and Qiao, Yu},
  journal={arXiv preprint arXiv:2304.15010},
  year={2023}
}
```

## Acknowledgement
This repo benefits from [LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), and [Alpaca-Lora](https://github.com/tloen/alpaca-lora). Thanks for their wonderful works.
