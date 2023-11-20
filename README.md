# Learning Domain-Aware Detection Head with Prompt Tuning

The official implementation of `Learning Domain-Aware Detection Head with Prompt Tuning` ([arxiv](https://arxiv.org/abs/2306.05718)).

This codebase is based on [RegionCLIP](https://github.com/microsoft/RegionCLIP).

1. Put your dataset at './datasets/your_dataset'. Please follow the format of Pascal Voc.
For example:
- dataset
  - cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages
  - foggy_cityscapes_voc
    - VOC2007
      - Annotations
      - ImageSets
      - JPEGImages

2. Put your pre-trained VLM model at somewhere you like, for example, './ckpt', and edit the MODEL.WEIGHTS in train_da_pro_c2f.sh.

3. Following RegionCLIP, generate class embedding and put it at somewhere you like, and edit the MODEL.CLIP.TEXT_EMB_PATH.

4. Training: train_da_pro_c2f.sh  Testing: test_da_pro_c2f.sh
Training is customizable. You can directly use the parameters of other VLMs as backbone and then adjust only domain-adaptive prompt. You can also follow the steps of Regionclip to customize a backbone on your own dataset, then conduct adaptation.


A training sample: 
1) Initial pre-trained model with VLM (like CLIP or RegionCLIP).
2) Set LEARNABLE_PROMPT.TUNING to False to fine-tune the pre-trained backbone with domain adversarial loss.
3) Set LEARNABLE_PROMPT.TUNING to True to freeze the backbone and tune a learnable domain-adaptive prompt on two domains. 
