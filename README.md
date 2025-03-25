You can complete the training on the Deepcrack dataset by following the steps below.
1. install torch
    ```shell
    conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
2. install dependent libraries
    ```shell
    pip install -r requirements.txt
    ```
3. install Mamba
    ```shell
    cd models/encoders/selective_scan && pip install . && cd ../../..
    ```
4. train
    ```shell
    python train.py
    ```
If you have used our code, please cite our paper.
```bibtex
@article{ZHANG2025110536,
title = {Dual-branch crack segmentation network with multi-shape kernel based on convolutional neural network and Mamba},
journal = {Engineering Applications of Artificial Intelligence},
volume = {150},
pages = {110536},
year = {2025},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2025.110536},
url = {https://www.sciencedirect.com/science/article/pii/S0952197625005366},
author = {Jianming Zhang and Dianwen Li and Zhigao Zeng and Rui Zhang and Jin Wang}
}
 ```
