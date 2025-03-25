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