<<<<<<< HEAD
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
=======
# DBCNet
# a dual-branch crack segmentation net
If the above manuscript is accepted, the code and data to the manuscript will be publicly available for free.
>>>>>>> 1c2d4038b5a3b81f8ebab2161cc6f5b02b88fe6b
