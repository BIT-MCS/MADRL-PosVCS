# MADRL-PosVCS

## :page_facing_up: Description
Indoor localization is drawing more and more attentions due to the growing demand of various location-based services, where fingerprinting is a popular data driven techniques that does not rely on complex measurement equipment, yet it requires site surveys which is both labor-intensive and time-consuming.

Vehicular crowdsensing (VCS) with unmanned vehicles (UVs) is a novel paradigm to navigate a group of UVs to collect sensory data from certain point-of-interests periodically (PoIs, i.e., coverage holes in localization scenarios). 

In this paper, we formulate the multi-floor indoor fingerprint collection task with periodical PoI coverage requirements as a constrained optimization problem. Then, we propose a multi-agent deep reinforcement learning (MADRL) based solution, ``MADRL-PosVCS'', which consists of a primal-dual framework to transform the above optimization problem into the unconstrained duality, with adjustable Lagrangian multipliers to ensure periodic fingerprint collection. We also propose a novel intrinsic reward mechanism consists of the mutual information between a UV's observations and environment transition probability parameterized by a Bayesian Neural Network (BNN) for exploration, and a elevator-based reward to allow UVs to go cross different floors for collaborative fingerprint collections. 

## :wrench: Installation
1. Clone repo
    ```bash
    git clone https://github.com/BIT-MCS/MADRL-PosVCS
    cd MADRL-PosVCS
    ```
2. Install dependent packages
    ```sh
    conda create --name PosVCS python==3.9
    conda activate PosVCS
    python -m pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -r requirements.txt
    ```
   You might need to adjust your Torch version according to your GPU and operating system, or you can opt to use the CPU instead.
## :computer: Training

Train our solution
```bash
python ./MAPPO/train/train.py
```

## :checkered_flag: Testing

Test with the trained models 

```sh
python ./MAPPO/train/train.py --model_dir your_model_path --generate_outputs True
```
You can also view your trained trajectories through the command below, followed by the previous command.
```sh
python ./MAPPO/env_render.py
```

## :clap: code_Reference
- https://github.com/tinyzqh/light_mappo


## :e-mail: Contact

If you have any question, please email `kiwichoy@163.com`.


