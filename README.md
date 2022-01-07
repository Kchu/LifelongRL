# Python implementation of Lifelong Reinforcement Learning (Lifelong RL/LLRL). 

# SR-LLRL
Shaping Rewards for LifeLong Reinforcement Learning

## Brief Introduction
Codes for experimenting with proposed approaches to Lifelong RL, attached to our 2021 IEEE SMC paper "Accelerating lifelong reinforcement learning via reshaping rewards".

Authors: Kun Chu, Xianchao Zhu, William Zhu.

If you use these codes, please **cite our paper** as

K. Chu, X. Zhu and W. Zhu, "[Accelerating Lifelong Reinforcement Learning via Reshaping Rewards](https://ieeexplore.ieee.org/document/9659064)*," 2021 IEEE International Conference on Systems, Man, and Cybernetics (SMC), 2021, pp. 619-624, doi: 10.1109/SMC52423.2021.9659064.

## Usage
To generate experiemental results, run main.py; 

To draw all of our plots, run result_show_task.py and result_show_episode.py. 

Note that you must choose your learning algorithms or parameters inside the code to generate results/figures. 

## Important Note
These codes need to import some libraries of python, especially [simple_rl](https://github.com/david-abel/simple_rl) provided by [David Abel](https://github.com/david-abel). However, please note that I have made some improvements and changes based on his codes, so please download the simple_rl inside the fold directly instead of installing from the python official libraries.

## Experimental Demonstration
![png1](https://github.com/Kchu/LifelongRL/blob/master/SR-LLRL/IEEE_SMC_2021_Plots/figures/Environments.png)
![png2](https://github.com/Kchu/LifelongRL/blob/master/SR-LLRL/IEEE_SMC_2021_Plots/figures/Result_1.png)
![png3](https://github.com/Kchu/LifelongRL/blob/master/SR-LLRL/IEEE_SMC_2021_Plots/figures/Result_2.png)

## Acknowledgment

Here I want to sincerely thank [David Abel](https://david-abel.github.io/), a great young scientist. He generously shared the source code of his paper in [Github](https://github.com/david-abel/transfer_rl_icml_2018) and gave detailed answers to any of my questions/doubts in the process of conducting this research. I admire his academic achievements, and more importantly, his enthusiastic help and scientific spirit.

# Last

Feel free to contact me (kun_chu@outlook.com) with any questions.