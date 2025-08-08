# AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise

Official repository for IEEE Vision Language / Human Centered Computing (VL/HCC 2025) paper: **IEEEXplore | [ArXiv](https://arxiv.org/abs/2508.01015)**

## AutoSIGHT Architecture Overview

<p align="center">
  <img src="Image_Assets/network_diagram.png" width="500" />
</p>

### Abstract
> Can we teach machines to assess the expertise of humans solving visual tasks automatically based on eye tracking features? This paper proposes AutoSIGHT, Automatic System for Immediate Grading of Human experTise, that classifies expert and non-expert performers, and builds upon an ensemble of features extracted from eye tracking data while the performers were solving a visual task. Results on the task of iris Presentation Attack Detection (PAD) used for this study show that with a small evaluation window of just 5 seconds, AutoSIGHT achieves an average average Area Under the ROC curve performance of 0.751 in subject-disjoint train-test regime, indicating that such detection is viable. Furthermore, when a larger evaluation window of up to 30 seconds is available, the Area Under the ROC curve (AUROC) increases to 0.8306, indicating the model is effectively leveraging more information at a cost of slightly delayed decisions. This work opens new areas of research on how to incorporate the automatic weighing of human and machine expertise into human-AI pairing setups, which need to react dynamically to nonstationary expertise distribution between the human and AI players (e.g. when the experts need to be replaced, or the task at hand changes rapidly). Along with this paper, we offer the eye tracking data used in this study collected from 6 experts and 53 non-experts solving iris PAD visual task.


## Citation
'''
@article{dowling2025autosight,
  title={AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise},
  author={Dowling, Byron and Probcin, Jozef and Czajka, Adam},
  journal={arXiv preprint arXiv:2508.01015},
  year={2025}
}
'''
