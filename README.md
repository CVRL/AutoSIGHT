# AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise

Official repository for IEEE Vision Languages and Human Centric Computing (VL/HCC 2025) paper: **IEEEXplore | [ArXiv](https://arxiv.org/abs/2508.01015)**

## AutoSIGHT Architecture Overview

<p align="center">
  <img src="Image_Assets/network_diagram.png" width="500" />
</p>

## Abstract
> Can we teach machines to assess the expertise of humans solving visual tasks automatically based on eye tracking features? This paper proposes AutoSIGHT, Automatic System for Immediate Grading of Human experTise, that classifies expert and non-expert performers, and builds upon an ensemble of features extracted from eye tracking data while the performers were solving a visual task. Results on the task of iris Presentation Attack Detection (PAD) used for this study show that with a small evaluation window of just 5 seconds, AutoSIGHT achieves an average average Area Under the ROC curve performance of 0.751 in subject-disjoint train-test regime, indicating that such detection is viable. Furthermore, when a larger evaluation window of up to 30 seconds is available, the Area Under the ROC curve (AUROC) increases to 0.8306, indicating the model is effectively leveraging more information at a cost of slightly delayed decisions. This work opens new areas of research on how to incorporate the automatic weighing of human and machine expertise into human-AI pairing setups, which need to react dynamically to nonstationary expertise distribution between the human and AI players (e.g. when the experts need to be replaced, or the task at hand changes rapidly). Along with this paper, we offer the eye tracking data used in this study collected from 6 experts and 53 non-experts solving iris PAD visual task.

## Data Sample
Example JSON Object
```    {
        "participant": "2024-136-068",
        "biometricStatus": "Expert",
        "batch": 14,
        "slideNumber": "1",
        "irisImageLink": "/fbi_images_unblurred/77_05052d288.png",
        "heatmapFilename": "E_068_77_05052d288_Heatmap.npy",
        "attackType": "Real Iris",
        "initial": "Normal",
        "final": "Normal",
        "imageStartIndex": "441",
        "initialDecisionIndex": "675",
        "verbalPhaseCompleteIndex": "1097",
        "finalDecisionIndex": "1099",
        "fixationTimes_InitialPhase": [],
        "fixationCount_Initial": 77,
        "averageFixationTime_Initial_MS": 127.08,
        "GRI_Initial": 0.165,
        "fixationCount_Verbal+Final": 80,
        "averageFixationTime_Verbal+Final_MS": 124.73,
        "fixationCount_Cumulative": 157,
        "averageFixationTime_Cumulative_MS": 125.88,
        "sequences_5_Second": [],
        "sequences_10_Second": [],
        "sequences_15_Second": [],
        "sequences_20_Second": [],
        "sequences_30_Second": []
    }
```


## Citation
```
@article{dowling2025autosight,
  title={AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise},
  author={Dowling, Byron and Probcin, Jozef and Czajka, Adam},
  journal={arXiv preprint arXiv:2508.01015},
  year={2025}
}
```

## Acknowledgments

1. This work was supported by the U.S. Department of Defense (Contract No. W52P1J-20-9-3009). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the U.S. Department of Defense or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation here on
