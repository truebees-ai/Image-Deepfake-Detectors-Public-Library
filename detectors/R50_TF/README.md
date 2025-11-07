# TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks

This is the official code implementation of the "IJCNN 2025" paper ["TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks"](https://arxiv.org/abs/2504.20658)\
Authors: Stefano Dell'Anna, Andrea Montibeller, Giulia Boato 

## Abstract

<table>
<tbody>
<tr>
<td style="width:30%; vertical-align: top;"> AI-generated synthetic media are increasingly used in real-world scenarios, often with the purpose of spreading misinformation and propaganda through social media platforms, where compression and other processing can degrade fake detection cues. Currently, many forensic tools fail to account for these in-the-wild challenges. In this work, we introduce TrueFake, a large-scale benchmarking dataset of 600,000 images including top notch generative techniques and sharing via three different social networks. This dataset allows for rigorous evaluation of state-of-the-art fake image detectors under very realistic and challenging conditions. Through extensive experimentation, we analyze how social media sharing impacts detection performance, and identify current most effective detection and training strategies. Our findings highlight the need for evaluating forensic models in conditions that mirror real-world use.</td>
<td style="width:70%; vertical-align: top;"> <img src="./images/visual_abstract.png" alt="alt text" width="1000"/> ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀</td>
</tr>
</tbody>
</table>

# Detector

## Use pretrained network
The R50-E2P network was trained according to the information provided in the paper on a limited subset of non-shared data.\
If you plan on using the network for other purposes other than to replicate the results in the paper, you should re-train it on a more comprehensive subset.

# Results
Check [TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks](https://arxiv.org/abs/2504.20658)\

<table>
<tbody>
<tr>
<td style="width:50%; vertical-align: top;"> <img src="./images/non-shared.png" alt="alt text" width="400"/> </td>
<td style="width:50%; vertical-align: top;"> <img src="./images/shared.png" alt="alt text" width="400"/> </td>
</tr>
</tbody>
</table>

# Please Cite
If you use this material please cite:

```
@misc{dellanna2025truefake,
      title={TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks}, 
      author={Stefano Dell'Anna and Andrea Montibeller and Giulia Boato},
      year={2025},
      eprint={2504.20658},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2504.20658}, 
}
```