---
layout: post
title:  "Paper review: Predicting Deeper into the Future of Semantic Segmentation"
date:   2020-05-23 23:02:44 +0100
author: Maximilian Bömer
categories: review
show_sidebar: true
hero_image: 'https://media.arxiv-vanity.com/render-output/2991125/x1.png'
image: 'https://media.arxiv-vanity.com/render-output/2991125/x1.png'
published: true
comments: true
excerpt_separator: <!--more-->

---
# Paper review - Predicting Deeper into the Future of Semantic Segmentation

#### Introduction

In robotics it is crucial to correctly interpret the agent’s environment for good decision making. Besides the semantics and geometry of a scene, it is important to understand the scene dynamics. The here analyzed paper *Predicting Deeper into the Future of Semantic Segmentation* [1] focuses on this task and generates segmentation maps for not yet seen video frames based on a given input sequence (see Fig. 1). This blog post discusses the contributions and critique of the limitations of the work described. Additionally the following questions are answered: How the authors worked to overcome or mitigate any limitations and how does the work fit in the wider body of literature? In the end the scientific methodology and experimental results are evaluated and an outlook on the potential use for autonomous driving is given.
<!--more-->
<figure align="center">
  <img img src="/assets/predsem_img/title.png" alt="fcn_architecture" style="zoom:60%;" class="center"/>
  <figcaption>Fig. 1: Semantic-level scene dynamics to predict semantic segmentations of unobserved future frames given several past frames [1].</figcaption>
</figure>
<div style="text-align: justify">

#### Main contributions of paper

##### Introduction of the task of predicting semantic segmentation maps for future frames in a video sequence

Previous works in video forecasting either focus on predicting raw RGB intensities [2] of future frames or avoid the high dimensional pixel space by focusing on semantic properties. For this different forms of abstractions like directly predicting frame transformations [3], visual representations [4] or feature point trajectories [5] are used. Pauline Luc et al. relax the problem by focusing on ”future high-level scene properties” and therefore introduce the new task of directly predicting semantic segmentation maps. The underlying idea is that the capacity of the model is better allocated by focusing on the dense semantic information of the scene, than on the RGB values, which are not required for many applications. 

##### Development of an autoregressive model, which enables stable mid-term predictions 

For single-frame prediction a model is presented, which takes N semantic segmentation maps as input and directly predicts the segmentation for the next timestep t+1. The input and target maps get precomputed using a Dilation10 network [6]. It is shown that the models, which just use segmentations as inputs and targets outperform these, which additionally use or predict RGB values or just focus on the prediction of raw pixel intensities. The architecture of the described S2S model is a multi-scale network like in [2], with two spatial scales. It predicts a coarse global output for a downsampled input map and uses the upsampled output of the first module as well as the correct scaled input frames to refine this prediction in the second module. Instead of using the class values of the semantic segmentation maps as inputs and targets, the final softmax layer’s pre-activations are used as they have shown to contain more information [7] (see Fig. 2).

<figure align="center">
  <img img src="/assets/predsem_img/architecture.png" alt="fcn_architecture" style="zoom:45%;" class="center"/>
  <img img src="/assets/predsem_img/batchVSautoregressive.png" alt="fcn_architecture" style="zoom:55%;" class="center"/>
  <figcaption>Fig. 2: Multi-scale architecture of the S2S model that predicts the semantic segmentation of the next frame given the segmentation maps of the NI previous frames. The autoregressive model (top right) shares parameters over time, in contrast to the batch model (bottom right); dependency links are colored accordingly. [1]</figcaption>
</figure>

To predict more than one time step in the future Pauline Luc et al. propose an autoregressive model based on the described architecture (see Fig. 2). The predicted output for timestep t gets used as input in the next iteration and the next frames are predicted in an iterative manner. It is shown that this approach, which exploits the recurrent structure of the problem, outperforms the variant, which based on the input at t predicts a batch of all required future frames. For training the network, losses with different properties are evaluated:

- L1-loss: Matches pixel/pre-activation predictions
- Gradient difference loss (GDL): Matches regional differences (e.g. on contours)
- Adversial loss based on EM distance: Allows different turns of events as they do not need to match the oracle but just require to fool the discriminator
- Autoregressive fine-tuning: Backpropagation through time to account for error propagation in autoregressive setup

It is shown that the best variant for mid-term prediction is trained with L1, GDL and autoregressive fine-tuning on the softmax pre-activations. The final model is able to outperform the two baselines (I) copying the input frame to output and II) a warped form of the input frame based on estimated optical flow) for short-term and mid-term predictions (up to 0.5 s). Long term predictions (up to 10 s) show much worse results than I). 

#### Classification in the broader spectrum of literature

Predicting scene dynamics is a widely studied topic. As described in II-A) there are several ideas to reduce the complexity by concentrating on high-level scene properties instead of modeling the raw pixel intensities. Here we focus on two streams of research, which predict future scenes at a semantically meaningful level: Motion forecasting for objects and segmentations.

#####  Object forecasting 

Approaches, especially in autonomous driving, mainly use a three step approach: Detection, tracking and motion forecasting. Tracking systems identify tracks of existing objects in physically meaningful 3D world coordinates, like birds eye view, by exploiting depth information with their sensor setup [8], or in the 2D image plane by using position and object scale [9]. Herefore often extended Kalman filters are used to update the non-linear motion models based on the past measurements, which use the results of the respective object detectors [8]. Based on the identified tracks, the future trajectories are then extrapolated. This can be done by using linear models, which use the motion model of the Kalman filter, behavioral models [10], interacting Gaussian processes [11] or LSTMs [12]. Instead of learning these modules independently, approaches like Fast and Furious [13] or IntentNet [14] try to directly exploit spatio-temporal 3D information, which enables endto-end learning and a better propagation of uncertainties.

##### Segmentation forecasting

The approach analysed in this report [1] falls within this research area, where the goal is not to focus on selected objects but moreover predicting the whole semantic evolution of the scene. This enables a more holistic form of visual scene understanding. Several publications build upon the here done ground work. P. Luc et al. [15] modify the problem statement in their follow up work by predicting future instance segmentations to overcome the problem that semantic segmentations do not account for individual objects. They realise their approach by predicting the highest level features of the used Mask R-CNN for future frames. These two works get fused in [16] by encoding instance and semantic segmentation information in a single representation using distance maps. Other ideas to improve the prediction of future semantic segmentations include jointly predicting scene parsing and optical flow for future frames [17]. Due to the synergy of both tasks this approach is able to reveal significant better results. Further improvements got achieved by using a bidirectional ConvLSTM, which helps to encode the temporal relation of the input frames [18]. An other extension is proposed in [19], where the authors try to capture epistemic and aleatoric uncertainty in the predicted segmentations by at the same time encouraging diversity in the predicted multi-modal futures. They introduce a new Bayesian inference framework, which goes beyond the often used log-likelihood estimate, which enforces models to predict the mean of possible scenarios. 

#### Discussion

This section has the goal to critically discuss limitations of the presented work and show how the authors mitigated some of them. The focus here is to show, which properties are required to extract meaningful representations of visual scenes, which enable predictions of the future and build a strong basis for decision making, and if the presented model is able to capture them. 

##### Limitations of the work

One of the key limitations of the proposed work is that the system is not explicitly aware of the 3D geometry of the scene as the here proposed input is a projection into 2D space. Furthermore no notion of instances is available. These restrictions have direct impact on the performance and can be seen as reason for the reported poor performance of the model, when handling occlusions. Another weakness is the inability to deal with long term dependencies. The cause for this can be found in the architecture as the model takes four input frames and is therefore limited exploiting past information. Furthermore no pass-through of hidden information like in recurrent networks is possible. The inability to predict multi-modal futures for constant inputs and estimating the uncertainty of events is another restriction. This could have been mitigated by using the presented adversial loss during training and choosing a sampling scheme like in Bayesian inference or using a GAN or VAE like model to get a meaningful probability distribution. The performance of the prediction is also constrained as no ground truth data is available and the targets propagate the errors of the preceding segmentation network.

##### Mitigated limitations

The use of the semantic segmentation model as oracle reduces on the other hand the dependency on costly dense video annotations, which makes it easy to scale the approach to large datasets. Another mitigated limitation is the extension of the task from forecasting trajectories of selected objects to semantic segmentation, which seems to be a more complete form of visual scene understanding. Additionally the autoregressive architecture enables an unlimited view into the future as the model can theoretically predict scenes of arbitrary length. 

##### Evaluation of the scientific methodology and experimental results

The paper impresses by its constant bottom up approach, where initially several options are presented, then compared and in the end a comprehensible decision is made. Examples are the comparisons of the different model architectures X2X, XS2X, XS2S, XS2XS and S2S as well as the decision for the autoregressive and against the batch approach to predict semantic segmentation maps for multiple future time steps. Even though the prediction of semantic segmentation maps was a novel task at the time of publication, it would be interesting to see how other video forecasting approaches like [2] would have performed in comparison to the presented X2X models. On a side note it is nice to see that the used models were published for reproducibility, unfortunately without the training code. The presented results show limitations regarding accuracy and the inability to predict multi-modal futures as described in IVB, which can be seen in particular in the averaged predictions for long-term scenarios (see Fig. 3). 

<figure align="center">
  <img img src="/assets/predsem_img/eval.png" alt="fcn_architecture" style="zoom:60%;" class="center"/>
  <figcaption>Fig. 3: Last input segmentation, and ground truth segmentations at 1, 4, 7, and 10 seconds into the future (top row), and corresponding predictions of the autoregressive S2S model trained with fine-tuning (bottom row) [1].</figcaption>
</figure>

#### Conclusion and outlook on application in autonomous driving

I personally enjoyed reading the paper as well as seeing the development of the idea in the follow up works towards a more holistic form of visual scene understanding. It appears interesting to me how the presented approach can be scaled to be robust for various scenes and edge cases by being simultaneously real time capable. On the other hand, I am curious to see how it will be possible to incorporate 3D geometric information, long term temporal relationships and an awareness of uncertainty for the multi-modal future and how these representations can be leveraged in real life robotic applications.

Wayve demonstrated in their published work *Probabilistic Future Prediction for Video Scene Understanding* [20], that learning temporal representations improve their driving policy for their reinforcement learning based end-to-end approach. Here I focus on the aforementioned addressed limitations, and how they are tried to be mitigated in there paper:

- **Geometry and instance awareness:** The additional auxiliary task of depth estimation introduces information about the 3D geometry of the scene. Limitations, which could not have been mitigated are in my opinion that the system still works with a 2D projection of the scene and also no explicit knowledge of instances is available.
- **Long term dependencies:** The use of the spatio-temporal module, which uses 3D convolutions to encode temporal, spatial and global context, seems to be a more sophisticated choice than the in of P. Luc et al. used approach of 2D convolutions for stacked input frames. Still the framework has limitations to make use of long term relationships as the number of input frames is fixed and no pass-through of information possible. The temporal encoding gets additionally enforced by the auxiliary task of flow estimation in the perception module.
- **Multi-modal futures:** Sampling future predictions from a distribution, which is conditioned on the past context and trained by minimizing the distance to the future distribution of observed sequences, shows visually convincing and plausible results. This mitigates the key limitation of the approach presented by P. Luc et al., which was not able to capture any form of multimodality and averaged possible futures.
- **Estimation of uncertainties:** Another extension is the ability to estimate uncertainties about the predicted future scenes.
- **Limitation of performance caused by use of precomputed semantic segmentation maps:** In contrast to P. Luc et al. the here presented framework is using raw RGB images as input and is therefore just depending on the precomputed semantic segmentation for the training targets, which decreases the influence of potential error propagation.
- **Scalability and real-time capability:** The here proposed work also shows that the idea of predicting future semantic segmentation maps can be scaled by at the same time being real-time capable and that the learned representations are able to improve real-world driving policies.

#### References

[1] N. Neverova, P. Luc, C. Couprie, J. J. Verbeek, and Y. LeCun, “Predicting deeper into the future of semantic segmentation,” CoRR, vol. abs/1703.07684, 2017. 

[2] M. Mathieu, C. Couprie, and Y. LeCun, “Deep multi-scale video prediction beyond mean square error,” 2015. 

[3] C. Finn, I. Goodfellow, and S. Levine, “Unsupervised learning for physical interaction through video prediction,” 2016. 

[4] C. Vondrick, H. Pirsiavash, and A. Torralba, “Anticipating visual representations from unlabeled video,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 98–106, June 2016. 

[5] J. Walker, C. Doersch, A. Gupta, and M. Hebert, “An uncertain future: Forecasting from static images using variational autoencoders,” CoRR, vol. abs/1606.07873, 2016. 

[6] F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” 2015. 

[7] L. J. Ba and R. Caruana, “Do deep nets really need to be deep?,” CoRR, vol. abs/1312.6184, 2013. 

[8] A. Ess, K. Schindler, B. Leibe, and L. V. Gool, “Object detection and tracking for autonomous navigation in dynamic environments,” The International Journal of Robotics Research, vol. 29, pp. 1707 – 1725, 2010. 

[9] L. Zhang, Y. Li, and R. Nevatia, “Global data association for multiobject tracking using network flows,” 06 2008. 

[10] K. Yamaguchi, A. C. Berg, L. E. Ortiz, and T. L. Berg, “Who are you with and where are you going?,” in CVPR 2011, pp. 1345–1352, June 2011. 

[11] P. Trautman, J. Ma, R. M. Murray, and A. Krause, “Robot navigation in dense human crowds: Statistical models and experimental studies of human–robot cooperation,” The International Journal of Robotics Research, vol. 34, pp. 335 – 356, 2015. 

[12] A. Alahi, K. Goel, V. Ramanathan, A. Robicquet, L. Fei-Fei, and S. Savarese, “Social lstm: Human trajectory prediction in crowded spaces,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 961–971, June 2016. 

[13] W. Luo, B. Yang, and R. Urtasun, “Fast and furious: Real time endto-end 3d detection, tracking and motion forecasting with a single convolutional net,” in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3569–3577, June 2018. 

[14] S. Casas, W. Luo, and R. Urtasun, “Intentnet: Learning to predict intention from raw sensor data,” in Proceedings of The 2nd Conference on Robot Learning (A. Billard, A. Dragan, J. Peters, and J. Morimoto, eds.), vol. 87 of Proceedings of Machine Learning Research, pp. 947– 956, PMLR, 29–31 Oct 2018. 

[15] P. Luc, C. Couprie, Y. LeCun, and J. Verbeek, “Predicting future instance segmentations by forecasting convolutional features,” CoRR, vol. abs/1803.11496, 2018. 

[16] C. Couprie, P. Luc, and J. Verbeek, “Joint Future Semantic and Instance Segmentation Prediction,” in ECCV Workshop on Anticipating Human Behavior, vol. 11131 of Lecture Notes in Computer Science, (Munich, Germany), pp. 154–168, Springer, Sept. 2018. 

[17] X. Jin, H. Xiao, X. Shen, J. Yang, Z. Lin, Y. Chen, Z. Jie, J. Feng, and S. Yan, “Predicting scene parsing and motion dynamics in the future,” 2017. 

[18] S. S. Nabavi, M. Rochan, and Y. Wang, “Future semantic segmentation with convolutional LSTM,” CoRR, vol. abs/1807.07946, 2018. 

[19] A. Bhattacharyya, M. Fritz, and B. Schiele, “Bayesian prediction of future street scenes using synthetic likelihoods,” CoRR, vol. abs/1810.00746, 2018.

[20] A. Hu, F. Cotter, N. Mohan, C. Gurau, and A. Kendall, “Probabilistic future prediction for video scene understanding,” arXiv:2003.06409, 2020.