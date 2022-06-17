# A reading list for some interesting papers on Human-centered AI, with a focus on computer vision

## Data collection and annotation
How training data are collected by human subjects and how can we speed up the annotation process.
* Kovashka, A., Russakovsky, O., Fei-Fei, L. and Grauman, K., 2016. Crowdsourcing in computer vision. Foundations and Trends in Computer Graphics and Vision
* Tsipras, D., Santurkar, S., Engstrom, L., Ilyas, A. and Madry, A., 2020. From imagenet to image classification: Contextualizing progress on benchmarks. In International Conference on Machine Learning
* Acuna, D., Ling, H., Kar, A. and Fidler, S., 2018. Efficient interactive annotation of segmentation datasets with polygon-rnn++. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition
* Mandlekar, Ajay, et al. "Roboturk: A crowdsourcing platform for robotic skill learning through imitation." Conference on Robot Learning. PMLR, 2018.

## Weak supervision and self-supervision
How we reduce the human supervision in different learning schemes.
* Li, B., Weinberger, K.Q., Belongie, S., Koltun, V. and Ranftl, R., 2022. Language-driven Semantic Segmentation. ICLR'22
* Lan, S., Yu, Z., Choy, C., Radhakrishnan, S., Liu, G., Zhu, Y., ... & Anandkumar, A. (2021). DISCOBOX: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision. ICCV 2021
* Ke, Tsung-Wei, Jyh-Jing Hwang, and Stella X. Yu., 2021. Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning. ICLR, 2021
* He, K., Chen, X., Xie, S., Li, Y., Dollár, P. and Girshick, R., 2021. Masked autoencoders are scalable vision learners. arXiv preprint arXiv:2111.06377.
* Weakly supervised object localization: https://github.com/xiaomengyc/Weakly-Supervised-Object-Localization 
* Sun, C., Shrivastava, A., Singh, S. and Gupta, A., 2017. Revisiting unreasonable effectiveness of data in deep learning era. In Proceedings of the IEEE international conference on computer vision

## Explainable ML
How we visualize features and explain the prediction of AI models.
* Bau, D., Zhu, J.Y., Strobelt, H., Lapedriza, A., Zhou, B. and Torralba, A., 2020. Understanding the role of individual units in a deep neural network. Proceedings of the National Academy of Sciences
* Olah, C., Mordvintsev, A. and Schubert, L., 2017. Feature visualization. Distill, 2(11)
* Olah, C., Satyanarayan, A., Johnson, I., Carter, S., Schubert, L., Ye, K. and Mordvintsev, A., 2018. The building blocks of interpretability. Distill
* CAM, grad-CAM, and many other CAM variants. https://github.com/frgfm/torch-cam , https://github.com/jacobgil/pytorch-grad-cam 
* Ghorbani, A., Wexler, J., Zou, J.Y. and Kim, B., 2019. Towards automatic concept-based explanations. Advances in Neural Information Processing Systems,

## Debate on explainable ML
Is expalinable ML really meaningful and useful?
* Adebayo, J., Muelly, M., Abelson, H. and Kim, B., 2021, Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation. In International Conference on Learning Representations.
* Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. and Kim, B., 2018. Sanity checks for saliency maps. arXiv preprint arXiv:1810.03292.
* Leavitt, M.L. and Morcos, A., 2020. Towards falsifiable interpretability research. arXiv preprint arXiv:2010.12016, on the importance of single directions for generalization (https://arxiv.org/pdf/1803.06959.pdf), revisiting the importance of individual units in CNNs via ablation (https://arxiv.org/pdf/1806.02891.pdf) 
* Ghassemi, Marzyeh, Luke Oakden-Rayner, and Andrew L. Beam. "The false hope of current approaches to explainable artificial intelligence in health care." The Lancet Digital Health 3.11 (2021): e745-e750.
* Zachary Lipton. The Mythos of Model Interpretability. 2017

## AI Bias (dataset bias and algorithmic bias)
How we identify the various biases in AI models and data collection processes.
* Hendricks, L.A., Burns, K., Saenko, K., Darrell, T. and Rohrbach, A., 2018. Women also snowboard: Overcoming bias in captioning models. In Proceedings of the European Conference on Computer Vision (ECCV)
* Hooker, S., 2021. Moving beyond “algorithmic bias is a data problem”. Patterns
* Torralba, A. and Efros, A.A., 2011, June. Unbiased look at dataset bias. In CVPR 2011
* Buolamwini, J. and Gebru, T., 2018, January. Gender shades: Intersectional accuracy disparities in commercial gender classification. In Conference on fairness, accountability and transparency
* Algorithmic bias: https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/

## AI robustness 
How AI models are robust to adversarial samples and the real-world perturbations.
* Song, D., Eykholt, K., Evtimov, I., Fernandes, E., Li, B., Rahmati, A., Tramer, F., Prakash, A. and Kohno, T., 2018. Physical adversarial examples for object detectors. In 12th USENIX workshop on offensive technologies (WOOT 18). Another relevant paper on LiDAR adversarial attack: Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures
* Ilyas, A., Santurkar, S., Tsipras, D., Engstrom, L., Tran, B. and Madry, A., 2019. Adversarial examples are not bugs, they are features. Advances in neural information processing systems, 32
* Xiao, K.Y., Engstrom, L., Ilyas, A. and Madry, A., 2020, September. Noise or Signal: The Role of Image Backgrounds in Object Recognition. In International Conference on Learning Representations.
* Hendrycks, D. and Dietterich, T., 2018, September. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. In International Conference on Learning Representations.
* Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F.A. and Brendel, W., 2018. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. arXiv preprint arXiv:1811.12231.

## Human-AI Collaboration
How we collaborate with AI models to accomplish tasks.
* Cai, Carrie J., et al. "Human-centered tools for coping with imperfect algorithms during medical decision-making." CHI'19
* Tschandl, P., Rinner, C., Apalla, Z., Argenziano, G., Codella, N., Halpern, A., Janda, M., Lallas, A., Longo, C., Malvehy, J. and Paoli, J., 2020. Human–computer collaboration for skin cancer recognition. Nature Medicine, 26(8), pp.1229-1234.
* OpenAI Codex (Github Copilot): AI Pair-programming: https://openai.com/blog/openai-codex/
* Buçinca, Z., Malaya, M.B. and Gajos, K.Z., 2021. To trust or to think: cognitive forcing functions can reduce overreliance on AI in AI-assisted decision-making. Proceedings of the ACM on Human-Computer Interaction

## Human-AI Creation
How generative models such as GAN and diffusion models can be used for interactive content creation.
* Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors (https://arxiv.org/pdf/2203.13131.pdf)
* Wang, S.Y., Bau, D. and Zhu, J.Y., 2021. Sketch your own gan. ICCV. (https://peterwang512.github.io/GANSketching/)
* Park, Taesung, et al. "Semantic image synthesis with spatially-adaptive normalization." CVPR 2019.
* Unsupervised discovery of steerable dimensions: https://genforce.github.io/sefa/, GANSpace (https://github.com/harskish/ganspace), LatentCLR (https://github.com/catlab-team/latentclr)
* Yan, Chuan, et al. "FlatMagic: Improving Flat Colorization through AI-driven Design for Digital Comic Professionals." CHI Conference on Human Factors in Computing Systems. 2022.
* DALLE2 paper Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125.

## Human-in-the-loop autonomy
Howe humans work together with AI for accomplishing control tasks.
* Reddy, S., Dragan, A.D. and Levine, S., Shared Autonomy via Deep Reinforcement Learning. RSS 2018
* Li, Q., Peng, Z. and Zhou, B., 2022. Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization. ICLR'22
* Lee, K., Smith, L.M. and Abbeel, P., 2021, July. PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training. In International Conference on Machine Learning
* Spencer, J., Choudhury, S., Barnes, M., Schmittle, M., Chiang, M., Ramadge, P. and Srinivasa, S., 2021. Expert Intervention Learning. Autonomous Robots
* Zhang, R., Torabi, F., Warnell, G. and Stone, P., 2021. Recent advances in leveraging human guidance for sequential decision-making tasks. Autonomous Agents and Multi-Agent Systems
* Reddy, Sid, Anca Dragan, and Sergey Levine. "Pragmatic Image Compression for Human-in-the-Loop Decision-Making." Advances in Neural Information Processing Systems
* Christiano, P.F., Leike, J., Brown, T., Martic, M., Legg, S. and Amodei, D., 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems (https://openai.com/blog/deep-reinforcement-learning-from-human-preferences/ )
* Hilton, J., Cammarata, N., Carter, S., Goh, G. and Olah, C., 2020. Understanding rl vision. Distill, 5(11), p.e29.

## Knowledge generation from superhuman AI
How we acquire knowledge from superhuman AI.
* McGrath, T., Kapishnikov, A., Tomašev, N., Pearce, A., Hassabis, D., Kim, B., ... & Kramnik, V. (2021). Acquisition of Chess Knowledge in AlphaZero. arXiv preprint arXiv:2111.09259.
* Wurman, P.R., Barrett, S., Kawamoto, K., MacGlashan, J., Subramanian, K., Walsh, T.J., Capobianco, R., Devlic, A., Eckert, F., Fuchs, F. and Gilpin, L., 2022. Outracing champion Gran Turismo drivers with deep reinforcement learning. Nature
* Vinyals, Oriol, et al. "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575.7782 (2019): 350-354.

