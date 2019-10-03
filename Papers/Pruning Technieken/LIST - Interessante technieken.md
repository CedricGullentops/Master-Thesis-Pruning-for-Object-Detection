# Lijst van papers over Pruning technieken - specifiek prunen van convolutiefilters

**Favorieten:**

22) [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration ](http://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html) **GELEZEN, ZEER GOED** -- TECH 7: CIFAR-10 FLOP's -52.3% acc 93.74, kleine stijging!

19) [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html) **GELEZEN, ZEER GOED** -- TECH 6: CIFAR-10 FLOP's -53% acc 93.3%

=> Taylor Expansie gebaseerde methode met uitbreiding met benadering van Hessian

16) [Centripetal SGD for Pruning Very Deep Convolutional Networks With Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) **GELEZEN, ZEER GOED** -- TECH 5: ResNet-110 FLOP's -39%, 94%  TOP1 ! **2019**

=> Zeer intuitief en goede resultaten, andere methode dan Taylor Expansie

3) [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) **GELEZEN, ZEER GOED** -- TECH3: VGG-16, 34% FLOP reductie => -2.3% acc, 48% FLOP reductie => -4.8% **2017** SAMEN MET 20) [Importance Estimation for Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html) **GELEZEN**

=> Lijkt me een goede techniek op basis van belangrijkheid om mee te beginnen, redelijk recent en goede resultaten. Beter dan regularisatie methodes door globale herschaling van criteria en hogere accuraatheid tijdens prunen.



**Anderen:**

8) [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) **GELEZEN** --  TECH 1: inference cost VGG-16 -34%, ResNet-110 -38%

2) [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/abs/1512.08571)  **GELEZEN, GOED OVERZICHT** -- TECH 2: netwerkgrootte -75%

17) [Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html) **GELEZEN** --  TECH 4: ResNet-110 FLOP's en Parameters  -38%, Channels - 62 %, 93% Acc (Net iets slechter dan 16?)