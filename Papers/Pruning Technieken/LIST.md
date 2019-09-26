# Lijst van papers over Pruning technieken 

**Algemeen:**

1) [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149#)  **GELEZEN**

2) [Structured Pruning of Deep Convolutional Neural Networks](https://dl.acm.org/citation.cfm?id=3005348) -- Unavailable? **ZEKER NOG LEZEN** -- TECH2

3) [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) **GELEZEN, ZEER GOED** -- TECH3

4) [Learning Both Weights and Connections for Efficient Neural Network](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network)  **GELEZEN**

5) [Channel Pruning for Accelerating Very Deep Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html)

6) [Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning](http://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html)

7) [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)

8) [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710) **GELEZEN** --  TECH1

9) [Dynamic Network Surgery for Efficient DNNs](http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns)

10) [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](http://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html)

11) [Data-free parameter pruning for Deep Neural Networks](https://arxiv.org/abs/1507.06149)

12) [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)

13) [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)

14) [CNNpack: Packing Convolutional Neural Networks in the Frequency Domain](http://papers.nips.cc/paper/6389-cnnpack-packing-convolutional-neural-networks-in-the-frequency-domain)

23) [Optimal Brain Damage](http://papers.nips.cc/paper/250-optimal-brain-damage.pdf) **GELEZEN**

24) [Optimal Brain Surgeon and general network pruning](https://ieeexplore.ieee.org/abstract/document/298572) **GELEZEN**

25) [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/pdf/1710.01878.pdf) **ZEKER NOG EENS LEZEN** -- waarom prunen

26) [Learning to Prune Filters in Convolutional Neural Networks](https://arxiv.org/pdf/1801.07365.pdf) **ZEKER NOG EENS LEZEN** -- meer praktisch



**Sinds 2019:**

15) [Attention Based Pruning for Shift Networks](https://arxiv.org/abs/1905.12300)

16) [Centripetal SGD for Pruning Very Deep Convolutional Networks With Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)

17) [Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.html)

18) [Pruning deep convolutional neural networks for efficient edge computing in condition assessment of infrastructures](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12449)

19) [Collaborative Channel Pruning for Deep Networks](http://proceedings.mlr.press/v97/peng19c.html)

20) [Importance Estimation for Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)

21) [Pruning Redundant Neurons and Kernels of Deep Convolutional Neural Networks](https://patents.google.com/patent/US20190122113A1/en)

22) [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](http://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html)

---

**Opmerkingen:**

1)  Algemene paper met vooral statistieken rond resultaten van pruning, quantisatie en Huffman. Redenen voor compressie: storage, energy consumption, speedup. Pruning en quantisatie zijn onafhankelijk van elkaar (fig 7, 8). Gewichts vermindering door pruning in % voor VGG-16 en andere. 8% is pruning limiet, hierna vermindering accuraatheid (voor hun technieken). Quantisatie werkt beter op geprunde netwerken. Accuraatheid per type layer tov. # bits voor precisie. Grootste hoeveelheid geheugen zit in FC layers. Een voorbeeld van geen voordeel bij pruning 6.3. Lijst van referenties andere belangrijke papers omtrent pruning!

23) Paper over omvang-gebaseerde pruning technieken, met name Optimal Brain Damage (OBD), techniek gebaseerd op tweede afgeleide informatie. Basis idee voor pruning, parameters vermindert met factor x8. Twee nadelen: vereist finetuning en vertraagd het leerprocess. Hoofdidee => kleine netwerken zijn beter in veralgemenen. 

24) Verbetering op OBD, Optimal Brain Surgery (OBS) gebaseerd op Hessian matrix. Delete geen cruciale parameters en heeft geen hertraining nodig omdat het de andere gewichten ook aanpast.

4) Pruning kan gecombineerd worden met andere technieken zoals HashedNets. Gewichten onder bepaalde threshold worden verwijdert. Regularisatie dient om generalisatie fouten te verminderen en beinvloed de impact van pruning en hertraining. L2 regulisatie is het beste. Geprunde parameters moeten hertrained worden en niet opnieuw geinitialiseerd. Neurale netwerken ondervinden vaak *the vanishing gradient problem* wanneer het netwerk dieper word wat pruning errors moeilijker maakt om te herstellen.  (=> paper Yoshua Bengio, Patrice Simard, and Paolo Frasconi. Learning long-term dependencies with gradient descent is difficult.) Dit probleem zorgt ervoor dat de eerste lagen trager hertrained worden dan diepere lagen. Diepe netwerken, die ook complexere dingen kunnen doen, hebben hier het meeste last van. Dit is ook de reden dat men RELU based activatie methodes gebruikt in plaats van Sigmoid of Tanh. Eerst worden connecties gepruned, hierna worden de verbonden neuronen die nu geen connecties meer hebben (inkomend of uitgaand) verwijdert. Pruning threshold is quality parameter vermenigvuldigt met de standaard afwijking van de gewichten van een laag. Pruning zorgt ervoor dat de gewichten het midden van een afbeelding belangrijker vinden doordat de meeste voorbeelden zich hier bevinden. Pruning word best enkel gedaan op netwerken die klaar zijn voor deployment. 5 pruning iteraties gebruikt.

8) Veel links naar andere papers met pruning technieken! Vooral interessant voor FLOP reductie en het niet gebruiken van sparse netwerken, libraries en hardware. One-shot pruning en hertrainen. Een filter verwijderen verwijdert de bijbehordende feature maps voor alle opeenvolgende lagen. Kleinere filters prunen werkt beter. Elke laag appart getest. Grote hoeveelheid van filters kunnen verwijdert worden zonder verlies in nauwkeurigheid. Meest sensitieve lagen liggen bij de residual blocks waar het aantal feature mappen verandert. Eerste laag van deze residual block verwijderen verlaagt het meeste FLOP.

3) Bevat meerdere pruning criteria en redenen om te prunen! **Zeer goed overzicht.** Lagen met max. pooling zijn belangrijker dan andere. Elke laag heeft feature maps die globaal belangrijk zijn en somige andere die minder belangrijk zijn. => gebalanceerd prunen over alle lagen is het best. Taylor criteria beste nauwkeurigheid en beste relatieve performantie tov. #operaties. Hertrainen tijdens na het prunen verbeterd de nauwkeurigheid enorm.