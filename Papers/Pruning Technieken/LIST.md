# Lijst van papers over Pruning technieken 

**General:**

1) [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149#) 

2) [Structured Pruning of Deep Convolutional Neural Networks](https://dl.acm.org/citation.cfm?id=3005348) -- Unavailable?

3) [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)

4) [Learning Both Weights and Connections for Efficient Neural Network](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network)

5) [Channel Pruning for Accelerating Very Deep Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html)

6) [Designing Energy-Efficient Convolutional Neural Networks Using Energy-Aware Pruning](http://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html)

7) [Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)

8) [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)

9) [Dynamic Network Surgery for Efficient DNNs](http://papers.nips.cc/paper/6165-dynamic-network-surgery-for-efficient-dnns)

10) [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](http://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html)

11) [Data-free parameter pruning for Deep Neural Networks](https://arxiv.org/abs/1507.06149)

12) [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250)

13) [Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/abs/1808.06866)

14) [CNNpack: Packing Convolutional Neural Networks in the Frequency Domain](http://papers.nips.cc/paper/6389-cnnpack-packing-convolutional-neural-networks-in-the-frequency-domain)



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

1)  Algemene paper met vooral statistieken op resultaten van pruning, quantisatie en Huffman. Redenen voor compressie: storage, energy consumption, speedup. Pruning en quantisatie zijn onafhankelijk van elkaar (fig 7, 8). Gewichts vermindering door pruning in % voor VGG-16 en andere. 8% is pruning limiet, hierna vermindering accuraatheid (voor hun technieken). Quantisatie werkt beter op geprunde netwerken. Accuraatheid per type layer tov. # bits voor precisie. Grootste hoeveelheid geheugen zit in FC layers. Een voorbeeld van geen voordeel bij pruning 6.3. Lijst van andere belangrijke papers omtrent pruning!