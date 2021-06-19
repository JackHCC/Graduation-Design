# Graduation-Design
北京科技大学本科毕业设计-基于车辆轨迹时空数据的城市热点预测模型研究

### 摘要
智能交通在近年得到了学术界和产业界的广泛重视。尤其是随着道路网的不断完善，交通车流越来越庞大，交通流预测显得越来越重要，分析并预测交通状况和交通热点分布情况是交通管控的基础，对城市交通管控有着十分重要的意义。随着车辆轨迹大数据技术、人工智能和机器学习技术的发展，基于机器学习和大数据对车辆密度进行预测已成为重要的技术趋势。
本文基于车辆轨迹大数据，利用机器学习技术对城市交通热点进行预测，主要的研究内容和创新点罗列如下：
首先，建立车流密度提取模型，利用核密度估计算法从车辆轨迹时空数据中提取车辆密度特征，并实现热点预测的可视化。本文从交通属性中车辆密度的角度去分析，相比传统的车流量和车速属性，让交通预测具有更加全局的特征信息，为交通管控增添一个新的维度与视角。
其次，提出预测滑动窗口模型，构建预测所需要的训练数据集，并使用标准的归一化方法进行处理，利用支持向量回归算法进行出租车车辆密度预测和热点预测，最后借助公认的评价指标对模型性能进行评估。为后续神经网络预测工作提供基础性参考。
再次，利用经典的神经网络——多层感知器模型对比不同层数和不同神经元个数的网络结构的性能，并使用循环神经网络中的长短期记忆模型进行预测，完成北京市出租车热点预测并达到预期效果。本文为机器学习应用于交通领域的全局和局部预测提供了新的思路，为该方向的研究提供基础性指标参考。
最后，总结短时预测模式下本文所述模型在不同时间尺度下的预测性能，并提出长时预测的概念，为后续研究提供新的交通预测思路，将交通的短时预测方向扩充到长时预测的场景下。
关键词：	机器学习，核密度估计，交通热点预测，支持向量回归，多层感知器


### Urban hot spot prediction model based on spatiotemporal data of vehicle trajectory
Abstract
In recent years, intelligent transportation has received extensive attention from academia and industry. Especially with the continuous improvement of the road network, the traffic flow is becoming larger and larger, and the traffic flow prediction is becoming more and more important. The analysis and prediction of traffic conditions and the distribution of traffic hotspots are the basis of traffic control, which is of great significance for urban traffic control. With the development of vehicle trajectory big data technology, artificial intelligence and machine learning technology, vehicle density prediction based on machine learning and big data has become an important technical trend.
In this thesis, machine learning technology is used to predict urban traffic hot spots based on vehicle track big data. The main research contents and innovation points are listed as follows.
Firstly, the vehicle flow density extraction model is established, and the kernel density estimation algorithm is used to extract the vehicle density features from the spatiotemporal data of vehicle tracks, and the visualization of hot spot prediction is realized. In this thesis, from the perspective of vehicle density in traffic attributes, compared with the traditional vehicle flow and speed attributes, the traffic prediction has more global characteristic information, adding a new dimension and perspective to traffic control.
Secondly, the prediction sliding window model was proposed to construct the training data set required for the prediction, and the standard normalization method was used for processing. The support vector machine regression algorithm was used for the taxi vehicle density prediction and hot spot prediction. Finally, the performance of the model was evaluated by the recognized evaluation indexes. It provides the basic flow for the subsequent neural network prediction.
Thirdly, the classical neural network called multi-layer perceptron model was used to compare the performance of network structures with different layers and neurons, and the long and short term memory model in the cyclic neural network was used to predict the hot spots of taxis in Beijing, and the expected results were achieved. This thesis provides a new idea for the application of machine learning to global and local prediction in the field of transportation, and provides a basic index reference for the research in this direction.
Finally, the prediction performance of the model described in this thesis under the short-time prediction mode is summarized under different time scales, and the concept of long-time prediction is proposed to provide a new forecasting idea for the follow-up research, and the direction of short-time traffic prediction is extended to the scenario of long-time prediction.
