CKGAT 
=====
Hi! You are welcome to visit here!<br>
This repository is used to release the code of CKGAT, a newly proposed method for knowledge graph-based top-N recommendation by our research team. **CKGAT** stands for the **Collaborative Knowledge-aware Graph Attention Network** for top-N Recommendation, which can learn refined ripple set embeddings, thereby generating accurate user embeddings and item embeddings, so as to accurately capture users’ potential interests in items. To the best of our knowledge, it is the first method that uses the knowledge-aware graph attention network to learn the refined ripple set embeddings. The research paper of CKGAT has been published in the open access international journal Applied Sciences, which is available at: https://www.mdpi.com/2076-3417/12/3/1669. The citation format in the MDPI and ACS Style is as follows: 

Xu, Z.; Liu, H.; Li, J.; Zhang, Q.; Tang, Y. CKGAT: Collaborative Knowledge-Aware Graph Attention Network for Top-N Recommendation. *Appl. Sci.* **2022**, 12(3), 1669. https://doi.org/10.3390/app12031669

In essence, our CKGAT method makes two important improvements to the Collaborative Knowledge-aware Attentive Network (CKAN) [by Z. Wang et al., SIGIR'20, https://doi.org/10.1145/3397271.3401141], a typical state-of-the-art propagation-based recommendation method: the generation of ripple set embeddings, and the generation of user/item embeddings. When implementing our CKGAT method, we used the PyTorch framework to modify the CKAN code in two major aspects: (1) Modify the knowledge-aware attentive embedding layer in CKAN to the knowledge-aware GAT-based attentive embedding layer in CKGAT; (2) Modify the aggregator in CKAN to the attention aggregator in CKGAT. 


Four real-world datasets (Last.FM, Book-Crossing, MovieLens 20M, and Dianping-Food) were used to empirically evaluate the performance of CKGAT, and the experimental results show that CKGAT overall outperforms three baseline methods (BPRMF, CKE, and KPRN) and six state-of-the-art propagation-based recommendation methods (RippleNet, CKAN, KGCN, KGNN-LS, KGAT, and KGIN) in terms of recommendation accuracy, and outperforms four representative propagation-based recommendation methods (KGNN-LS, KGAT, CKAN, and KGIN) in terms of recommendation diversity. Detailed information about the experimental datasets and the comparison methods are given below.


Experimental Datasets
------
* **Last.FM** (https://grouplens.org/datasets/hetrec-2011/)**.** This dataset contains social networking, tagging, and music artist listening information from a set of 2,000 users from Last.fm online music system. Instead of the original dataset, our experiments directly used the preprocessed Last.FM dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/weberrr/CKAN/tree/master/data] by Z. Wang et al. [1]

* **Book-Crossing** (https://grouplens.org/datasets/book-crossing/) [2]**.** This dataset collects explicit ratings (ranging from 0 to 10) from different readers about various books in the book-crossing community. Instead of the original dataset, our experiments directly used the preprocessed Book-Crossing dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/weberrr/CKAN/tree/master/data] by Z. Wang et al. [1]

* **MovieLens 20M** (https://grouplens.org/datasets/movielens/20m/) [3]**.** This dataset is a widely used benchmark dataset in movie recommendation, which contains approximately 20 million explicit user ratings for movies (ranging from 1 to 5) on the MovieLens website. Instead of the original dataset, our experiments directly used the preprocessed MovieLens 20M dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/weberrr/CKAN/tree/master/data] by Z. Wang et al. [1]

* **Dianping-Food** (https://www.dianping.com/)**.** This dataset is provided by Meituan-Dianping Group (Dianping.com), which contains 10 million interaction data (including clicks and purchases, etc.) between approximately 2 million users and 1,000 restaurants. Instead of the original dataset, our experiments directly used the preprocessed Dianping-Food dataset and its corresponding knowledge graph released on GitHub [available at https://github.com/hwwang55/KGNN-LS/tree/master/data/restaurant] by H. Wang et al. [4]

References:

[1] Wang, Z.; Lin, G.; Tan, H.; Chen, Q.; Liu, X. CKAN: Collaborative Knowledge-aware Attentive Network for Recommender
Systems. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval,
Virtual Event, China, 25–30 July 2020; pp. 219–228. https://doi.org/10.1145/3397271.3401141

[2] Ziegler, C.-N.; McNee, S.M.; Konstan, J.A.; Lausen, G. Improving recommendation lists through topic diversification. In Proceedings of the 14th international conference on World Wide Web, Chiba, Japan, 10-14 May 2005. pp. 22–32. https://doi.org/10.1145/1060745.1060754

[3] Harper, F.M.; Konstan, J.A. The MovieLens Datasets: History and Context. *ACM Trans. Interact. Intell. Syst.* **2016**, *5*, 1-19. https://doi.org/10.1145/2827872

[4] Wang, H.; Zhang, F.; Zhang, M.; Leskovec, J.; Zhao, M.; Li, W.; Wang, Z. Knowledge-aware Graph Neural Networks
with Label Smoothness Regularization for Recommender Systems. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Anchorage, AK, USA, 4–8 August 2019; pp. 968–977. https://doi.org/10.1145/3292500.3330836

Comparison Methods
------
* **BPRMF** is a Bayesian personalized ranking (BPR) optimized matrix factorization (MF) model achieved by applying LearnBPR to MF.<br>
*`Paper:`* Rendle, S.; Freudenthaler, C.; Gantner, Z.; Schmidt-Thieme, L. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence, UAI '09, Montreal, QC, Canada, 18–21 June 2009; pp. 452–461. https://arxiv.org/abs/1205.2618<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **CKE** is a typical knowledge graph embedding-based recommendation method, which combines the structural knowledge, textual knowledge and visual knowledge of items to learn item representations.<br>
*`Paper:`* Zhang, F.; Yuan, N.J.; Lian, D.; Xie, X.; Ma, W.-Y. Collaborative Knowledge Base Embedding for Recommender Systems. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, San Francisco, CA, USA, 13–17 August 2016; pp. 353-362. https://doi.org/10.1145/2939672.2939673<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **KPRN** is a typical connection-based recommendation method, which generates path representations by combining the semantics of entities and relations and distinguishes the importance of different paths, thereby capturing user preferences.<br>
*`Paper:`* Wang, X.; Wang, D.; Xu, C.; He, X.; Cao, Y.; Chua, T.-S. Explainable Reasoning over Knowledge Graphs for Recommendation. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence, AAAI '19, Honolulu, Hawaii, USA, January 27–February 1, 2019; pp. 5329–5336. https://doi.org/10.1609/aaai.v33i01.33015329<br>
*`Code:`* https://github.com/xiangwang1223/KPRN

* **RippleNet** is a classical propagation-based recommendation method, which enhances user representations by propagating users’ potential preferences in the knowledge graph.<br>
*`Paper:`* Wang, H.; Zhang, F.; Wang, J.; Zhao, M.; Li, W.; Xie, X.; Guo, M. RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, CIKM '18, Torino, Italy, 22–26 October 2018; pp. 417–426. https://doi.org/10.1145/3269206.3271739<br>
*`Code:`* https://github.com/hwwang55/RippleNet

* **CKAN** belongs to the propagation-based recommendation methods, which uses a heterogeneous propagation strategy and an attention network to learn ripple set embeddings, thereby generating user embeddings and item embeddings.<br>
*`Paper:`* Wang, Z.; Lin, G.; Tan, H.; Chen, Q.; Liu, X. CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR '20, Virtual Event, China, 25–30 July 2020; pp. 219–228. https://doi.org/10.1145/3397271.3401141<br>
*`Code:`* https://github.com/weberrr/CKAN

* **KGCN** belongs to the propagation-based recommendation methods, which applies the graph convolutional network to the knowledge graph to aggregate neighborhood information to refine item representations.<br>
*`Paper:`* Wang, H.; Zhao, M.; Xie, X.; Li, W.; Guo, M. Knowledge Graph Convolutional Networks for Recommender Systems. In Proceedings of 28th The World Wide Web Conference, WWW '19, San Francisco, CA, USA, May 13-17, 2019; pp. 3307-3313. https://doi.org/10.1145/3308558.3313417<br>
*`Code:`* https://github.com/hwwang55/KGCN

* **KGNN-LS** belongs to the propagation-based recommendation methods, which adds a label smoothing mechanism to the KGCN framework to propagate user interaction labels, so as to provide effective recommendations.<br>
*`Paper:`* Wang, H.; Zhang, F.; Zhang, M.; Leskovec, J.; Zhao, M.; Li, W.; Wang, Z. Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD '19, Anchorage, AK, USA, 4–8 August 2019; pp. 968–977. https://doi.org/10.1145/3292500.3330836<br>
*`Code:`* https://github.com/hwwang55/KGNN-LS

* **KGAT** belongs to the propagation-based recommendation methods, which applies the graph attention network to the collaborative knowledge graph to learn user representations and item representations.<br>
*`Paper:`* Wang, X.; He, X.; Cao, Y.; Liu, M.; Chua, T.-S. KGAT: Knowledge Graph Attention Network for Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD '19, Anchorage, AK, USA, 4–8 August 2019; pp. 950–958. https://doi.org/10.1145/3292500.3330989<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **KGIN** is currently the state-of-the-art propagation-based recommendation method. It uses auxiliary item knowledge to explore the users’ intention behind the user-item interactions, thus refining the representations of users and items.<br>
*`Paper:`* Wang, X.; Huang, T.; Wang, D.; Yuan, Y.; Liu, Z.; He, X.; Chua, T.-S. Learning Intents behind Interactions with Knowledge Graph for Recommendation. WWW '21: The Web Conference 2021, Virtual Event/Ljubljana, Slovenia, 19–23 April 2021; pp. 878–887. https://doi.org/10.1145/3442381.3450133<br>
*`Code:`* https://github.com/huangtinglin/Knowledge_Graph_based_intent_Network
