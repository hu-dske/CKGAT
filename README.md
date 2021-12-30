CKGAT
=====
Hi! You are welcome to visit here!<br>
This library will be used to release the source code of CKGAT, a newly proposed method for knowledge graph-based top-N recommendation. The research paper of CKGAT has been submitted to an International journal.* ***PLEASE NOTE THAT*** CKGAT’s code will ***not*** be released immediately (postponed until the paper is accepted and published). *<br>
<br>
CKGAT stands for Collaborative Knowledge-aware Graph Attention Network for top-N Recommendation. Four real-world datasets (Last.FM, Book-Crossing, MovieLens 20M, and Dianping-Food) were used to empirically evaluate the performance of CKGAT, and the experimental results show that CKGAT overall outperforms three baseline methods and six state-of-the-art propagation-based recommendation methods in terms of recommendation accuracy, and outperforms four representative propagation-based recommendation methods in terms of recommendation diversity. The information about these methods are given below.


Comparison Models
------
* **BPRMF** is a Bayesian personalized ranking (BPR) optimized matrix factorization (MF) model achieved by applying LearnBPR to MF.<br>
*`Paper:`* Zhang, F.; Yuan, N.J.; Lian, D.; Xie, X.; Ma, W.-Y. Collaborative Knowledge Base Embedding for Recommender Systems. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 13–17 August 2016; pp. 353-362. https://doi.org/10.1145/2939672.2939673<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **CKE** is a typical knowledge graph embedding-based recommendation method, which combines the structural knowledge, textual knowledge and visual knowledge of items to learn item representations.<br>
*`Paper:`* Rendle, S.; Freudenthaler, C.; Gantner, Z.; Schmidt-Thieme, L. BPR: Bayesian Personalized Ranking from Implicit Feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence, Montreal, QC, Canada, 18–21 June 2009; pp. 452–461. 836 https://arxiv.org/abs/1205.2618<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **KPRN** is a typical connection-based recommendation method, which generates path representations by combining the semantics of entities and relations and distinguishes the importance of different paths, thereby capturing user preferences.<br>
*`Paper:`* Wang, X.; Wang, D.; Xu, C.; He, X.; Cao, Y.; Chua, T.-S. Explainable Reasoning over Knowledge Graphs for Recommendation. In Proceedings of the 33rd AAAI Conference on Artificial Intelligence, Honolulu, Hawaii, USA, January 27–February 1, 2019; pp. 5329–5336. https://doi.org/10.1609/aaai.v33i01.33015329<br>
*`Code:`* https://github.com/xiangwang1223/KPRN

* **RippleNet** is a classical propagation-based recommendation method, which enhances user representations by propagating users’ potential preferences in the knowledge graph.<br>
*`Paper:`* Wang, H.; Zhang, F.; Wang, J.; Zhao, M.; Li, W.; Xie, X.; Guo, M. RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, Torino, Italy, 22–26 October 2018; pp. 417–426. https://doi.org/10.1145/3269206.3271739<br>
*`Code:`* https://github.com/hwwang55/RippleNet

* **CKAN** belongs to the propagation-based recommendation methods, which uses a heterogeneous propagation strategy and an attention network to learn ripple set embeddings, thereby generating user embeddings and item embeddings.<br>
*`Paper:`* Wang, Z.; Lin, G.; Tan, H.; Chen, Q.; Liu, X. CKAN: Collaborative Knowledge-aware Attentive Network for Recommender Systems. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, Virtual Event, China, 25–30 July 2020; pp. 219–228. https://doi.org/10.1145/3397271.3401141<br>
*`Code:`* https://github.com/weberrr/CKAN

* **KGCN** belongs to the propagation-based recommendation methods, which applies the graph convolutional network to the knowledge graph to aggregate neighborhood information to refine item representations.<br>
*`Paper:`* Wang, H.; Zhao, M.; Xie, X.; Li, W.; Guo, M. Knowledge Graph Convolutional Networks for Recommender Systems. In Proceedings of 28th The World Wide Web Conference, San Francisco, CA, USA, May 13-17, 2019; pp. 3307-3313. https://doi.org/10.1145/3308558.3313417<br>
*`Code:`* https://github.com/hwwang55/KGCN

* **KGNN-LS** belongs to the propagation-based recommendation methods, which adds a label smoothing mechanism to the KGCN framework to propagate user interaction labels, so as to provide effective recommendations.<br>
*`Paper:`* Wang, H.; Zhang, F.; Zhang, M.; Leskovec, J.; Zhao, M.; Li, W.; Wang, Z. Knowledge-aware Graph Neural Networks with Label Smoothness Regularization for Recommender Systems. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Anchorage, AK, USA, 4–8 August 2019; pp. 968–977. https://doi.org/10.1145/3292500.3330836<br>
*`Code:`* https://github.com/hwwang55/KGNN-LS

* **KGAT** belongs to the propagation-based recommendation methods, which applies the graph attention network to the collaborative knowledge graph to learn user representations and item representations.<br>
*`Paper:`* Wang, X.; He, X.; Cao, Y.; Liu, M.; Chua, T.-S. KGAT: Knowledge Graph Attention Network for Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Anchorage, AK, USA, 4–8 August 2019; pp. 950–958. https://doi.org/10.1145/3292500.3330989<br>
*`Code:`* https://github.com/xiangwang1223/knowledge_graph_attention_network/tree/master/Model

* **KGIN** is currently the state-of-the-art propagation-based recommendation method. It uses auxiliary item knowledge to explore the users’ intention behind the user-item interactions, thus refining the representations of users and items.<br>
*`Paper:`* Wang, X.; He, X.; Cao, Y.; Liu, M.; Chua, T.-S. KGAT: Knowledge Graph Attention Network for Recommendation. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Anchorage, AK, USA, 4–8 August 2019; pp. 950–958. https://doi.org/10.1145/3292500.3330989<br>
*`Code:`* https://github.com/huangtinglin/Knowledge_Graph_based_intent_Network
