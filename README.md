# CollMetric

A Tensorflow implementation of Collaborative Metric Learning (CML): 

*Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge Belongie, and Deborah Estrin. 2017. Collaborative Metric Learning. In Proceedings of the 26th International Conference on World Wide Web (WWW '17) ([perm_link](http://dl.acm.org/citation.cfm?id=3052639), [pdf](http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf))*

** Note: the original Theano implementation is deprecated and is kept in the *old_experiment_code branch*
# Feature
* Produces user-item joint embedding that accurately captures the user-item preference information, and user-user, item-item similarity. 
* Allows the exploitation of item features (e.g. tags, text, image features) to further improve the recommendation accuracy.
* Outperforms state-of-the-art collaborative filtering algorithms in accuracy on a wide range of recommendation tasks
* Enjoys an extremely efficient Top-K recommendation search using approximate nearest-neighbor.
# Requirements
 * python3
 * tensorflow
 * scipy
 * scikit-learn
# Usage
```bash
# install requirements
pip3 install -r requirements.txt
# run demo tensorflow model
python3 CML.py
```
