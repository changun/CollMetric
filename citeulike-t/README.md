### The following text is from http://www.wanghao.in/CDL.htm ###
This dataset, *citeulike-t*, was used in the paper **[Collaborative Topic Regression with Social Regularization](https://www.cs.ucsb.edu/~binyichen/IJCAI13-400.pdf)** 
[Wang, Chen and Li]. 
It was collected from CiteULike and Google Scholar. 
CiteULike allows users to create their own collections of articles. 
There are abstracts, titles, and tags for each article. 
It is collected by us independently from the dataset citeulike-a. 
We manually select 273 seed tags and collect all the articles with at least one of these tags. 
We also crawl the citations between the articles from Google Scholar. 
Note that the final number of tags associated with all the collected articles is far more than the number (273) of seed tags.


The text information (item content) of citeulike-a is preprocessed by following the same procedure as that in citeulike-a. After removing the stop words, we choose the top 20000 discriminative words according to the tf-idf values as our vocabulary.

Some statistics are listed as follows:

* users 					7947 
* items 					25975 
* tags 					52946 
* citations 				32565
* user-item pairs 		134860 

## DATA FILES ##
* citations.dat	citations between articles
* tag-item.dat	articles corresponding to tags, one line corresponds to articles relating to the same tags (note that it is different from the other dataset citeulike-a and that this is the version prior to preprocess thus would have more tags than used in the paper)
* mult.dat		bag of words for each article
* rawtext.dat		raw data
* tags.dat		tags, sorted by tag-id's
* users.dat		rating matrix (user-item matrix)
* vocabulary.dat	corresponding words for file mult.dat

```BibTex
@inproceedings{DBLP:conf/ijcai/WangCL13,
  author    = {Hao Wang and
               Binyi Chen and
               Wu-Jun Li},
  title     = {Collaborative Topic Regression with Social Regularization
               for Tag Recommendation},
  booktitle = {IJCAI},
  year      = {2013},
  ee        = {http://www.aaai.org/ocs/index.php/IJCAI/IJCAI13/paper/view/7006},
  crossref  = {DBLP:conf/ijcai/2013},
  bibsource = {DBLP, http://dblp.uni-trier.de}
}
```
```
@proceedings{DBLP:conf/ijcai/2013,
  editor    = {Francesca Rossi},
  title     = {IJCAI 2013, Proceedings of the 23rd International Joint
               Conference on Artificial Intelligence, Beijing, China, August
               3-9, 2013},
  booktitle = {IJCAI},
  publisher = {IJCAI/AAAI},
  year      = {2013},
  isbn      = {978-1-57735-633-2},
  bibsource = {DBLP, http://dblp.uni-trier.de}
}
```
