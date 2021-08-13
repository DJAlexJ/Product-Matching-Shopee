# Product-Matching-Shopee
Product matching based on images and titles.

Matching is a problem of finding similar products based on their images, titles, descriptions, set of attributes etc. Here we have goods' titles and its images. Intuitively, we need to get somehow vector representations of each product and then find the closest ones. Those pairs would be the candidates for possible matches. 


This solution is based on BERT-like models for getting text embeddings and CV models for image embeddings. The whole process is following:
1. Getting embeddings from images (NFnet-l0, EfficientNet-b0, however, you can you use any model from timm library, because the training code is rather flexible). Both networks were trained on 512 images, but during inference they used images with slightly increase size - 544. Networks were trained using [arcface module](https://arxiv.org/pdf/1801.07698.pdf)
2. Getting text embeddings from titles with Indonesian DistilBert, Paraphrase Mutlingual SBERT Model and tf-idf. (main english words were translated into indonesian) 
3. Embeddings preprocessing (a.k.a. query expansion, you can check this in `inference_notebook.ipynb` for more details)
4. Postprocessing: using knn for image/text embeddings and then combining neighbors to build final set of matches candidates.
5. Applying adaptive thresholds to get final predictions. (We use adaptive approach in order to find a balance between False-positives and False-negatives for separate cases. However, you can use here more complex model like LightGBM to determine whether a particular pair is a match)

<img width="983" alt="image" src="https://user-images.githubusercontent.com/18465332/129375976-c60eb3d0-6db0-455f-a36b-beaf57b4b596.png">

Running train script can be done as follows:
1. Specify paths to data and hyperparameters values in `config.py`
2. Run `train.py`
