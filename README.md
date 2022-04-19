# openml-pimp
This work aims to answer the questions:
* Which hyperparameters to optimize
* Over what ranges / priors

## Short Video
[![Youtube Video](https://img.youtube.com/vi/mS4vL7_rSWQ/0.jpg)](https://www.youtube.com/watch?v=mS4vL7_rSWQ)

(Explanation of the work in less than 3 minutes)

## Data
All data in this paper can be reproduced using [this notebook](https://github.com/janvanrijn/openml-pimp/blob/master/KDD2018/results.ipynb). Additionally, [this folder](https://github.com/janvanrijn/openml-pimp/tree/master/KDD2018/data/arff) contains arff files containing a large fraction of the meta-data generated for this project.

## Dockerfile

For running all code in the best way, you need to install container which is inside Dockerfile:
I want to explain what should you do step by step maybe that helps others. then run following code in the terminal where  your Dockerfile is :

`docker build .`
after some minutes the image file will download and your container should be installed for being sure:
`docker ps -a`
when you find it there is not! please run the following code

`sudo docker run -d --name jupyter openml/jupyter-python`

Now you should see your image:
`sudo docker ps -a`

due to your Jupyter maybe run in a different port, it is a good idea to set it in your docker:

```
sudo docker stop jupyter
sudo docker rm jupyter
```
`sudo docker run -d --name jupyter -p 8888:8888 openml/jupyter-python`

Now you can import Openml in your Jupyter just run Jupyter with the following URL:
127.0.0.1:8888


## Paper

[Hyperparameter Importance Across Datasets](https://dl.acm.org/citation.cfm?id=3220058) by Jan N. van Rijn and Frank Hutter [(ArXiv)](https://arxiv.org/abs/1710.04725)

## Bibtex
If you find this work useful, please cite:
```
@inproceedings{Rijn2018Hyperparameter,
  title        = {Hyperparameter Importance Across Datasets},
  author       = {van Rijn, Jan N. and Hutter, Frank},
  booktitle    = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year         = {2018},
  organization = {ACM},
  pages        = {2367--2376}
}
```

