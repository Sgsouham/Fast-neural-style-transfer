# Fast-neural-style-transfer


## Objective
I wanted to re-implement the [original](https://arxiv.org/abs/1508.06576) Neural Style Transfer paper from scratch but the problem with that was the per pixel loss made the results pleasaing to the eye but not very fast.

So I went forward with the perceptual loss from the [Fast Neural Style Transfer](https://cs.stanford.edu/people/jcjohns/eccv16/) as it solves
 - The NST in real time.
 - Super resolution for crisp results.
 
## The implementation idea for those who are new to paper implementations (like me!)

So first after reading the paper, I tried understanding the architecture of the system.

![Architecture](images/model.png)

As we can see, the entire model comprises of two main part, the image transformation network and the vgg network.


The image transformation network used in the original paper is [given in here](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16Supplementary.pdf)

So I started of making two seperate python files for the two different models.
 - loss_net.py
 - transformer_net.py
As the name suggests, on reading the paper, the meaning of each will be better umderstood as the loss_net.py is the vgg model and the transformer_net.py is the proposed model for the super resolution and feature extraction purpose.


the utils.py is just a simple python file which contains all the necessary helper functions like converting the images into nparrays and stylizing etc.

The Main file used to train the network is the nst_train.py

