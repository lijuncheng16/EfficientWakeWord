# Dissecting Efficient Convolution Networks for Wake Word Detection

This repo hosts the code and models of "Dissecting Efficient Convolution Networks for Wake Word Detection" (submitted to ICASSP 2023)

Authors: Cody Berger*, Juncheng B Li*, Karthik Ganesan, Aaron Berger, Dmitri Berger, Florian Metze (*: Co-first author)

Appendix.pdf contains additional details about the models, training that was submitted. 

# Instructions:
Please follow the train script ```train.py``` to train models. 
```test.py``` is for testing the models in a non-streaming fashion. Non-streaming's definition can be found in our paper.
Times are logged with PyTorch forward prehooks and hooks.



