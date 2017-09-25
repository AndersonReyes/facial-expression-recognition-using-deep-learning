# DeepFaceRecognition
Facial Expression Recognition using tensorflow on the FER2013 dataset using convolutional networks. I got about 50% accurary.

Architecture is very simple:  
Conv [32]  
&darr;  
Max Pool [2x2]  
&darr;  
Conv [64]  
&darr;  
Max Pool [2x2]  
&darr;  
Conv [128]  
&darr;  
Max Pool [2x2]  
&darr;  
Fully connected [1000]  
&darr;  
Softmax
