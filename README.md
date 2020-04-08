# S7---Advanced-Convolutions  
Advanced Convolutions like  
1. Dilated convolutions with maximum dilation = 2
2. Groupwise separable convolution  
have been implemented on the CIFAR-10 dataset.
### Results achieved -  
83.59% best test accuracy in 20 epochs. 

## Model Architecture
Conv-16 --> Conv-32 --> Transition1(Maxpool+Conv-32) --> Conv-64 --> Dilated Conv-64 --> Transition2(Maxpool + Conv32)  
--> Conv-64 --> Depth-wise, groups-64 Conv-128 --> Transition3(maxpool+Conv-64) --> Conv-64 --> Conv-128 -->  
GAP --> Fully Connected(128->1)
