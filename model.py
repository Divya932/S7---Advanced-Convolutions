import torch.nn as nn


class Net(nn.Module):
    '''def create_depthwise_conv2d(self, in_channels, out_channels, kernel_size=(3,3), dilation=1, padding=1, bias=False, bn=True, dropout=0, relu=True, padding_mode="zeros"):
        #return self.activate(self.separable_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding=padding, bias=bias, padding_mode=padding_mode),
                 #out_channels, bn, dropout, relu)
        return(nn.Sequential(nn.separable_conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=k_size,padding = padding,dilation = dilation,bias = bias, padding_mode=p_mode ),
                             nn.BatchNorm2d(out_channels),
                             nn.Dropout(0.10),
                             nn.ReLU(),))'''
        

    def __init__(self):
        super(Net, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (3,3),
                                                            stride = 1,padding = 1,dilation = 1,groups = 1,
                                                                      bias = False, padding_mode="zeroes"),                    #inp_dim = 32x32x3 | out_dim = 32x32x16 | r.f. = 3
                                         nn.BatchNorm2d(16),
                                         nn.Dropout(0.10),
                                         nn.ReLU(),

                                         nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), 
                                                                     stride = 1, padding = 1, dilation = 1),                   #inp_dim = 32x32x16 | out_dim = 32x2x32 | r.f. = 3 + (3-1)*1 = 5
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(),)
            
        
        self.transition1 = nn.Sequential(nn.MaxPool2d(2,2),                                                                    #inp_dim = 32x32x32 | out_dim = 16x16x32 | r.f. = 5+((2-1)*1) = 6
                                         nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),bias = False,padding =1))  #inp_dim = 16x16x32 | out_dim = 16x16x32 | r.f. = 6+((3-1)*2) = 10
        
        
        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3),
                                                            stride = 1,padding = 1,dilation = 1,groups = 1,
                                                                      bias = False, padding_mode="zeroes"),                    #inp_dim = 16x16x32 | out_dim = 16x16x64 | r.f. = 10+((2)*2) = 14
                                         nn.BatchNorm2d(64),
                                         nn.Dropout(0.10),
                                         nn.ReLU(),
                                         
                                         nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), stride = 1,       #inp_dim = 16x16x64 | out_dim = 16x16x64 | r.f. = 14 + (5-1)*2 = 22
                                                                                             padding = 1, dilation = 2),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())
        
        self.transition2 = nn.Sequential(nn.MaxPool2d(2,2),                                                                    #inp_dim = 16x16x64 | out_dim = 8x8x64   | r.f. = 22+((2-1)*2) = 24
                                         nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1),
                                                   bias = False,padding =0,dilation = 1))                                      #inp_dim = 8x8x64   | out_dim = 8x8x32   | r.f. = 24+(0*4) = 24

        
#Depth-wise Separable layer
        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding = 1,      #inp_dim = 8x8x32   | out_dim = 8x8x64   | r.f. = 24+(2*4) = 32
                                                   bias = False, padding_mode = "zeroes"),
                                         nn.BatchNorm2d(64),
                                         nn.Dropout(0.10),
                                         nn.ReLU(),
            
                                         nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3), stride = 1,       #inp_dim = 8x8x64 | out_dim = 8x8x64 | r.f. = 32 + (3-1)*4 = 40
                                                   padding = 1,dilation = 1,groups = 64, bias = False, padding_mode="zeroes"), 
                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (1,1), stride = 1,       
                                                                                             padding = 0),                      #inp_dim = 8x8x64 | out_dim = 8x8x128 | r.f. = 40
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),)
        
        self.transition3 = nn.Sequential(nn.MaxPool2d(2,2),                                                                     #inp_dim = 8x8x128 | out_dim = 4x4x128   | r.f. = 40+((2-1)*4) = 44
                                         nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(1,1),                           #inp_dim = 4x4x128 | out_dim = 4x4x64   | r.f. = 44
                                                   bias = False,padding =0))


        self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3,3),
                                                            stride = 1,padding = 1,dilation = 1,groups = 1,
                                                                      bias = False, padding_mode="zeroes"),                    #inp_dim = 4x4x64 | out_dim = 4x4x64  | r.f. = 44+((3-1)*8) = 60
                                         nn.BatchNorm2d(64),
                                         nn.Dropout(0.10),
                                         nn.ReLU(),

                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), stride = 1,       #inp_dim = 4x4x64 | out_dim = 4x4x128 | r.f. = 60 + (3-1)*8 = 76
                                                                                             padding = 1, dilation = 1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),)
                                         
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)  # In: 4x4x128 | Out: 1x1x128 | RF: 76
        self.layer5 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.transition1(x)
        x = self.conv_block2(x)
        x = self.transition2(x)
        x = self.conv_block3(x)
        x = self.transition3(x)
        x = self.conv_block4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.layer5(x)

        return x


#net = Net()