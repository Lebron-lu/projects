import torch.nn as nn
import torch

class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )
    def forward(self, x):
        return self.block(x)

    
    
class square(nn.Module):
    def __init__(self):
        super(square, self).__init__()
    def forward(self,x):
        return torch.pow(x, 2)
    
class mean(nn.Module):
    def __init__(self):
        super(mean, self).__init__()
    def forward(self,x):
        return torch.mean(x)
    
class mean2(nn.Module):
    def __init__(self):
        super(mean2, self).__init__()
    def forward(self,x):
        return torch.mean(x, dim=2)
    
class div(nn.Module):
    def __init__(self):
        super(div, self).__init__()
    def forward(self,x, y):
        return torch.div(x, y)
    
class cat(nn.Module):
    def __init__(self):
        super(cat, self).__init__()
    def forward(self,x):
        return torch.cat(x,1)

    
class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, phase, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.phase = phase
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1) #0
        self.ly1 = nn.BatchNorm2d(num_features=64)
        self.ly2 = nn.ReLU()  #2
        self.ly3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1))
        self.ly4 = small_basic_block(ch_in=64, ch_out=128) #4
        self.ly5 = nn.BatchNorm2d(num_features=128)
        self.ly6 = nn.ReLU() #6
        self.ly7 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2))
        self.ly8 = small_basic_block(ch_in=64, ch_out=256) #8
        self.ly9 = nn.BatchNorm2d(num_features=256)
        self.ly10 = nn.ReLU() #10
        self.ly11 = small_basic_block(ch_in=256, ch_out=256)
        self.ly12 = nn.BatchNorm2d(num_features=256) #12 
        self.ly13 = nn.ReLU()
        self.ly14 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2))
        self.ly15 = nn.Dropout(dropout_rate)
        self.ly16 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1)
        self.ly17 = nn.BatchNorm2d(num_features=256)
        self.ly18 = nn.ReLU()
        self.ly19 = nn.Dropout(dropout_rate)
        self.ly20 = nn.Conv2d(in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1)
        self.ly21 = nn.BatchNorm2d(num_features=class_num)
        self.ly22 = nn.ReLU()  # 22
        
        self.ly23 = nn.Conv2d(in_channels=448+self.class_num, out_channels=self.class_num, kernel_size=(1, 1), stride=(1, 1))
        self.ap1 = nn.AvgPool2d(kernel_size=5, stride=5)
        self.ap2 = f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))
        self.square = square()
        self.mean = mean()
        self.div = div()
        self.cat = cat()
        self.output = mean2()
        
    def forward(self, input):
        x0 = self.input(input)
        x1 = self.ly1(x0)
        x2 = self.ly2(x1)
        x3 = self.ly3(x2)
        x4 = self.ly4(x3)
        x5 = self.ly5(x4)
        x6 = self.ly6(x5)
        x7 = self.ly7(x6)
        x8 = self.ly8(x7)
        x9 = self.ly9(x8)
        x10 = self.ly10(x9)
        x11 = self.ly11(x10)
        x12 = self.ly12(x11)
        x13 = self.ly13(x12)
        x14 = self.ly14(x13)
        x15 = self.ly15(x14)
        x16 = self.ly16(x15)
        x17 = self.ly17(x16)
        x18 = self.ly18(x17)
        x19 = self.ly19(x18)
        x20 = self.ly20(x19)
        x21 = self.ly21(x20)
        x22 = self.ly22(x21)
        
        x2 = self.ap1(x2)
        x2_pow = self.square(x2)
        x2_mean = self.mean(x2_pow)
        x2 = self.div(x2, x2_mean)
        
        x6 = self.ap1(x6)
        x6_pow = self.square(x6)
        x6_mean = self.mean(x6_pow)
        x6 = self.div(x6, x6_mean)
        
        x13 = self.ap2(x13)
        x13_pow = self.square(x13)
        x13_mean = self.mean(x13_pow)
        x13 = self.div(x13, x13_mean)
        
        x22_pow = self.square(x22)
        x22_mean = self.mean(x22_pow)
        x22 = torch.div(x22, x22_mean)
        
        x = self.cat([x2, x6, x13, x22])
        x = self.ly23(x)
        output = self.output(x)
        
        return output

def build_lprnet(lpr_max_len=8, phase=False, class_num=66, dropout_rate=0.5):

    Net = LPRNet(lpr_max_len, phase, class_num, dropout_rate)

    if phase == "train":
        return Net.train()
    else:
        return Net.eval()
