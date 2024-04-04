# 我下面写的是没有带发放率和T的。
# 计算量
def conv(inchannel,outchannel,kernel,P,stride,H,W,batch):
    # 一次卷积的计算量 乘加运算
    cal = 2*kernel*kernel-1
    # inchannel个特征图上执行卷积需要进行卷积的次数
    times = inchannel*((H-kernel+P)/stride + 1)*((W-kernel+P)/stride + 1)
    total = (inchannel-1)*outchannel*batch*times*cal
    return total

# 乘法
def conv_mult(inchannel,outchannel,kernel,P,stride,H,W,batch):
    # 一次卷积的计算量 乘运算
    cal = kernel*kernel
    # inchannel个特征图上执行卷积需要进行卷积的次数
    times = inchannel*((H-kernel+P)/stride + 1)*((W-kernel+P)/stride + 1)
    total = outchannel*batch*times*cal
    return total

# 加法
def conv_plus(inchannel,outchannel,kernel,P,stride,H,W,batch):
    # 一次卷积的计算量 乘加运算
    cal = kernel*kernel-1
    # inchannel个特征图上执行卷积需要进行卷积的次数
    times = inchannel*((H-kernel+P)/stride + 1)*((W-kernel+P)/stride + 1)
    total = (inchannel-1)*outchannel*batch*times*cal
    return total