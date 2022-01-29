from pesq import pesq

'''
loss function 
'''


def loss_pesq(rate, ori, denoise):
    return pesq(rate, ori, denoise)
