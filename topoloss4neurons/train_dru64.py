# Setup Model

from model import dru as dru_nets
print ('building models ...')
hidden_size = 1024
net = dru_nets.druv2(args,n_classes=2, feature_scale = 0.5).cuda()


data, label = batch

sdatav = Variable(data).cuda()
slabelv = Variable(label).cuda()

h0, s0 = generate_sh_inits_wei(sdatav, hidden_size = hidden_size)

s_pred1, s_pred2 = enc_shared(sdatav, h0, s0) 

def generate_sh_inits_wei(images, hidden_size = 256):
    n_classes=1
    W, H = 512, 512
    w = int(np.floor(np.floor(np.floor(W/2)/2)/2)/2) ##ss assuming 4 x /2 resolution reduction at the point of recursion
    h = int(np.floor(np.floor(np.floor(H/2)/2)/2)/2)

    h0 = torch.ones([images.shape[0], hidden_size, w, h],
                                dtype=torch.float32)

    s0 = torch.ones([images.shape[0], n_classes, W, H],
                                dtype=torch.float32) ##ss changed from ones to zeros

    return h0.cuda(), s0.cuda()
