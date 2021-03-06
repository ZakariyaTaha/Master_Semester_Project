from .dru_utils3D import *


class druv2(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 ):
        super(druv2, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = int(hidden_size * 8 / self.feature_scale)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        filters = [32, 64, 128, 256]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + 1, filters[0], self.is_batchnorm)
        ##ss self.conv1 = unetConv2(self.in_channels + self.n_classes, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # convgru
        assert(self.hidden_size == filters[3])
        self.gru = ConvDRU(filters[2], self.hidden_size)
        # self.softmax = nn.Softmax2d()
        # final conv (without any concat)
        self.conv_down = nn.Conv3d(filters[0], n_classes, 1)
        # self.conv_down_1 = nn.Conv2d(filters[1], n_classes, 1)

    def forward(self, inputs, h, s):
        list_st = []
        list_st_1 = []
        for i in range(self.steps):

            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

#             conv4 = self.conv4(maxpool3)
#             maxpool4 = self.maxpool4(conv4)

            h = self.gru(maxpool3, h)
#             up4 = self.up_concat4(conv4, h)
            up3 = self.up_concat3(conv3, h)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            s = self.conv_down(up1)
            # out_1 = self.conv_down_1(up2)
            list_st += [s]
            # list_st_1 += [out_1]
            # s = self.softmax(out)
            # s = s[:,1:2] ##ss added
        return list_st #, list_st_1

class druv2_relu(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 ):
        super(druv2_relu, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = int(hidden_size * 8 / self.feature_scale)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        filters = [32, 64, 128, 256]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + 1, filters[0], self.is_batchnorm)
        ##ss self.conv1 = unetConv2(self.in_channels + self.n_classes, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # convgru
        assert(self.hidden_size == filters[3])
        self.gru = ConvDRU(filters[2], self.hidden_size)
        # self.softmax = nn.Softmax2d()
        # final conv (without any concat)
        self.conv_down = nn.Conv3d(filters[0], n_classes, 1)
        # self.conv_down_1 = nn.Conv2d(filters[1], n_classes, 1)

    def forward(self, inputs, h, s):
        list_st = []
        list_st_1 = []
        for i in range(self.steps):

            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

#             conv4 = self.conv4(maxpool3)
#             maxpool4 = self.maxpool4(conv4)

            h = self.gru(maxpool3, h)
#             up4 = self.up_concat4(conv4, h)
            up3 = self.up_concat3(conv3, h)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            s = F.relu(self.conv_down(up1))
            # out_1 = self.conv_down_1(up2)
            list_st += [s]
            # list_st_1 += [out_1]
            # s = self.softmax(out)
            # s = s[:,1:2] ##ss added
        return list_st #, list_st_1

class dru(nn.Module):
    def __init__(self,
                 args,
                 n_classes=19,
                 initial=1,
                 steps=3,
                 gate=3,
                 hidden_size=32,
                 feature_scale=1,
                 is_deconv=True,
                 in_channels=3,
                 is_batchnorm=True,
                 ):
        super(dru, self).__init__()
        self.args = args
        self.steps = steps
        self.feature_scale = feature_scale
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.is_batchnorm = is_batchnorm
        self.is_deconv = is_deconv

        filters = [32, 64, 128, 256, 512]  # [8, 16, 32, 64, 128] [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels + self.n_classes, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)

        # self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # convgru
        assert(self.hidden_size == filters[4])
        self.gru = ConvDRU(filters[3], self.hidden_size)

        # final conv (without any concat)
        self.conv_down = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs, h, s):
        list_st = []
        for i in range(self.steps):
            stacked_inputs = torch.cat([inputs, s], dim=1)

            conv1 = self.conv1(stacked_inputs)
            maxpool1 = self.maxpool1(conv1)

            conv2 = self.conv2(maxpool1)
            maxpool2 = self.maxpool2(conv2)

            conv3 = self.conv3(maxpool2)
            maxpool3 = self.maxpool3(conv3)

            conv4 = self.conv4(maxpool3)
            maxpool4 = self.maxpool4(conv4)

            h = self.gru(maxpool4, h)
            up4 = self.up_concat4(conv4, h)
            up3 = self.up_concat3(conv3, up4)
            up2 = self.up_concat2(conv2, up3)
            up1 = self.up_concat1(conv1, up2)

            s = self.conv_down(up1)
            list_st += [s]

        return list_st


class ConvDRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvDRU, self).__init__()

        self.reset_gate = unetConv2(input_size, input_size, True)
        self.update_gate = unetConv2(input_size, hidden_size, True)
        self.out_gate = unetConv2(input_size, hidden_size, True)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        reset = torch.sigmoid(self.reset_gate(input_))
        # print('input_, reset, h, shape ', input_.shape, reset.shape, h.shape)
        # stacked_inputs_ = torch.cat([input_, h * reset], dim=1)

        out_inputs = torch.tanh(self.out_gate(input_ * reset))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())


class ConvDualRU(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 ):
        super(ConvDualRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reset_gate = unetConv2(input_size, hidden_size, True)
        self.update_gate = unetConv2(input_size, hidden_size, True)
        self.out_gate = unetConv2(input_size + hidden_size, hidden_size, True)

    def forward(self, input_, h=None):
        # batch_size = input_.data.size()[0]
        # spatial_size = input_.data.size()[2:]

        # data size is [batch, channel, height, width]
        # print('input_.type', input_.data.type())
        # print('prev_state.type', prev_state.data.type())

        update = torch.sigmoid(self.update_gate(input_))
        reset = torch.sigmoid(self.reset_gate(input_))
        # print('input_, reset, h, shape ', input_.shape, reset.shape, h.shape)
        stacked_inputs_ = torch.cat([input_, h * reset], dim=1)

        out_inputs = torch.tanh(self.out_gate(stacked_inputs_))
        h_new = h * (1 - update) + out_inputs * update
        return h_new

    def __repr__(self):
        return 'ConvDRU: \n' + \
               '\t reset_gate: \n {}\n'.format(self.reset_gate.__repr__()) + \
               '\t update_gate: \n {}\n'.format(self.update_gate.__repr__()) + \
               '\t out_gate:  \n {}\n'.format(self.out_gate.__repr__())
