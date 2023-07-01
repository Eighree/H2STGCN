import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_Block(nn.Module):
    def __init__(self, step, dim_in, dim_out, input_window, order=1):
        super(ST_Block, self).__init__()
        self.order = order
        self.dim_in = dim_in * order
        self.dim_out = dim_out
        # matrix for temporal dimension
        padding = nn.ConstantPad2d((0, step, step, 0), 0)
        self.seq_matrix = torch.eye(input_window) + padding(torch.eye(input_window - step))
        self.mlp = torch.nn.Conv2d(self.dim_in, self.dim_out, kernel_size=(1, 1),
                                   padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, sp_matrix):
        # aggregation temporal information
        seq_matrix = self.seq_matrix.to(sp_matrix.device)
        x_t = torch.einsum('tk,bink->bint', seq_matrix, x)
        # aggregation spatial information
        out = []
        for k in range(self.order):
            x_st = torch.einsum('ncvl,vw->ncwl', x_t, sp_matrix)
            out.append(x_st)
            x_t = x_st
        # spatial-temporal feature transformation
        h_st = torch.cat(out, dim=1)
        h_st = self.mlp(h_st)
        return h_st


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), x.shape[2:])
        x2 = F.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)  # (1, 3C, H, W)
        fusion = self.conv(concat)

        return fusion


class JSTGCN(nn.Module):
    def __init__(self, args):
        super(JSTGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.residual_dim = 48
        self.dilation_dim = 48
        self.skip_dim = 48 * 3
        self.end_dim = 48 * 6
        self.input_window = args.horizon
        self.output_window = args.horizon
        self.device = torch.device('cuda:0')

        self.layers = [1, 2, 4, 4]

        self.node_embedding = nn.Parameter(torch.randn(2, self.num_node, 10).to(self.device),
                                           requires_grad=True).to(self.device)
        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=self.residual_dim * 2,
                                    kernel_size=(1, 1))

        self.filter_similar = nn.ModuleList()
        self.filter_compete = nn.ModuleList()
        self.filter_mix = nn.ModuleList()

        self.mix_similar = nn.ModuleList()
        self.mix_compete = nn.ModuleList()

        self.conv_mix = nn.ModuleList()

        self.dis_similar = nn.ModuleList()
        self.dis_compete = nn.ModuleList()

        self.bn_similar = nn.ModuleList()
        self.bn_compete = nn.ModuleList()
        #self.bn_mix = nn.ModuleList()

        self.skip_convs = nn.ModuleList()


        for i in self.layers:
            self.filter_similar.append(ST_Block(i, self.residual_dim, self.dilation_dim * 2, self.input_window))
            self.filter_compete.append(ST_Block(i, self.residual_dim, self.dilation_dim * 2, self.input_window))
            self.filter_mix.append(ST_Block(i, self.residual_dim, self.dilation_dim * 2, self.input_window))

            self.mix_compete.append(MSC(self.dilation_dim))
            self.mix_similar.append(MSC(self.dilation_dim))

            self.conv_mix.append(nn.Conv2d(in_channels=self.dilation_dim * 3,
                                           out_channels=self.dilation_dim * 2,
                                           kernel_size=(1, 1)))

            self.dis_similar.append(nn.Conv2d(in_channels=self.dilation_dim,
                                              out_channels=self.dilation_dim,
                                              kernel_size=(1, 1)))
            self.dis_compete.append(nn.Conv2d(in_channels=self.dilation_dim,
                                              out_channels=self.dilation_dim,
                                              kernel_size=(1, 1)))

            self.bn_similar.append(nn.BatchNorm2d(self.residual_dim))
            self.bn_compete.append(nn.BatchNorm2d(self.residual_dim))
            #self.bn_mix.append(nn.BatchNorm2d(self.residual_dim))

            self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_dim,
                                             out_channels=self.skip_dim,
                                             kernel_size=(1, 1)))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_dim,
                                    out_channels=self.end_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_dim,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        inputs = source
        inputs = inputs.permute(0, 3, 2, 1)
        x = self.start_conv(inputs)
        x_similar, x_compete = torch.split(x, self.residual_dim, dim=1)
        x_mix = torch.zeros(x_similar.shape).cuda()
        skip = 0

        A_similar = torch.tanh(F.relu(torch.mm(self.node_embedding[0], self.node_embedding[0].T)))
        A_compete = torch.tanh(F.relu(-torch.mm(self.node_embedding[1], self.node_embedding[1].T)))

        D_similar = torch.sum(A_similar, dim=-1)
        D_similar[D_similar < 10e-5] = 10e-5

        D_compete = torch.sum(A_compete, dim=-1)
        D_compete[D_compete < 10e-5] = 10e-5

        diag_similar = torch.diag(torch.reciprocal(torch.sqrt(D_similar)))
        matrix_similar_m = torch.mm(torch.mm(diag_similar, A_similar), diag_similar)
        matrix_similar = torch.eye(self.num_node).to(self.device) + matrix_similar_m

        diag_compete = torch.diag(torch.reciprocal(torch.sqrt(D_compete)))
        matrix_compete_m = torch.mm(torch.mm(diag_compete, A_compete), diag_compete)
        matrix_compete = torch.eye(self.num_node).to(self.device) - matrix_compete_m

        matrix_mix = torch.eye(self.num_node).to(self.device) + matrix_similar_m - matrix_compete_m

        torch.save(matrix_similar, "data_04_pos_3_22_2.pt")
        torch.save(matrix_compete, "data_04_neg_3_22_2.pt")

        for i in range(len(self.layers)):
            residual_similar = x_similar
            residual_compete = x_compete

            s_f, s_g = torch.split(self.filter_similar[i](residual_similar, matrix_similar), self.dilation_dim, dim=1)
            x_similar = s_f * torch.sigmoid(s_g)

            c_f, c_g = torch.split(self.filter_compete[i](residual_compete, matrix_compete), self.dilation_dim, dim=1)
            x_compete = c_f * torch.sigmoid(c_g)

            x_similar = self.mix_similar[i](x_similar)
            x_compete = self.mix_compete[i](x_compete)



            s = self.skip_convs[i](x_compete[:, :, :, -1:])
            skip = skip + s

            x_similar = x_similar + residual_similar
            x_compete = x_compete + residual_compete
            #x_mix = x_mix + residual_mix

            x_similar = self.bn_similar[i](x_similar)
            x_compete = self.bn_compete[i](x_compete)
            #x_mix = self.bn_mix[i](x_mix)

        x = F.relu(skip[:, :, :, -1:])
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x, matrix_similar, matrix_compete
