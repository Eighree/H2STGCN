import torch
import torch.nn as nn
import torch.nn.functional as F

class ST_Block(nn.Module):
    def __init__(self, k, dim_in, dim_out, input_window, p_or_n, order=2):
        super(ST_Block, self).__init__()
        self.order = order
        self.dim_in = dim_in * order
        self.dim_out = dim_out
        self.k = k
        self.p_or_n = p_or_n  # 1: positive -1: negative
        self.input_window = input_window

        self.mlp1 = torch.nn.Conv2d(self.dim_in, self.dim_out, kernel_size=(1, 1),
                                    padding=(0, 0), stride=(1, 1), bias=True)

        self.mlp2 = torch.nn.Conv2d(self.dim_in, self.dim_out, kernel_size=(1, 1),
                                    padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x, node_embedding_t, node_embedding):
        inner_product_t = self.p_or_n * torch.matmul(node_embedding_t[:,self.k:,:,:],
                                                     node_embedding[:,self.k:,:,:].permute(0, 1, 3, 2))
        A_t = torch.eye(x.shape[2]).to(node_embedding.device) + self.p_or_n * F.normalize(
            torch.tanh(inner_product_t * (inner_product_t > 0.00)), p=1, dim=-1)
        x_t = x[:, :, :, self.k:]

        inner_product_t_k = self.p_or_n * torch.matmul(node_embedding_t[:,:(self.input_window - self.k),:,:],
                                                       node_embedding[:,self.k:,:,:].permute(0, 1, 3, 2))
        A_t_k = torch.eye(x.shape[2]).to(node_embedding.device) + self.p_or_n * F.normalize(
            torch.tanh(inner_product_t_k * (inner_product_t_k > 0.00)), p=1, dim=-1)
        x_t_k = x[:, :, :, :(self.input_window - self.k)]

        out_1 = [x_t]
        out_1.append(torch.einsum('ncvl,nlvw->ncwl', x_t, A_t))
        h_1 = self.mlp1(torch.cat(out_1, dim=1))

        out_2 = [x_t_k]
        out_2.append(torch.einsum('ncvl,nlvw->ncwl', x_t_k, A_t_k))
        h_2 = self.mlp2(torch.cat(out_2, dim=1))

        h_st = h_1 + h_2
        h_st = F.pad(h_st, [self.k, 0, 0, 0])

        return h_st



class TEmbedding(nn.Module):
    '''
    TE:     [batch_size, num_his, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    C:      candidate_group
    retrun: [batch_size, num_his, C]
    '''
    def __init__(self, D, candidate_group):
        super(TEmbedding, self).__init__()
        self.mlp1 = torch.nn.Conv1d(295, D, kernel_size=1, padding=0, bias=True)
        self.mlp2 = torch.nn.Conv1d(D, candidate_group, kernel_size=1, padding=0, bias=True)

    def forward(self, TE, T=288):
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 7).to(TE.device)  # B T 7
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T).to(TE.device)  # B T 288
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 7, 7)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 288, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1)  # B T 295
        TE = F.relu(self.mlp1(TE.permute(0, 2, 1)))  # B D T
        TE = F.relu(self.mlp2(TE)).permute(0, 2, 1)  # B T C
        del dayofweek, timeofday
        return TE

        return TE


class MSC(nn.Module):
    def __init__(self, channels):
        super(MSC, self).__init__()
        self.channels = channels
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))

        self.conv = nn.Sequential(
            nn.Conv2d(3*channels, channels, kernel_size=1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = F.interpolate(self.pool1(x), x.shape[2:])
        x2 = F.interpolate(self.pool2(x), x.shape[2:])
        concat = torch.cat([x, x1, x2], 1)
        fusion = self.conv(concat)
        return fusion


class H2STGCN(nn.Module):
    def __init__(self, args):
        super(H2STGCN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.residual_dim = args.hideen_dim
        self.dilation_dim = args.hideen_dim
        self.skip_dim = args.hideen_dim * 4
        self.end_dim = args.hideen_dim * 6
        self.embed_dim = args.embed_dim
        self.candidate_group = args.candidate_group
        self.input_window = args.horizon
        self.output_window = args.horizon
        self.device = torch.device('cuda:0')

        self.layers = [1, 2, 4, 4]

        self.tembedding = TEmbedding(args.hideen_dim*2, args.candidate_group)

        self.p_interaction = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim).to(self.device),
                                          requires_grad=True).to(self.device)

        self.n_interaction = nn.Parameter(torch.randn(self.embed_dim, self.embed_dim).to(self.device),
                                          requires_grad=True).to(self.device)

        self.node_embedding = nn.Parameter(
            torch.randn(self.candidate_group, self.num_node, self.embed_dim).to(self.device),
            requires_grad=True).to(self.device)

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=self.residual_dim * 2,
                                    kernel_size=(1, 1))

        self.filter_similar = nn.ModuleList()
        self.filter_compete = nn.ModuleList()

        self.mix_similar = nn.ModuleList()
        self.mix_compete = nn.ModuleList()

        self.conv_mix = nn.ModuleList()

        self.dis_similar = nn.ModuleList()
        self.dis_compete = nn.ModuleList()

        self.skip_convs = nn.ModuleList()
        self.bn_similar = nn.ModuleList()
        self.bn_compete = nn.ModuleList()
        self.bn_mix = nn.ModuleList()


        for i in self.layers:
            self.filter_similar.append(
                ST_Block(i, self.residual_dim, self.dilation_dim * 2, self.input_window, p_or_n=1))
            self.filter_compete.append(
                ST_Block(i, self.residual_dim, self.dilation_dim * 2, self.input_window, p_or_n=-1))

            self.mix_compete.append(MSC(self.dilation_dim))
            self.mix_similar.append(MSC(self.dilation_dim))

            self.conv_mix.append(nn.Conv2d(in_channels=self.dilation_dim * 2,
                                           out_channels=self.dilation_dim * 2,
                                           kernel_size=(1, 1)))

            self.dis_similar.append(nn.Conv2d(in_channels=self.dilation_dim,
                                              out_channels=self.dilation_dim,
                                              kernel_size=(1, 1)))
            self.dis_compete.append(nn.Conv2d(in_channels=self.dilation_dim,
                                              out_channels=self.dilation_dim,
                                              kernel_size=(1, 1)))

            self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_dim,
                                             out_channels=self.skip_dim,
                                             kernel_size=(1, 1)))

            self.bn_similar.append(nn.BatchNorm2d(self.residual_dim))
            self.bn_compete.append(nn.BatchNorm2d(self.residual_dim))
            self.bn_mix.append(nn.BatchNorm2d(self.residual_dim))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_dim,
                                    out_channels=self.end_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_dim,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        inputs = source[:, :, :, 0:1]
        te = source[:, :, 1, 1:]  # B T C
        inputs = inputs.permute(0, 3, 2, 1)
        x = self.start_conv(inputs)
        x_similar, x_compete = torch.split(x, self.residual_dim, dim=1)
        #x_mix = torch.zeros(x_similar.shape).cuda()

        skip = 0

        temb = self.tembedding(te) # b t c

        #temp = self.temp.repeat(32,1,1)

        T_embedding = torch.einsum('btc,cnf->btnf', temb, self.node_embedding)  #  B T N E
        T_embedding_p = torch.einsum('btnf,fp->btnp', T_embedding, self.p_interaction)  # B T N E
        T_embedding_n = torch.einsum('btnf,fp->btnp', T_embedding, self.n_interaction)  # B T N E

        for i in range(len(self.layers)):
            residual_similar = x_similar
            residual_compete = x_compete
            #residual_mix = x_mix

            s_f, s_g = torch.split(self.filter_similar[i](residual_similar, T_embedding_p, T_embedding), self.dilation_dim, dim=1)
            x_similar = s_f * torch.sigmoid(s_g)

            c_f, c_g = torch.split(self.filter_compete[i](residual_compete, T_embedding_n, T_embedding), self.dilation_dim, dim=1)
            x_compete = c_f * torch.sigmoid(c_g)

            x_mix_similar = self.mix_similar[i](x_similar)
            x_mix_compete = self.mix_compete[i](x_compete)

            x_mix_fuse_f, x_mix_fuse_g = torch.split(
                self.conv_mix[i](torch.cat([x_mix_similar, x_mix_compete], dim=1)), self.dilation_dim,
                dim=1)
            x_mix = x_mix_fuse_f * torch.sigmoid(x_mix_fuse_g)

            dis_similar = x_mix - x_mix_similar
            dis_compete = x_mix - x_mix_compete

            x_similar = x_similar + torch.sigmoid(self.dis_similar[i](dis_similar)) * dis_similar
            x_compete = x_compete + torch.sigmoid(self.dis_compete[i](dis_compete)) * dis_compete

            s = self.skip_convs[i](x_mix[:, :, :, -1:])
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
        return x

