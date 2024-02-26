import torch
from torch import nn
from torch import Tensor
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention
from layers.ROPEAttentionLayer import ROPEAttentionGroupSequenceIngroupLayer
from layers.Embed import PatchEmbedding,Patching
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
import math

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    

class FixMaskLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    group: int

    def __init__(self, group: int, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=48, stride=2):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        patch_len = configs.patch_len
        stride = configs.stride
        self.d_model = configs.d_model
        padding = stride
        self.sequence_num = configs.sequence_num
        self.enc_in = configs.enc_in
        self.dropout = configs.dropout
        padding = stride

        # patching and embedding
        self.patching = Patching(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.masked_linear = FixMaskLinear(group=8, in_features=patch_len, out_features=configs.d_model)
        
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)       

        self.stride_list = [2**i for i in range(1,configs.sequence_num)]
        self.total_len = int((configs.seq_len - patch_len) / stride + 2)
        group_pos = [[0]*self.total_len]

        for i in range(0,configs.sequence_num - 1):
            self.total_len = self.total_len+math.ceil(self.patch_num/self.stride_list[i])  
            group_pos.append([i+1]* math.ceil(self.patch_num/self.stride_list[i]))
            
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ROPEAttentionGroupSequenceIngroupLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads, group_pos),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
    
        # Prediction Head
        self.head_nf = self.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(self.enc_in, self.head_nf, self.pred_len,
                                    head_dropout = self.dropout)
        padding_list = []
        for i in range(configs.sequence_num - 1):
            if self.patch_num % self.stride_list[i] == 0:
                padding_len = 0
                padding_list.append(padding_len)
                continue

            padding_len = math.ceil((self.stride_list[i] - self.patch_num % self.stride_list[i])/2)
            padding_list.append(padding_len)

        
        self.max_pooling_list = nn.ModuleList([nn.MaxPool1d(self.stride_list[i], stride=self.stride_list[i], padding= padding_list[i]) for i in range(configs.sequence_num - 1)])
        self.sep = Parameter(torch.randn(1,configs.d_model))
        self.conv_transpose_list = nn.ModuleList([nn.ConvTranspose1d(in_channels=self.d_model, out_channels=self.d_model, stride = self.stride_list[i], kernel_size=self.stride_list[i], padding= padding_list[i], output_padding=0, dilation=1, padding_mode="zeros", bias=False) for i in range(configs.sequence_num - 1)])


    def multi_scale(self, x):
        # z: [bs * nvars x d_model x patch_num]
        x_permute = x.permute(0,2,1)
        # z: [bs * nvars x sequence_len x d_model]
        sequence_list = [self.max_pooling_list[i](x_permute).permute(0,2,1) for i in range(self.sequence_num - 1)]
        self.sequence_sep_len = [x.shape[1]]
        self.sequence_sep_len.extend([sequence_list[i].shape[1] for i in range(self.sequence_num - 1)])
        x_embedding = torch.cat((x, *sequence_list), dim=1)
        
        return x_embedding

    def conv_transpose(self, enc_out):
        enc_sequence_list = []
        index = 0
        for i in range(len(self.sequence_sep_len)):
            end = index + self.sequence_sep_len[i]
            enc_sequence_list.append(enc_out[:,index:end,:])
            index = end
        sequence = enc_sequence_list[0]
        for i in range(1,len(enc_sequence_list)):
            conv = self.conv_transpose_list[i-1](enc_sequence_list[i].permute(0,2,1)).permute(0,2,1)
            sequence = sequence + conv
        return sequence


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patching(x_enc)
        # print(enc_out.shape)

        enc_out = self.masked_linear(enc_out)

        enc_out = self.multi_scale(enc_out)
        
        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        
        enc_out = self.conv_transpose(enc_out)
        
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]