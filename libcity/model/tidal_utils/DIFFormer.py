import torch
import torch.nn as nn

def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]
    return output [N, H, D]
    '''
    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2) # [N, H, M]
        ks = ks / torch.norm(ks, p=2) # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs) # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs) # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1) # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer # [N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output
    

def difformer_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [B, N, H, M]
    ks: key tensor [B, L, H, M]
    vs: value tensor [B, L, H, D]
    return output [B, N, H, D]
    '''
    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2) # [B, N, H, M]
        ks = ks / torch.norm(ks, p=2) # [B, L, H, M]
        N = qs.shape[1]
        # print('qs.shape:',qs.shape)
        # print('ks.shape:',ks.shape)
        # print('vs.shape:',vs.shape)
        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)
        # print('kvs.shape:',kvs.shape)
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs) # [N, H, D]
        all_ones = torch.ones([vs.shape[1]]).to(vs.device)
        vs_sum = torch.einsum("l,blhd->bhd", all_ones, vs) # [B, H, D]
        attention_num += vs_sum.unsqueeze(1).repeat(1,vs.shape[1], 1, 1) # [B, N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        ks_sum = torch.einsum("blhm,l->bhm", ks, all_ones)
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [B, N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer # [B, N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks) / attention_normalizer # [B, N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("bnhm,blhm->bnlh", qs, ks))  # [B, N, L, H]

        # denominator
        all_ones = torch.ones([ks.shape[1]]).to(ks.device)
        attention_normalizer = torch.einsum("bnlh,l->bnh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(2).repeat(1, 1, ks.shape[1], 1)  # [B, N, L, H]

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("bnlh,blhd->bnhd", attention, vs)  # [B, N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output

class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''
    def __init__(self, in_channels,
               out_channels,
               num_heads,
               kernel='simple',
               use_weight=True,
               output_attn=False):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_weight = use_weight
        self.output_attn=output_attn

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input):
        # feature transformation
        B,N,D = query_input.shape
        query = self.Wq(query_input).reshape(B, N, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(B, N, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(B, N, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(B, N, 1, self.d)

        # compute full attentive aggregation
        if self.output_attn:
            attention_output, attn = difformer_attention_conv(query, key, value, self.kernel, self.output_attn)  # [B, N, H, d]
        else:
            attention_output = difformer_attention_conv(query,key,value,self.kernel) # [B, N, H, d]

        final_output = attention_output
        final_output = final_output.mean(dim=2)

        if self.output_attn:
            return final_output, attn
        else:
            return final_output, None
        

if __name__ =='__main__':
    x = torch.randn((4,10,64))
    model = DIFFormerConv(64,64,2)
    y,_ =model(x,x)
    print(y.shape)