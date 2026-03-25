#模型部分，训练未作
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#总体框架
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed

        self.generator = generator
    def forward(self,src,tgt, src_mask, tgt_mask):
        return self.decode(
            self.encode(src, src_mask),
            src_mask,
            tgt,
            tgt_mask
        )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# 堆叠层数
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self,feature,eps=1e-6):
        super().__init__()
        # a_2,b_2可以让强行做归一化后保持稳定与平衡
        self.a_2=nn.Parameter(torch.ones(feature))
        self.b_2=nn.Parameter(torch.zeros(feature))
        self.eps=eps

    def forward(self,x):
        # 沿最后一个维度，不做维度变换
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps) +self.b_2


#残差连接
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super().__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))



#缩放点积注意力
def attention(query,key,value,mask=None,dropout=None):
    d_k=query.size(-1)
    # torch.matmul只对后两位进行点乘
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)
    p_attn=F.softmax(scores,dim=-1)

    if dropout is not None:
        p_attn=dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn

# 多头注意力
class MultiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        super().__init__()
        assert d_model % h ==0
        self.d_k=d_model //h
        self.h=h
        # self.linears nn.ModuleList列表，self.linears[i] nn.linear实例
        # q,k,y,wo
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(dropout)

    def forward(self,query,key,value,mask=None):
        #注意力分数为[batch, num_heads, seq_len_q, seq_len_k]
        #部分掩码缺少头数维度，对所有类型代码进行维度适配
        if mask is not None:
            mask=mask.unsqueeze(1)
        nbatches=query.size(0)
        # [batch,seq_len,d_module],[batch,seq_len,h,d_k],[batch,h,seq_len,d_k]
        query,key,value=[lin(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2)
                         for lin,x in zip(self.linears,(query,key,value))]
        x,self.attn=attention(query,key,value,mask=mask,dropout=self.dropout)
        #拼接
        #contiguous()维度转置（transpose/permute）、切片（narrow）、广播扩展（部分）导致非连续，view可能报错
        x=(x.transpose(1,2).contiguous().view(nbatches,-1,self.h*self.d_k))
        # 清内存
        del query,key,value

        return self.linears[-1](x)

#前馈网络
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        #nn.Linear本质可学习
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)

        self.dropout=nn.Dropout(dropout)
    # dropout放在激活后，特征最丰富，破坏线性层（w1前），减少冗余特征对降维层的干扰
    def forward(self,x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

#解码器x1
class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super().__init__()

        self.self_attn=self_attn
        self.feed_forward=feed_forward

        self.size=size

        self.sublayer=clones(SublayerConnection(size, dropout), 2)

    def forward(self,x,mask):
        # 自注意力qky一样
        # self.mask?
        x=self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)
#解码器x6
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)

        return self.norm(x)

#解码器X1
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward

        self.size=size

        self.sublayer=clones(SublayerConnection(size, dropout), 3)
    def forward(self,x,memory,src_mask, tgt_mask):
        m=memory
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)
#解码器x6
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x=layer(x,memory,src_mask, tgt_mask)

        return self.norm(x)

#词嵌入
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        super().__init__()
        # 创建一个可训练的的词嵌入查找表
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self,x):
        return self.lut(x)*math.sqrt(self.d_model)

#位置嵌入
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        #创建全零张量
        pe=torch.zeros(max_len,d_model)
        #序列位置索引，改为二维列向量
        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))

        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        # 增加批次维度，与词嵌入统一
        pe=pe.unsqueeze(0)
        #设为固定的不可学习的，可随模型保存加载的参数
        self.register_buffer("pe",pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1)]

        return self.dropout(x)

#linear&softmax
class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super().__init__()
        self.proj=nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)

#使用
def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,h=8,dropout=0.1):
    #深拷贝，使各层不共享
    c = copy.deepcopy
    attn =MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    # nn.Sequential(...)顺序执行容器
    model=EncoderDecoder(Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
                         Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout),N),
                         nn.Sequential(Embeddings(d_model, src_vocab),c(position)),
                         nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                         Generator(d_model, tgt_vocab))
    #初始化参数
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return model
