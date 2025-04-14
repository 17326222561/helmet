# 这是一个示例 Python 脚本。
import os
import gc
import struct
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # 限制单线程

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置
import math
import copy
import torch.nn.functional as F
from sympy.physics.control.control_plots import matplotlib
import numpy as np
import torch
import sys
import trimesh
import pandas as pd
import re
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


sys.float_repr_style = 'fixed'
torch.set_printoptions(sci_mode=False)
checkpoint = torch.load('all_models_weights-2.pth')

target_point=100000

Nx=2
d_model = 100  # 107

def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def self_attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    if mask is not None:
        scores=scores.masked_fill(mask==1,-1e9)
    self_attn_softmax = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)
    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax

class module(nn.Module):
    def __init__(self,size,out):
        super(module,self).__init__()
        self.h1=nn.Linear(5,3)
        self.h2=nn.Linear(3,2)
        self.out=nn.Linear(2,1)
        self.test1 = nn.Linear(size, out)
        self.leaky_relu=nn.LeakyReLU()
        # self.leaky_relu = nn.ReLU()
    def forward(self,x):
        out=self.test1(x)
        out=self.leaky_relu(out)
        out = out.squeeze(1)
        return out


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x.to(device)
        x = self.conv1(x)
        x=self.relu(x)
        x = self.conv2(x)
        x=self.relu(x)
        x = self.conv3(x)
        x=self.relu(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self,  eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)  # 求标准差
        return (x - mean) / (std + self.eps)   # LayerNorm的计算公式

class SublayerConnection(nn.Module):
    def __init__(self, size=0, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, sublayer):
        t=x+self.layer_norm(sublayer(x))
        return t

class EncoderLayer(nn.Module):
    def __init__(self,self_attn,feed_forward,size=0):
        super(EncoderLayer,self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer = clones(SublayerConnection(), 2)
        self.size = size
    def forward(self,x,mask=0):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x))
        # print(x)
        return x
        # return self.sublayer[1](x,self.feed_forward)
class EncoderLayer2(nn.Module):
    def __init__(self,self_attn,feed_forward):
        super(EncoderLayer2, self).__init__()
        # self.self_attn=self_attn
        self.self_attn=clones(self_attn, 2)
        self.feed_forward=feed_forward
        self.sublayer = clones(SublayerConnection(), 2)
    def forward(self,x,memory,mask=None):

        x = self.sublayer[0](x,lambda x:self.self_attn[0](x,x,x))
        x = self.sublayer[1](x,lambda x:self.self_attn[1](x,memory,memory))
        return x
# 所以在这解码器层的代码中，在前向传播部分，掩码注意力层与交叉注意力层的参数是不是共享了
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers=clones(layer,N)
        # self.norm=LayerNorm()
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
class Encoder2(nn.Module):
    def __init__(self,layer,N):
        super(Encoder2,self).__init__()
        self.layers=clones(layer,N)
        # self.norm=LayerNorm()
    def forward(self,x,memory):
        for layer in self.layers:
            x = layer(x,memory)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self,x):
        t=self.w2(self.leaky_relu(self.w1(x)))
        return t

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        n_batch = query.size(0)  # size(0)->有几个矩阵 size(1)->单个矩阵行数 size（2）->单个矩阵列数
        # n_batch=1
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(0)
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        x, self.attn_softmax = self_attention(query, key, value, dropout=None, mask=mask)

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        t=self.linear_out(x)
        return t

class upmodle(nn.Module):
    def __init__(self,d_model):
        super(upmodle,self).__init__()
        self.h=nn.Linear(1,d_model)
    def forward(self, x):
        reshaped_tensor = x.unsqueeze(1)
        transposed_tensor = reshaped_tensor.permute(0, 2, 1)
        transposed_tensor=self.h(transposed_tensor)
        return transposed_tensor

class upmodle_point(nn.Module):
    def __init__(self,d_model):
        super(upmodle_point,self).__init__()
        self.h=nn.Linear(3,d_model)
    def forward(self, x):
        transposed_tensor=self.h(x)
        return transposed_tensor


class demodel(nn.Module):
    def __init__(self,d_model):
        super(demodel,self).__init__()
        self.h = nn.Linear(d_model, 1)
    def forward(self, x):
        reshaped_tensor = self.h(x)
        # print(reshaped_tensor)
        reshaped_tensor=reshaped_tensor.squeeze()
        # print(reshaped_tensor)
        return reshaped_tensor

class model(nn.Module):
    def __init__(self,stl_path,excel_path):
        super(model, self).__init__()
        self.data_points=None
        self.stl_path = stl_path
        self.excel_path = excel_path
    def predict_danger(self):
        intersection_points = []
        """ 加载并显示 STL 文件 + 计算交点 """
        if not self.stl_path:
            print("❌ STL 路径为空")
            return

        try:
            mesh = trimesh.load_mesh(self.stl_path)  # 加载 STL 为 Trimesh 对象
            normalized_directions_tensor = torch.load('normalized_directions_tensor.pt')
            centroid_test = np.array([0, 0, 0], dtype=np.float32)


            for direction in normalized_directions_tensor:
                ray_origin = centroid_test
                ray_direction = direction.numpy()

                # 计算交点
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[ray_direction * 1000]
                )

                if locations.shape[0] > 0:
                    intersection_points.append(locations[-1])  # 取最后的交点

            if intersection_points:
                print(f"✅ 计算到 {len(intersection_points)} 个交点")
                print(intersection_points)  # ✅ 打印交点坐标
            else:
                print("⚠️ 没有找到任何交点")
        except Exception as e:
            print(f"❌ Trimesh 计算交点失败: {e}")

        return np.array(intersection_points, dtype=np.float32)




    def browse_stl(self):
        """ 选择 STL 文件 """
        if self.stl_path:
            with open(self.stl_path, 'rb') as f:
                # 跳过 80 字节的头部

                f.read(80)

                # 读取三角形面数
                num_triangles = struct.unpack('<I', f.read(4))[0]

                target_points = target_point
                points = []
                total_vertices = num_triangles * 3  # 每个三角形 3 个顶点

                sample_prob = target_points / total_vertices
                for _ in range(num_triangles):
                    f.read(12)  # 跳过法向量
                    for _ in range(3):
                        x, y, z = struct.unpack('<3f', f.read(12))
                        # 按概率采样
                        if np.random.random() < sample_prob:
                            points.append([x, y, z])
                    f.read(2)  # 跳过属性字段


                # points = []
                # for _ in range(num_triangles):
                #     # 读取每个三角形的法向量（3 个浮点数）
                #     f.read(12)
                #
                #     # 读取每个三角形的 3 个顶点（每个顶点 3 个浮点数）
                #     for _ in range(3):
                #         x, y, z = struct.unpack('<3f', f.read(12))
                #         points.append([x, y, z])
                #
                #     # 跳过 2 字节的属性字段
                #     f.read(2)
            self.data_points = torch.tensor(points)

    def predict_acceleration(self):
        if not self.stl_path and self.excel_path:
            print("❌ STL或表格路径为空")
            return
        self.browse_stl()
        file_path = self.excel_path

        df1 = pd.read_excel(file_path)  # 读取 Excel
        print("✅ 成功读取 Excel 文件！")
        xx1 = df1['头型尺寸']
        xx1 = xx1.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx1 = torch.tensor(xx1.values, dtype=torch.float32).view(-1, 1)
        xx2 = df1['头型重量(Kg)']
        xx2 = xx2.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx2 = torch.tensor(xx2.values, dtype=torch.float32).view(-1, 1)
        xx3 = df1['头盔重量(Kg)']
        xx3 = xx3.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx3 = torch.tensor(xx3.values, dtype=torch.float32).view(-1, 1)
        xx4 = df1['帽壳厚度（mm）']
        xx4 = xx4.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx4 = torch.tensor(xx4.values, dtype=torch.float32).view(-1, 1)
        xx5 = df1['内衬密度']
        xx5 = xx5.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx5 = torch.tensor(xx5.values, dtype=torch.float32).view(-1, 1)
        xx6 = df1['撞击位置坐标']
        xx6 = xx6.apply(
            lambda x: [float(num) for num in re.split(r'#', str(x)) if num]
        )
        xx6 = [num for sublist in xx6 for num in sublist]
        xx6 = torch.tensor(xx6, dtype=torch.float32).view(-1, 3)
        xx9 = df1['头型重心坐标（与碰撞点形成撞击方向矢量）']
        xx9 = xx9.apply(
            lambda x: [float(num) for num in re.split(r'#', str(x)) if num]
        )
        xx9 = [num for sublist in xx9 for num in sublist]
        xx9 = torch.tensor(xx9, dtype=torch.float32).view(-1, 3)
        xx10 = df1['实测撞击速度(m/s)']
        xx10 = xx10.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx10 = torch.tensor(xx10.values, dtype=torch.float32).view(-1, 1)
        xx11 = df1['冲击能量(J)']
        xx11 = xx11.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx11 = torch.tensor(xx11.values, dtype=torch.float32).view(-1, 1)
        x1 = torch.cat((xx1, xx2, xx3, xx4, xx5, xx6, xx9, xx10, xx11), dim=1)
        batch_size = x1.shape[0]
        x1_points = self.data_points.unsqueeze(0).expand(batch_size, -1, -1)
        x1_points = torch.unique(x1_points, dim=1, return_inverse=False, return_counts=False)
        print(x1.shape)

        c = copy.deepcopy
        x_up_model = upmodle(d_model).to(device)
        x_point_up_model = upmodle_point(d_model).to(device)
        x_down_model = demodel(d_model).to(device)




        atten = MultiHeadAttention(head=100, d_model=d_model).to(device)

        ff = PositionwiseFeedForward(d_model, d_model).to(device)
        enlayer = EncoderLayer(c(atten), c(ff)).to(device)
        delayer = EncoderLayer2(c(atten), c(ff)).to(device)
        encoders = Encoder(enlayer, Nx).to(device)
        encoders2 = Encoder2(delayer, Nx).to(device)

        conv_net = ConvNet().to(device)
        net = module(x1.shape[1], 1).to(device)


        x_up_model.load_state_dict(checkpoint['x_up_model'])
        print('-------------------------------------------------------------')
        x_point_up_model.load_state_dict(checkpoint['x_point_up_model'])
        x_down_model.load_state_dict(checkpoint['x_down_model'])

        atten.load_state_dict(checkpoint['atten'])
        ff.load_state_dict(checkpoint['ff'])
        enlayer.load_state_dict(checkpoint['enlayer'])
        delayer.load_state_dict(checkpoint['delayer'])
        encoders.load_state_dict(checkpoint['encoders'])
        encoders2.load_state_dict(checkpoint['encoders2'])
        conv_net.load_state_dict(checkpoint['conv_net'])
        net.load_state_dict(checkpoint['net'])


        # 将所有模型设置为评估模式 (如果是用于推理)
        x_up_model.eval().to(device)
        x_point_up_model.eval().to(device)
        x_down_model.eval().to(device)
        atten.eval().to(device)
        ff.eval().to(device)
        enlayer.eval().to(device)
        delayer.eval().to(device)
        encoders.eval().to(device)
        encoders2.eval().to(device)
        conv_net.eval().to(device)
        net.eval().to(device)

        output_x_data_points = conv_net(x1_points.unsqueeze(1))

        output_x_data_points = output_x_data_points.squeeze(1)
        output_x_data_points = x_point_up_model(output_x_data_points)

        x_points_en = encoders(output_x_data_points)
        x_up = x_up_model(x1.to(device))
        x_en = encoders2(x_up, x_points_en)
        x_down = x_down_model(x_en)
        if(x_down.shape[0] == 1):
            x_down = x_down.view(1,-1)
        print(x_down.shape)
        out=net(x_down).cpu()
        torch.cuda.empty_cache()
        gc.collect()
        return out

