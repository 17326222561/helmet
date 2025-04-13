# 这是一个示例 Python 脚本。
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"  # 限制单线程
# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置
import math
import copy
import time
import collections
from copy import deepcopy
import torch.nn.functional as F
from sympy.physics.control.control_plots import matplotlib
from torch import optim
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import struct
import pandas as pd
import torch
import sys
from stl import mesh
import pyvista as pv
import re
import os
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 设置后端为TkAg
sys.float_repr_style = 'fixed'
torch.set_printoptions(sci_mode=False)
checkpoint = torch.load('all_models_weights-2.pth')


import trimesh
import sys
import torch
import pyvista as pv
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog, QLineEdit, QLabel, QDoubleSpinBox, QDialog
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
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
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(PositionwiseFeedForward,self).__init__()
        self.w1=nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.leaky_relu = nn.LeakyReLU()
    def forward(self,x):
        t=self.w2(self.leaky_relu(self.w1(x)))
        return t

def getPostitionEncoding(seq_len,dim,n=10000):
    PE = np.zeros(shape=(60, 4))
    for pos in range(seq_len):
        for i in range(int(dim/2)):
            den=np.power(n,2*i/dim)
            PE[pos,2*i]=np.sin(pos/den)
            PE[pos,2*i+1]=np.cos(pos/den)
    t=PE[1,:].reshape(-1,1)
    return PE
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):#max行数，d_model是列数
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1)
        wi=torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*wi)
        pe[:,1::2]=torch.cos(pos*wi)
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)
        print(pe)
    def forward(self,x):
        x=x+Variable(self.pe[:, :x.size(1)],requires_gard=False)
        return x
def clones(module,n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask
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
class upmodle(nn.Module):
    def __init__(self,d_model):
        super(upmodle,self).__init__()
        self.h=nn.Linear(1,d_model)
    def forward(self, x):
        reshaped_tensor = x.unsqueeze(1)
        transposed_tensor = reshaped_tensor.permute(0, 2, 1)
        transposed_tensor=self.h(transposed_tensor)
        # print(transposed_tensor)
        return transposed_tensor
class upmodle_point(nn.Module):
    def __init__(self,d_model):
        super(upmodle_point,self).__init__()
        self.h=nn.Linear(3,d_model)
    def forward(self, x):
        # reshaped_tensor = x.unsqueeze(1)
        # transposed_tensor = reshaped_tensor.permute(0, 2, 1)
        transposed_tensor=self.h(x)
        # print(transposed_tensor)
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
def parse_binary_stl(file_path):
    with open(file_path, 'rb') as f:
        # 跳过 80 字节的头部


        f.read(80)

        # 读取三角形面数
        num_triangles = struct.unpack('<I', f.read(4))[0]

        points = []
        for _ in range(num_triangles):
            # 读取每个三角形的法向量（3 个浮点数）
            f.read(12)

            # 读取每个三角形的 3 个顶点（每个顶点 3 个浮点数）
            for _ in range(3):
                x, y, z = struct.unpack('<3f', f.read(12))
                points.append([x, y, z])

            # 跳过 2 字节的属性字段
            f.read(2)

        return points
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(10,1), padding=(1,1), bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x=self.relu(x)
        x = self.conv2(x)
        x=self.relu(x)
        x = self.conv3(x)
        x=self.relu(x)
        return x

# --------------------------------------------------------------------------------------------------------
def load_and_process_data(excel_path,stl_path):
    df1 = pd.read_excel(excel_path)
    xx1 = df1['头型尺寸']
    xx1 = xx1.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx1 = torch.tensor(xx1.values, dtype=torch.float32).view(-1, 1)
    # print(xx1)
    # print(xx1)
    # print(xx1.shape)

    xx2 = df1['头型重量(Kg)']
    xx2 = xx2.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx2 = torch.tensor(xx2.values, dtype=torch.float32).view(-1, 1)
    # print(xx2)

    xx3 = df1['头盔重量(Kg)']
    xx3 = xx3.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx3 = torch.tensor(xx3.values, dtype=torch.float32).view(-1, 1)
    # print(xx3)

    xx4 = df1['帽壳厚度（mm）']
    xx4 = xx4.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx4 = torch.tensor(xx4.values, dtype=torch.float32).view(-1, 1)
    # print(xx4)

    xx5 = df1['内衬密度']
    xx5 = xx5.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx5 = torch.tensor(xx5.values, dtype=torch.float32).view(-1, 1)
    # print(xx5)
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
    # print(xx9)
    # print(xx9.shape)

    xx10 = df1['实测撞击速度(m/s)']
    xx10 = xx10.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx10 = torch.tensor(xx10.values, dtype=torch.float32).view(-1, 1)
    # print(xx10)

    xx11 = df1['冲击能量(J)']
    xx11 = xx11.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    xx11 = torch.tensor(xx11.values, dtype=torch.float32).view(-1, 1)
    # print(xx11)

    yy1 = df1['最大加速度(g)']
    yy1 = yy1.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
    y1 = torch.tensor(yy1.values, dtype=torch.float32).view(-1, 1)

    x1 = torch.cat((xx1, xx2, xx3, xx4, xx5, xx6,xx9, xx10, xx11), dim=1)
    points = parse_binary_stl(stl_path)
    data_points=torch.tensor(points)
    batch_size = x1.shape[0]
    x1_points = data_points.unsqueeze(0).expand(batch_size, -1, -1)

    return x1, y1,x1_points


import sys
import csv
import vtk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QFileDialog,
    QFrame, QPushButton
)
from PyQt5.QtCore import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stl_path = 0
        self.intersection_points_np = np.empty((0, 3), dtype=np.float32)
        self.setWindowTitle("STL 预测界面")
        self.setGeometry(100, 100, 900, 600)

        # 设置中心控件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建主布局（垂直布局）
        main_layout = QVBoxLayout(central_widget)

        # 第一行：输入 STL 模型
        row1_layout = QHBoxLayout()
        label_stl = QLabel("请输入STL模型")
        self.set_label_style(label_stl)

        self.line_edit_stl = QLineEdit()
        self.line_edit_stl.setPlaceholderText("请选择STL文件路径...")
        self.line_edit_stl.setReadOnly(True)
        self.line_edit_stl.mousePressEvent = self.browse_stl

        row1_layout.addWidget(label_stl)
        row1_layout.addWidget(self.line_edit_stl)
        row1_layout.setStretch(1, 1)  # 让输入框填满右侧

        # 第二行：输入普通文件
        row2_layout = QHBoxLayout()
        label_file = QLabel("请输入文件")
        self.set_label_style(label_file)

        self.line_edit_file = QLineEdit()
        self.line_edit_file.setPlaceholderText("请选择文件路径...")
        self.line_edit_file.setReadOnly(True)
        self.line_edit_file.mousePressEvent = self.browse_file

        row2_layout.addWidget(label_file)
        row2_layout.addWidget(self.line_edit_file)
        row2_layout.setStretch(1, 1)  # 让输入框填满右侧

        # 第三行：蓝色条状装饰条
        blue_bar = QFrame()
        blue_bar.setFixedHeight(20)
        blue_bar.setStyleSheet("background-color: lightblue; border-radius: 5px;")

        # 第四行：预测按钮和导出表格
        row4_layout = QHBoxLayout()

        # 左侧提示
        label_predict = QLabel("开始预测")
        self.set_label_style(label_predict)

        # 中间两个圆形按钮
        self.btn_danger = QPushButton("危险点预测")
        self.btn_danger.setFixedSize(100, 50)
        self.btn_danger.setStyleSheet("border-radius: 25px; background-color: red; color: white;")
        self.btn_danger.clicked.connect(self.predict_danger)

        self.btn_acceleration = QPushButton("加速度预测")
        self.btn_acceleration.setFixedSize(100, 50)
        self.btn_acceleration.setStyleSheet("border-radius: 25px; background-color: green; color: white;")
        self.btn_acceleration.clicked.connect(self.predict_acceleration)

        self.cleart = QPushButton("清楚重置")
        self.cleart.setFixedSize(100, 50)
        self.cleart.setStyleSheet("border-radius: 25px; background-color: blue; color: white;")
        self.cleart.clicked.connect(self.cleartt)

        # 右侧导出表格按钮
        self.btn_export = QPushButton("导出表格")
        self.btn_export.setFixedSize(100, 40)
        self.btn_export.setStyleSheet("background-color: orange; color: white; border-radius: 10px;")
        self.btn_export.clicked.connect(self.export_table)

        row4_layout.addWidget(label_predict)
        row4_layout.addWidget(self.btn_danger)
        row4_layout.addWidget(self.btn_acceleration)
        row4_layout.addWidget(self.cleart)
        row4_layout.addStretch()  # 让按钮居中
        row4_layout.addWidget(self.btn_export)

        # 第五行：渲染 STL 三维模型
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        # 添加各部分到主布局
        main_layout.addLayout(row1_layout)
        main_layout.addLayout(row2_layout)
        main_layout.addWidget(blue_bar)
        main_layout.addLayout(row4_layout)
        main_layout.addWidget(self.vtkWidget)

        # 启动 VTK 交互
        self.vtkWidget.GetRenderWindow().Render()
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

    def set_label_style(self, label):
        """ 设置 QLabel 的背景颜色、字体大小、对齐方式 """
        label.setStyleSheet("""
            background-color: lightblue;
            font-size: 14px;
            font-weight: bold;
            padding: 5px;
            border-radius: 5px;
        """)
        label.setFixedWidth(120)
        label.setFixedHeight(30)

    def browse_stl(self, event):
        """ 选择 STL 文件 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择STL模型", "", "STL Files (*.stl)")
        if file_path:
            self.line_edit_stl.setText(file_path)
            points = parse_binary_stl(self.line_edit_stl.text())
            self.data_points = torch.tensor(points)


    def browse_file(self, event):
        """ 选择普通文件 """
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*)")
        if file_path:
            self.line_edit_file.setText(file_path)

    def predict_danger(self):
        """ 加载并显示 STL 文件 + 计算交点 """
        self.stl_path = self.line_edit_stl.text()
        if not self.stl_path:
            print("❌ STL 路径为空")
            return

        try:
            # 加载 STL 文件
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.stl_path)
            reader.Update()

            if reader.GetOutput().GetNumberOfCells() == 0:
                print("❌ STL 加载失败！")
                return
            else:
                print("✅ STL 读取成功，面片数:", reader.GetOutput().GetNumberOfCells())

            # 创建 mapper 和 actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # STL 颜色 (灰色)

            # --- 计算交点 ---
            try:
                mesh = trimesh.load_mesh(self.stl_path)  # 加载 STL 为 Trimesh 对象
                normalized_directions_tensor = torch.load('normalized_directions_tensor.pt')
                centroid_test = np.array([0, 0, 0], dtype=np.float32)

                intersection_points = []
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
                    self.intersection_points_np = np.array(intersection_points, dtype=np.float32)
                    intersection_points_np = np.array(intersection_points, dtype=np.float32)
                    print(f"✅ 计算到 {len(intersection_points_np)} 个交点")
                    print(intersection_points_np)  # ✅ 打印交点坐标
                else:
                    print("⚠️ 没有找到任何交点")
                    intersection_points_np = np.array([])
            except Exception as e:
                print(f"❌ Trimesh 计算交点失败: {e}")
                intersection_points_np = np.array([])

            # --- 清空现有渲染器并添加新模型 ---
            self.renderer.RemoveAllViewProps()
            self.renderer.AddActor(actor)

            # 添加交点
            if intersection_points_np.size > 0:
                try:
                    points_vtk = vtk.vtkPoints()
                    for p in intersection_points_np:
                        points_vtk.InsertNextPoint(p[0], p[1], p[2])

                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(points_vtk)

                    glyph_source = vtk.vtkSphereSource()
                    glyph_source.SetRadius(2.0)  # 小球半径

                    glyph = vtk.vtkGlyph3D()
                    glyph.SetInputData(polydata)
                    glyph.SetSourceConnection(glyph_source.GetOutputPort())
                    glyph.SetScaleModeToDataScalingOff()

                    glyph_mapper = vtk.vtkPolyDataMapper()
                    glyph_mapper.SetInputConnection(glyph.GetOutputPort())

                    glyph_actor = vtk.vtkActor()
                    glyph_actor.SetMapper(glyph_mapper)
                    glyph_actor.GetProperty().SetColor(1, 0, 0)  # 红色点

                    self.renderer.AddActor(glyph_actor)

                except Exception as e:
                    print(f"❌ VTK 交点渲染失败: {e}")

            # 更新渲染
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()

        except Exception as e:
            print(f"❌ 发生错误: {e}")
    def cleartt(self):
        """ 加载并显示 STL 文件 + 计算交点 """
        self.stl_path = self.line_edit_stl.text()
        if not self.stl_path:
            print("❌ STL 路径为空")
            return

        try:
            # 加载 STL 文件
            reader = vtk.vtkSTLReader()
            reader.SetFileName(self.stl_path)
            reader.Update()

            if reader.GetOutput().GetNumberOfCells() == 0:
                print("❌ STL 加载失败！")
                return
            else:
                print("✅ STL 读取成功，面片数:", reader.GetOutput().GetNumberOfCells())

            # 创建 mapper 和 actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # STL 颜色 (灰色)
            # --- 清空现有渲染器并添加新模型 ---
            self.renderer.RemoveAllViewProps()
            self.renderer.AddActor(actor)
            # 更新渲染
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()

        except Exception as e:
            print(f"❌ 发生错误: {e}")
    def predict_acceleration(self):
        """ 点击加速度预测按钮，在终端打印数字 123，并读取 Excel 文件 """
        file_path = self.line_edit_file.text()  # 获取输入框内容并去除空格

        df1 = pd.read_excel(file_path)  # 读取 Excel
        print("✅ 成功读取 Excel 文件！")
        xx1 = df1['头型尺寸']
        xx1 = xx1.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx1 = torch.tensor(xx1.values, dtype=torch.float32).view(-1, 1)
        # print(xx1)
        # print(xx1)
        print(xx1.shape)

        xx2 = df1['头型重量(Kg)']
        xx2 = xx2.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx2 = torch.tensor(xx2.values, dtype=torch.float32).view(-1, 1)
        # print(xx2)

        xx3 = df1['头盔重量(Kg)']
        xx3 = xx3.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx3 = torch.tensor(xx3.values, dtype=torch.float32).view(-1, 1)
        # print(xx3)

        xx4 = df1['帽壳厚度（mm）']
        xx4 = xx4.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx4 = torch.tensor(xx4.values, dtype=torch.float32).view(-1, 1)
        # print(xx4)

        xx5 = df1['内衬密度']
        xx5 = xx5.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx5 = torch.tensor(xx5.values, dtype=torch.float32).view(-1, 1)
        # print(xx5)
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
        # print(xx10)

        xx11 = df1['冲击能量(J)']
        xx11 = xx11.apply(lambda x: float(re.sub(r'[^0-9.]', '', x)) if isinstance(x, str) else x)
        xx11 = torch.tensor(xx11.values, dtype=torch.float32).view(-1, 1)
        # print(xx11)

        x1 = torch.cat((xx1, xx2, xx3, xx4, xx5, xx6, xx9, xx10, xx11), dim=1)

        batch_size = x1.shape[0]
        x1_points = self.data_points.unsqueeze(0).expand(batch_size, -1, -1)
        x1_points = torch.unique(x1_points, dim=1, return_inverse=False, return_counts=False)
        print(x1.shape)

        c = copy.deepcopy
        x_up_model = upmodle(d_model)
        x_point_up_model = upmodle_point(d_model)
        x_down_model = demodel(d_model)




        atten = MultiHeadAttention(head=100, d_model=d_model)

        ff = PositionwiseFeedForward(d_model, d_model)
        enlayer = EncoderLayer(c(atten), c(ff))
        delayer = EncoderLayer2(c(atten), c(ff))
        encoders = Encoder(enlayer, Nx)
        encoders2 = Encoder2(delayer, Nx)

        conv_net = ConvNet()
        net = module(x1.shape[1], 1)

        print("checkpoint['x_up_model'] keys:", checkpoint['x_up_model'].keys())
        print("checkpoint['x_up_model'] shapes:", {k: v.shape for k, v in checkpoint['x_up_model'].items()})
        print(checkpoint)
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
        x_up_model.eval()
        x_point_up_model.eval()
        x_down_model.eval()
        atten.eval()
        ff.eval()
        enlayer.eval()
        delayer.eval()
        encoders.eval()
        encoders2.eval()
        conv_net.eval()
        net.eval()

        output_x_data_points = conv_net(x1_points.unsqueeze(1))

        output_x_data_points = output_x_data_points.squeeze(1)
        output_x_data_points = x_point_up_model(output_x_data_points)

        x_points_en = encoders(output_x_data_points)
        x_up = x_up_model(x1)
        x_en = encoders2(x_up, x_points_en)
        x_down = x_down_model(x_en)
        if(x_down.shape[0] == 1):
            x_down = x_down.view(1,-1)
        print(x_down.shape)
        out = net(x_down)
        self.out=out
        print("out", out)


    def export_table(self):
        print(self.out)
        print(self.out.shape)
        """ 导出交点数据到 CSV 文件 """
        if self.intersection_points_np.size != 0:
        # **设置默认保存路径**
            file_path = os.path.join(os.getcwd(), "危险点预测坐标.csv")  # 当前目录

        # 创建 Pandas DataFrame
            df = pd.DataFrame(self.intersection_points_np, columns=["X", "Y", "Z"])

        # 保存 CSV 文件
            df.to_csv(file_path, index=False)

            print(f"✅ 交点数据已保存到: {file_path}")
        if self.out.shape != torch.Size([0]):
        # **设置默认保存路径**
            file_path = os.path.join(os.getcwd(), "加速度预测.csv")  # 当前目录
            out_numpy = self.out.detach().cpu().numpy()
            out_numpy = out_numpy.reshape(-1, 1)
            df = pd.DataFrame(out_numpy, columns=["MAX(g)"])


        # 保存 CSV 文件
            df.to_csv(file_path, index=False)

            print(f"✅ 交点数据已保存到: {file_path}")

if __name__ == "__main__":
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())
    d_model=100#107
    head=2#5b
    Nx=2

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    print("验证集:")
    # x8_points = x8_points[:, :min_second_dim, :]
    output_x_data_points = conv_net(x4_points.unsqueeze(1))
    output_x_data_points = output_x_data_points.squeeze(1)

    output_x_data_points = x_point_up_model(output_x_data_points)

    x_points_en = encoders(output_x_data_points)
    x_up = x_up_model(x4)
    print(x_up.shape)
    x_en = encoders2(x_up, x_points_en)
    x_down = x_down_model(x_en)
    # x_down=x8+x_down
    # x_down=x_down+x_down
    out = net(x_down)
    i = 0
    errors = []  # 用于存储误差

    for row1, row2 in zip(out, y4.view(-1)):
        error = abs(row1 - row2)  # 计算误差，使用绝对值表示误差
        error_percentage = (error / row2) * 100  # 计算误差的百分数
        print(f" 模型预测第{i + 1}个样本加速度值: {row1},真实值为{row2},误差为百分之{error_percentage}%")
        errors.append(error_percentage)  # 将误差添加到列表

        i = i + 1

    max_error = max(errors)
    min_error = min(errors)
    avg_error = sum(errors) / len(errors)

    print(f"验证集8最大误差: {max_error}")
    print(f"验证集8最小误差: {min_error}")
    print(f"验证集8平均误差: {avg_error}")
    x_plot = torch.zeros((i+1), dtype=torch.float32)
    first_plot=torch.tensor([0.0], dtype=torch.float32)
    t=0

