import torch
import torch.nn as nn
import pandas as pd
import re
import os
import struct
import gc


# 定义模型类
class upmodle(nn.Module):
    def __init__(self, d_model):
        super(upmodle, self).__init__()
        self.h = nn.Linear(1, d_model)

    def forward(self, x):
        reshaped_tensor = x.unsqueeze(1)
        transposed_tensor = reshaped_tensor.permute(0, 2, 1)
        return self.h(transposed_tensor)


class upmodle_point(nn.Module):
    def __init__(self, d_model):
        super(upmodle_point, self).__init__()
        self.h = nn.Linear(3, d_model)

    def forward(self, x):
        return self.h(x)


class demodel(nn.Module):
    def __init__(self, d_model):
        super(demodel, self).__init__()
        self.h = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.h(x).squeeze()


def parse_binary_stl(file_path):
    with open(file_path, 'rb') as f:
        f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        points = []
        for _ in range(num_triangles):
            f.read(12)
            for _ in range(3):
                x, y, z = struct.unpack('<3f', f.read(12))
                points.append([x, y, z])
            f.read(2)
        return points


def predict_accelerations(excel_path, stl_path, d_model=128, head=2, Nx=2):
    # 加载检查点
    checkpoint = torch.load('all_models_weights-MSE5scs.pth')

    # 创建并加载模型
    x_up_model = upmodle(d_model).eval()
    x_point_up_model = upmodle_point(d_model).eval()
    x_down_model = demodel(d_model).eval()
    # 假设其他模型定义，需根据实际补充
    # atten = MultiHeadAttention(head=head, d_model=d_model).eval()
    # ff = PositionwiseFeedForward(d_model, d_model).eval()
    # enlayer = EncoderLayer(atten, ff).eval()
    # delayer = EncoderLayer2(atten, ff).eval()
    # encoders = Encoder(enlayer, Nx).eval()
    # encoders2 = Encoder2(delayer, Nx).eval()
    # conv_net = ConvNet().eval()
    # net = module(x.shape[1], 1).eval()

    x_up_model.load_state_dict(checkpoint['x_up_model'])
    # 加载其他模型状态字典，需根据实际补充

    # 加载 STL 数据
    data_points = torch.tensor(parse_binary_stl(stl_path))

    # 读取并处理 Excel 文件
    df = pd.read_excel(excel_path)
    xx1 = df['头型尺寸'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx1 = torch.tensor(xx1.values, dtype=torch.float32).view(-1, 1)
    xx2 = df['头型重量(Kg)'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx2 = torch.tensor(xx2.values, dtype=torch.float32).view(-1, 1)
    xx3 = df['头盔重量(Kg)'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx3 = torch.tensor(xx3.values, dtype=torch.float32).view(-1, 1)
    xx4 = df['帽壳厚度（mm）'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx4 = torch.tensor(xx4.values, dtype=torch.float32).view(-1, 1)
    xx5 = df['内衬密度'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx5 = torch.tensor(xx5.values, dtype=torch.float32).view(-1, 1)
    xx6 = df['撞击位置坐标'].apply(lambda x: [float(num) for num in re.split(r'#', str(x)) if num])
    xx6 = [num for sublist in xx6 for num in sublist]
    xx6 = torch.tensor(xx6, dtype=torch.float32).view(-1, 3)
    xx9 = df['头型重心坐标（与碰撞点形成撞击方向矢量）'].apply(
        lambda x: [float(num) for num in re.split(r'#', str(x)) if num])
    xx9 = [num for sublist in xx9 for num in sublist]
    xx9 = torch.tensor(xx9, dtype=torch.float32).view(-1, 3)
    xx10 = df['实测撞击速度(m/s)'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx10 = torch.tensor(xx10.values, dtype=torch.float32).view(-1, 1)
    xx11 = df['冲击能量(J)'].apply(lambda x: float(re.sub(r'[^0-9.]', '', str(x))) if isinstance(x, str) else x)
    xx11 = torch.tensor(xx11.values, dtype=torch.float32).view(-1, 1)

    x = torch.cat((xx1, xx2, xx3, xx4, xx5, xx6, xx9, xx10, xx11), dim=1)

    predictions = []
    for i in range(x.shape[0]):
        x_i = x[i].unsqueeze(0)
        x1_points_i = data_points.unsqueeze(0)

        # 模拟原流程，需根据实际模型补充
        output_x_data_points = conv_net(x1_points_i.unsqueeze(1))  # 假设 conv_net 定义
        output_x_data_points = output_x_data_points.squeeze(1)
        output_x_data_points = x_point_up_model(output_x_data_points)
        x_points_en = encoders(output_x_data_points)  # 假设 encoders 定义
        x_up = x_up_model(x_i)
        x_en = encoders2(x_up, x_points_en)  # 假设 encoders2 定义
        x_down = x_down_model(x_en)
        if x_down.shape[0] == 1:
            x_down = x_down.view(1, -1)
        out = net(x_down)  # 假设 net 定义
        predictions.append(out.item())

        gc.collect()  # 清理内存

    return predictions