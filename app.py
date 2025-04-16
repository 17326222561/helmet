from flask import Flask, jsonify, render_template, request, send_from_directory
import os
import pandas as pd
import json
import numpy as np
import torch
import model_pre
import trimesh




excel_path=None
stl_path=None

app = Flask(__name__)
# 设置JSON编码
app.config['JSON_AS_ASCII'] = False
# 设置上传文件夹路径
app.config['UPLOAD_FOLDER'] = 'uploads'
# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# 处理 NaN 值的函数
def handle_nan_values(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/clearCaching')
def clearCaching():
    files = os.listdir('uploads')
    try:
        for file in files:
            os.remove(os.path.join('uploads', file))
        return jsonify({'success': f'共删除了{len(files)}个文件'}),200
    except Exception as e:
        return jsonify({'error':f'文件删除错误{e}'}), 400







@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        prediction_type = data.get('predictionType')

        try:
            if prediction_type == '危险点':
                if not stl_path:
                    return jsonify({'status': 'error', 'message': '请先上传STL文件'}), 400
                if not excel_path:
                    return jsonify({'status': 'error', 'message': '请先上传表格文件'}), 400

                # 使用transformer_gui中的parse_binary_stl函数处理STL文件
                model = model_pre.model(stl_path, excel_path)
                points = model.predict_danger()

                # 将危险点数据保存到文件
                points_file = os.path.join(app.config['UPLOAD_FOLDER'], 'danger_points.csv')
                np.savetxt(points_file, points, delimiter=',', header='x,y,z', comments='')

                # 将numpy数组转换为普通列表
                formatted_points = []
                for point in points:
                    if isinstance(point, np.ndarray):
                        formatted_points.append(point.tolist())
                    else:
                        formatted_points.append(point)

                # 返回预测结果
                return jsonify({
                    'status': 'success',
                    'predictionType': prediction_type,
                    'message': f'危险点预测完成，共找到{len(points)}个点',
                    'points': formatted_points
                })
            elif prediction_type == '加速度':
                if not excel_path:
                    return jsonify({'status': 'error', 'message': '请先上传表格文件'}), 400
                if not stl_path:
                    return jsonify({'status': 'error', 'message': '请先上传stl文件'}), 400

                model = model_pre.model(stl_path, excel_path)
                accelerate = model.predict_acceleration()

                # 将加速度预测结果保存到文件
                accelerate_file = os.path.join(app.config['UPLOAD_FOLDER'], 'acceleration_results.csv')

                # 确保数据是numpy数组格式
                accelerate_np = accelerate.detach().numpy()

                # 保存到文件
                np.savetxt(accelerate_file, accelerate_np, delimiter=',', header='acceleration', comments='')

                # 读取保存的加速度结果
                saved_accelerate = np.loadtxt(accelerate_file, delimiter=',', skiprows=1)

                # 将numpy数组转换为列表格式
                accelerate_list = saved_accelerate.tolist()

                return jsonify({
                    'status': 'success',
                    'predictionType': prediction_type,
                    'message': '加速度预测完成',
                    'accelerate': [accelerate_list]  # 包装在列表中以匹配前端期望的格式
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': '无效的预测类型'
                }), 400

        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'预测过程中发生错误: {str(e)}'
            }), 500

    return render_template('index.html')

# Excel文件上传和处理路由
@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    if 'excel_file' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400
    
    file = request.files['excel_file']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    global excel_path
    excel_path = filename
    file.save(filename)

    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        return jsonify({'error': '请选择Excel文件'}), 400
    
    try:
        # 读取Excel文件，使用第8行作为表头
        df = pd.read_excel(file, header=0)
        
        # 将所有数据（包括表头）转换为字符串格式，避免类型问题
        df = df.astype(str)
        

        # 获取表头（第8行）
        headers = df.columns.tolist()
        
        # 获取数据行
        data_rows = df.values.tolist()
        
        # 将数据转换为JSON格式
        data = {
            'columns': headers,
            'data': data_rows,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'文件处理错误: {str(e)}'}), 500



# 文件上传路由
@app.route('/upload_stl', methods=['POST'])
def upload_stl():
    if 'stl_file' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400
    
    file = request.files['stl_file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not file.filename.lower().endswith('.stl'):
        return jsonify({'error': '请选择STL文件'}), 400
    # 安全地保存文件
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    global stl_path
    stl_path=filename
    file.save(filename)

    
    return jsonify({
        'message': '文件上传成功',
        'filename': file.filename
    })

# 获取上传的文件
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": f"File '{filename}' not found"}),404

    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# 获取templates目录下的静态文件
@app.route('/templates/<path:filename>')
def templates_file(filename):
    return send_from_directory('templates', filename)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)