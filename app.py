import shutil

from flask import Flask, jsonify, render_template, request, send_from_directory, session, redirect, url_for
from flask_session import Session
import os
import pandas as pd
import numpy as np
import model_pre
import uuid

app = Flask(__name__)
# 设置JSON编码
app.config['JSON_AS_ASCII'] = False
# 设置上传文件夹路径
app.config['UPLOAD_FOLDER'] = 'uploads'
# 设置会话密钥
app.secret_key = os.urandom(24)
# 配置Flask-Session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'flask_session')
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 会话有效期1分钟
# 初始化Flask-Session
Session(app)
# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)



# 首页路由
@app.route('/')
def index():
    # 检查是否有用户ID，如果没有则创建新用户
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
        os.makedirs(session_folder, exist_ok=True)
        return redirect(url_for('index', user_id=session['user_id']))

    return render_template('index.html')

@app.route('/clearCaching')
def clearCaching():
    user_id = session['user_id']
    file_dir=os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    shutil.rmtree(file_dir)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        data = request.get_json()
        prediction_type = data.get('predictionType')
        stlFileName=data.get('stlFileName')
        excelFileName=data.get('excelFileName')

        # 使用请求中的用户ID
        user_id=session['user_id']
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # 获取用户特定的文件路径
        stl_path = os.path.join(session_folder, stlFileName)
        excel_path = os.path.join(session_folder, excelFileName)


        try:
            if prediction_type == '危险点':
                if not os.path.exists(stl_path):
                    return jsonify({'status': 'error', 'message': '请先上传STL文件'}), 400
                if not os.path.exists(excel_path):
                    return jsonify({'status': 'error', 'message': '请先上传表格文件'}), 400

                points_file = os.path.join(session_folder, f'{stlFileName}_{excelFileName}_danger.csv')
                if os.path.exists(points_file):
                    points = np.loadtxt(points_file, delimiter=',', skiprows=1)
                else:
                    # 为每个用户创建独立的模型实例
                    model = model_pre.model(stl_path, excel_path)
                    points = model.predict_danger()

                    # 将危险点数据保存到用户特定的文件

                    np.savetxt(points_file, points, delimiter=',', header='x,y,z', comments='')

                formatted_points = []
                for point in points:
                    if isinstance(point, np.ndarray):
                        formatted_points.append(point.tolist())
                    else:
                        formatted_points.append(point)

                return jsonify({
                    'status': 'success',
                    'predictionType': prediction_type,
                    'message': f'危险点预测完成，共找到{len(points)}个点',
                    'points': formatted_points,
                })
            elif prediction_type == '加速度':
                if not os.path.exists(excel_path):
                    return jsonify({'status': 'error', 'message': '请先上传表格文件'}), 400
                if not os.path.exists(stl_path):
                    return jsonify({'status': 'error', 'message': '请先上传stl文件'}), 400

                accelerate_file = os.path.join(session_folder, f'{stlFileName}_{excelFileName}_acceleration.csv')
                if os.path.exists(accelerate_file):
                    saved_accelerate = np.loadtxt(accelerate_file, delimiter=',', skiprows=1)
                else:
                    # 为每个用户创建独立的模型实例
                    model = model_pre.model(stl_path, excel_path)
                    accelerate = model.predict_acceleration()
                    accelerate_np = accelerate.detach().numpy()
                    np.savetxt(accelerate_file, accelerate_np, delimiter=',', header='acceleration', comments='')
                    saved_accelerate = np.loadtxt(accelerate_file, delimiter=',', skiprows=1)


                accelerate_list = saved_accelerate.tolist()

                return jsonify({
                    'status': 'success',
                    'predictionType': prediction_type,
                    'message': '加速度预测完成',
                    'accelerate': [accelerate_list],
                    'user_id': user_id
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

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    # 首先检查是否有文件被上传
    if 'excel_file' not in request.files:
        return jsonify({'error': '没有文件被上传'}), 400
        
    # 获取文件
    file = request.files['excel_file']
    if 'user_id' not in session:
        return jsonify({
            'status': 'redirect',
            'message': '消息已经失效',
            'url': '/'
        })
    session['excel_filename'] = request.form['filename']
    session['excel_filesize'] = request.form['filesize']

    user_id = session['user_id']
    print(session)
    try:
        # 确保用户文件夹存在
        session_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # 保存文件
        filename = os.path.join(session_folder, file.filename)  # 使用固定文件名
        file.save(filename)

        # 读取Excel文件
        df = pd.read_excel(file, header=0)
        df = df.astype(str)

        headers = df.columns.tolist()
        data_rows = df.values.tolist()

        data = {
            'columns': headers,
            'data': data_rows,
            'total_rows': len(df),
            'total_columns': len(df.columns)
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'文件处理错误: {str(e)}'}), 500

@app.route('/upload_stl', methods=['POST'])
def upload_stl():
    if 'user_id' not in session:
        return jsonify({
            'status': 'redirect',
            'message':'消息已经失效',
            'url':'/'
        })
    user_id=session['user_id']
    if request.headers.get('Content-Type') == 'application/json':
        data=request.get_json()
        file_name = data.get('filename')
        file_size = data.get('filesize')
        stl_path=os.path.join(app.config['UPLOAD_FOLDER'],user_id, file_name)

        if os.path.exists(stl_path) :
            stl_size = os.path.getsize(stl_path)
            if file_size == stl_size:
                return jsonify({'status':'success'}),200
        return jsonify({'status':'error'}),200

    else:
        # 首先检查是否有文件被上传
        if 'stl_file' not in request.files:
            return jsonify({'error': '没有文件被上传'}), 400

        # 获取文件
        file = request.files['stl_file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 获取user_id，如果不存在则使用session中的user_id
        user_id = session['user_id']
        try:
            # 确保用户文件夹存在
            session_folder = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
            os.makedirs(session_folder, exist_ok=True)

            # 保存文件
            filename = os.path.join(session_folder, file.filename)
            session['filename'] = filename
            file.save(filename)

            return jsonify({
                'message': '文件上传成功',
                'filename': file.filename
            })
        except Exception as e:
            return jsonify({'error': f'文件上传错误: {str(e)}'}), 500




@app.route('/uploads/<path:StlName>/<path:ExcelName>/<type>')
def uploaded_file(StlName, ExcelName,type):
    if type=='danger':
        user_id = session['user_id']
        filename=f'{StlName}_{ExcelName}_danger.csv'
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], user_id, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found"}), 404

        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], user_id), filename)
    elif type=='acceleration':
        user_id = session['user_id']
        filename = f'{StlName}_{ExcelName}_acceleration.csv'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_id, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found"}), 404

        return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], user_id), filename)

@app.route('/templates/<path:filename>')
def templates_file(filename):
    return send_from_directory('templates', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)