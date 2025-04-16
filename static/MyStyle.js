let scene, camera, renderer, controls;
let isStlUploading = false; // 添加STL上传状态标志

// 初始化3D场景
function initScene() {
    scene = new THREE.Scene();
    
    // 设置场景背景为透明
    scene.background = null;
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    // 设置渲染器为透明背景
    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true  // 启用透明背景
    });
    renderer.setClearColor(0x000000, 0); // 设置透明背景

    const container = document.getElementById('model-viewer');
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // 添加轨道控制器
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.5;

    // 添加环境光
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // 添加平行光
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    camera.position.z = 5;

    // 开始动画循环
    animate();
}

// 动画循环
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// 处理窗口大小变化
window.addEventListener('resize', onWindowResize, false);

function onWindowResize() {
    const container = document.getElementById('model-viewer');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// 初始化场景
initScene();

// 处理文件选择
document.getElementById('stlFileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];

    if (!file) return;

    // 更新文件路径显示
    document.getElementById('stlFilePath').value = file.name;

    // 显示进度条
    const progressBar = document.querySelector('.progress-bar');
    const progressContainer = document.querySelector('.upload-progress');
    const uploadStatus = document.querySelector('.upload-status');
    progressContainer.style.display = 'block';
    uploadStatus.style.display = 'block';
    progressBar.style.width = '0%';

    // 设置上传状态
    isStlUploading = true;
    const predictButton = document.querySelector('.primary-button');
    predictButton.disabled = true;
    predictButton.textContent = '文件上传中...';

    // 禁用STL文件选择按钮
    const stlFileInput = document.getElementById('stlFileInput');
    const stlFileButton = document.querySelector('.file-input-group button');
    stlFileInput.disabled = true;
    stlFileButton.disabled = true;
    stlFileButton.style.opacity = '0.5';
    stlFileButton.style.cursor = 'not-allowed';
    stlFilePath.disabled = true;

    // 创建FormData对象用于后端上传
    const formData = new FormData();
    formData.append('stl_file', file);

    // 创建XMLHttpRequest对象来获取上传进度
    const xhr = new XMLHttpRequest();

    // 监听上传进度
    xhr.upload.addEventListener('progress', function(e) {
        if (e.lengthComputable) {
            const percentComplete = (e.loaded / e.total) * 100;
            progressBar.style.width = percentComplete + '%';
            console.log('上传进度: ' + percentComplete + '%');
        }
    });

    // 监听上传完成
    xhr.addEventListener('load', function() {
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            if (data.error) {
                alert(data.error);
                return;
            }
            // 上传完成后隐藏进度条
            setTimeout(() => {
                progressContainer.style.display = 'none';
                uploadStatus.style.display = 'none';
            }, 500);

            // 重置上传状态和按钮
            isStlUploading = false;
            predictButton.disabled = false;
            predictButton.textContent = '开始预测';

            // 重新启用STL文件选择按钮
            stlFileInput.disabled = false;
            stlFileButton.disabled = false;
            stlFileButton.style.opacity = '1';
            stlFileButton.style.cursor = 'pointer';
            stlFilePath.disabled = false;
        } else {
            alert('上传文件时发生错误');
            progressContainer.style.display = 'none';
            uploadStatus.style.display = 'none';
            // 重置上传状态和按钮
            isStlUploading = false;
            predictButton.disabled = false;
            predictButton.textContent = '开始预测';

            // 重新启用STL文件选择按钮
            stlFileInput.disabled = false;
            stlFileButton.disabled = false;
            stlFileButton.style.opacity = '1';
            stlFileButton.style.cursor = 'pointer';
            stlFilePath.disabled = false;
        }
    });

    // 发送请求到后端
    xhr.open('POST', '/upload_stl', true);
    xhr.send(formData);

    // 同时在前端直接加载和显示STL文件
    const reader = new FileReader();
    reader.onload = function(e) {
        const loader = new THREE.STLLoader();
        const geometry = loader.parse(e.target.result);

        // 清除现有的模型和危险点
        while(scene.children.length > 0){
            scene.remove(scene.children[0]);
        }

        // 添加新的模型
        const material = new THREE.MeshPhongMaterial({
            color: 0x00ff00,
            specular: 0x111111,
            shininess: 200
        });
        const mesh = new THREE.Mesh(geometry, material);

        // 直接显示模型
        scene.add(mesh);

        // 计算模型的边界框
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);

        // 设置相机位置
        camera.position.set(maxDim, maxDim, maxDim * 2);
        camera.lookAt(new THREE.Vector3(0, 0, 0));

        // 添加光源
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(ambientLight);
        scene.add(directionalLight);

        // 显示模型查看器
        document.getElementById('model-viewer').style.display = 'block';
        document.querySelector('#model-viewer #loadingOverlay').style.display = 'none';

        // 手动触发调整大小以确保正确的尺寸
        onWindowResize();
    };

    reader.readAsArrayBuffer(file);
});

// 添加Excel处理相关的JavaScript代码
let excelData = null;
let currentPage = 1;
let pageSize = 10;
let filteredData = [];

// 处理Excel文件上传
function selectInputFile() {
    const input=document.createElement('input')
    input.type = 'file';
    input.accept = '.xlsx,.xls';
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Display the selected file name
        document.getElementById('excelFilePath').value = file.name;

        const formData = new FormData();
        formData.append('excel_file', file);

        try {
            const response = await fetch('/upload_excel', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
                return;
            }

            excelData = data;
            excelData = data; // Store raw data

            // Populate column search dropdown
            const columnSelect = document.getElementById('columnSearchSelect');
            // Clear previous options except the default
            columnSelect.options.length = 1;
            excelData.columns.forEach((col, index) => {
                const option = document.createElement('option');
                option.value = index; // Use index as value
                option.textContent = col;
                columnSelect.appendChild(option);
            });

            // Initial display setup
            filteredData = [...data.data];
            displayExcelData();
            document.getElementById('excelContainer').style.display = 'block';
        } catch (error) {
            alert('文件处理出错：' + error.message);
        }
    };
    input.click();
}

// 显示Excel数据
function displayExcelData() {
    if (!excelData) return;

    // 设置表头（直接使用后端返回的columns）
    const headerRow = document.getElementById('tableHeader');
    headerRow.innerHTML = excelData.columns.map(col => `<th>${col}</th>`).join('');

    // 显示数据
    filteredData = [...excelData.data]; // 更新过滤数据
    updateTableDisplay();
}

// 更新表格显示
function updateTableDisplay() {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const pageData = filteredData.slice(startIndex, endIndex);

    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = pageData.map((row, rowIndex) => {
        return `<tr>${row.map((cell, cellIndex) => {
            return `<td>${cell || ''}</td>`;
        }).join('')}</tr>`;
    }).join('');

    // 更新分页信息
    const totalPages = Math.ceil(filteredData.length / pageSize);
    document.getElementById('pageInfo').textContent = `第 ${currentPage}/${totalPages} 页`;
    document.getElementById('totalInfo').textContent = `共 ${filteredData.length} 条记录`;

    // 更新按钮状态
    document.getElementById('prevPage').disabled = currentPage === 1;
    document.getElementById('nextPage').disabled = currentPage === totalPages;
}

// Combined search and filter function
function filterTableData() {
    const searchTerm = document.getElementById('tableSearch').value.toLowerCase();
    const selectedColumnIndex = parseInt(document.getElementById('columnSearchSelect').value);

    if (!excelData || !excelData.data) {
        filteredData = [];
    } else {
        filteredData = excelData.data.filter(row => {
            if (selectedColumnIndex === -1) {
                // Search all columns
                return row.some(cell =>
                    (cell !== undefined && cell !== null && cell.toString().toLowerCase().includes(searchTerm))
                );
            } else {
                // Search specific column
                const cellValue = row[selectedColumnIndex];
                return cellValue !== undefined && cellValue !== null &&
                       cellValue.toString().toLowerCase().includes(searchTerm);
            }
        });
    }

    currentPage = 1; // Reset to first page after filtering
    updateTableDisplay();
}

// Add event listeners for filtering
document.getElementById('tableSearch').addEventListener('input', filterTableData);
document.getElementById('columnSearchSelect').addEventListener('change', filterTableData);

// 切换页面大小
document.getElementById('pageSize').addEventListener('change', (e) => {
    pageSize = parseInt(e.target.value);
    currentPage = 1;
    updateTableDisplay();
});

// 翻页功能
document.getElementById('prevPage').addEventListener('click', () => {
    if (currentPage > 1) {
        currentPage--;
        updateTableDisplay();
    }
});

document.getElementById('nextPage').addEventListener('click', () => {
    const totalPages = Math.ceil(filteredData.length / pageSize);
    if (currentPage < totalPages) {
        currentPage++;
        updateTableDisplay();
    }
});

function startPrediction() {
    // 检查是否正在上传文件
    if (isStlUploading) {
        alert('请等待STL文件上传完成');
        return;
    }

    // 检查是否已上传STL文件
    const stlFilePath = document.getElementById('stlFilePath').value;
    if (!stlFilePath) {
        alert('请先上传STL文件');
        return;
    }

    // 检查是否已上传Excel文件
    const excelFilePath = document.getElementById('excelFilePath').value;
    if (!excelFilePath) {
        alert('请先上传Excel文件');
        return;
    }

    // 显示加载遮罩
    const loadingOverlay = document.querySelector('#model-viewer #loadingOverlay');
    const loadingText = document.querySelector('#loadingText');
    loadingOverlay.style.display = 'flex';

    // 获取选中的预测类型
    const predictionType = document.getElementById('cnas').checked ? '危险点' : '加速度';

    // 如果是加速度预测，更新加载文本
    if (predictionType === '加速度') {
        loadingText.textContent = '加速度预测中，这可能需要几分钟时间...';
    }

    // 设置较长的超时时间（5分钟）
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000);

    // 发送预测请求到后端
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            predictionType: predictionType,
            stlFile: stlFilePath,
            excelFile: excelFilePath
        }),
        signal: controller.signal
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        clearTimeout(timeoutId);
        // 隐藏加载遮罩
        loadingOverlay.style.display = 'none';

        if (data.status === 'success') {
            if (predictionType === '危险点') {
                // 处理危险点预测结果
                const points = data.points;
                console.log("接收到的危险点数据:", points);

                // 在3D视图中显示危险点
                displayDangerPoints(points);
                // 切换到危险点标签页
                switchTab('danger');
                alert(data.message);
            } else {
                // 处理加速度预测结果
                const accelerationData = data.accelerate;
                console.log("接收到的加速度数据:", accelerationData);

                // 切换到加速度标签页
                switchTab('acceleration');
                // 绘制加速度图表
                drawAccelerationChart(accelerationData);
                // 显示图表容器
                document.getElementById('chartContainer').style.display = 'block';
                alert(data.message);
            }
        } else {
            alert('预测失败：' + data.message);
        }
    })
    .catch(error => {
        clearTimeout(timeoutId);
        // 隐藏加载遮罩
        loadingOverlay.style.display = 'none';

        if (error.name === 'AbortError') {
            alert('预测超时，请重试');
        } else {
            console.error('Error:', error);
            alert('预测过程中发生错误: ' + error.message);
        }
    });
}

// 添加显示危险点的函数
function displayDangerPoints(points) {
    // 清除现有的点
    const existingPoints = scene.children.filter(child =>
        child instanceof THREE.Points ||
        (child instanceof THREE.Mesh && child.geometry instanceof THREE.SphereGeometry)
    );
    existingPoints.forEach(point => scene.remove(point));

    // 创建点的几何体
    const geometry = new THREE.BufferGeometry();

    // 将点数据转换为Float32Array
    const positions = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
        positions[i * 3] = points[i][0];     // x
        positions[i * 3 + 1] = points[i][1]; // y
        positions[i * 3 + 2] = points[i][2]; // z
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // 创建点的材质
    const material = new THREE.PointsMaterial({
        color: 0xff0000,  // 红色
        size: 8.0,        // 增大点的大小
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending  // 添加混合效果使点更亮
    });

    // 创建点云对象
    const pointCloud = new THREE.Points(geometry, material);

    // 为每个点添加一个小球体，使其更容易看见
    points.forEach(point => {
        const sphereGeometry = new THREE.SphereGeometry(2.0, 16, 16); // 增大球体半径
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0xff0000,
            transparent: true,
            opacity: 0.7,
            emissive: 0xff0000,
            emissiveIntensity: 0.7
        });
        const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere.position.set(point[0], point[1], point[2]);
        scene.add(sphere);
    });

    scene.add(pointCloud);

    // 更新渲染
    renderer.render(scene, camera);
}

async function exportTable() {
    const filename = 'danger_points.csv';

    try {
        // 检查文件是否存在，使用 HEAD 请求（更轻量）
        const checkResponse = await fetch(`/uploads/${filename}`, { method: 'HEAD' });
        if (!checkResponse.ok) {
            throw new Error('文件不存在');
        }

        // 如果文件存在，触发下载
        const link = document.createElement('a');
        link.href = `/uploads/${filename}`;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        console.error('错误:', error);
        alert(`无法下载文件: ${error.message}`);
    }
}

async function exportAccelerate() {
    const filename = 'acceleration_results.csv';
    try{
        const response=await fetch(`uploads/${filename}`,{method:'HEAD'});
        if (!response.ok){
            throw new Error('文件不存在')
        }
        const link=document.createElement("a")
        link.href=`uploads/${filename}`
        link.download=filename
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }catch (error){
        alert(`无法下载文件：${error.message}`)
    }

}

function clearWindow() {
    window.location.href='/'

}

async function clearCaching() {
    try{
        const response=await fetch('/clearCaching',{method:'GET'})
        if (response.ok){
            const data=await response.json()
            alert(`${data.success}`)

        }
    }catch (error){
        alert(`无法删除文件：${error.message}`)
    }

}





function clearReset() {
    alert('清除重置功能将在后续实现');
}

function exportDangerPoints() {
    alert('导出危险点坐标功能将在后续实现');
}

// 添加标签页切换效果
function switchTab(tab) {
    const tabs = document.querySelectorAll('.tab-button');
    tabs.forEach(t => t.classList.remove('active'));

    // 找到对应的标签按钮并激活
    const activeTab = Array.from(tabs).find(t => t.textContent.includes(tab === 'danger' ? '危险点' : '加速度'));
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // 显示对应的内容
    document.getElementById('dangerTab').style.display = tab === 'danger' ? 'block' : 'none';
    document.getElementById('accelerationTab').style.display = tab === 'acceleration' ? 'block' : 'none';

    // 如果是加速度标签，确保图表容器可见
    if (tab === 'acceleration') {
        const chartContainer = document.getElementById('chartContainer');
        if (chartContainer) {
            chartContainer.style.display = 'block';
        }
    }
}

window.addEventListener('beforeunload',function (event){
    clearCaching()
})


// 修改绘制图表的函数
function drawAccelerationChart(accelerationData) {
    // 检查数据是否存在
    if (!accelerationData || !Array.isArray(accelerationData)) {
        console.error('无效的加速度数据');
        return;
    }

    // 确保我们使用数组中的第一个元素，因为后端返回的是嵌套数组
    const data = Array.isArray(accelerationData[0]) ? accelerationData[0] : accelerationData;

    // 检查数据是否为空
    if (!data || data.length === 0) {
        console.error('加速度数据为空');
        return;
    }

    const ctx = document.getElementById('accelerationChart').getContext('2d');
    const labels = Array.from({length: data.length}, (_, i) => i + 1);

    // 创建完整的数据序列
    const allPoints = data.map((value, index) => {
        const numValue = Number(value);
        if (isNaN(numValue)) {
            console.error(`无效的加速度值: ${value}`);
            return null;
        }
        return numValue;
    });

    // 根据阈值将数据点分类
    const normalPoints = allPoints.map(value => value <= 230 ? value : null);
    const dangerPoints = allPoints.map(value => value > 230 && value <= 250 ? value : null);
    const failurePoints = allPoints.map(value => value > 250 ? value : null);

    // 如果已经存在图表，先销毁它
    const existingChart = Chart.getChart(ctx);
    if (existingChart) {
        existingChart.destroy();
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: '正常点 (≤230)',
                    data: normalPoints,
                    borderColor: 'rgb(75, 192, 75)',
                    backgroundColor: 'rgb(75, 192, 75)',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.1,
                    spanGaps: true,
                    order: 3  // 最上层显示
                },
                {
                    label: '危险点 (230-250)',
                    data: dangerPoints,
                    borderColor: 'rgb(220, 53, 69)',
                    backgroundColor: 'rgb(220, 53, 69)',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.1,
                    spanGaps: true,
                    order: 2
                },
                {
                    label: '失败点 (>250)',
                    data: failurePoints,
                    borderColor: 'rgb(173, 181, 189)',
                    backgroundColor: 'rgb(173, 181, 189)',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.1,
                    spanGaps: true,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: '加速度预测结果',
                    color: 'white',
                    font: {
                        size: 16
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        color: 'white'
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            if (value === null) return null;
                            return `加速度: ${value.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '加速度索引',
                        color: 'white',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'white'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '加速度值',
                        color: 'white',
                        font: {
                            size: 14
                        }
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'white'
                    }
                }
            }
        }
    });

    // 显示图表容器
    document.getElementById('chartContainer').style.display = 'block';

}