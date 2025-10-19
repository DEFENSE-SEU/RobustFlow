def parse_best_results(file_path):
    """解析best.txt文件，提取每个数据集的最佳结果路径"""
    datasets = {}  # 存储每个数据集的结果
    current_dataset = None  # 当前处理的数据集
    
    with open(file_path, 'r', encoding='utf-8') as f:  # 打开best.txt文件
        for line in f:  # 逐行读取
            line = line.strip()  # 去除首尾空白字符
            
            if line.startswith('[INFO] Entering task directory:'):  # 识别数据集目录行
                # 提取数据集名称
                dataset_name = line.split('\\')[-1]  # 获取路径最后一部分作为数据集名
                current_dataset = dataset_name  # 设置当前数据集
                datasets[current_dataset] = {}  # 初始化该数据集的字典
                
            elif line.startswith('[BEST]'):  # 识别最佳结果行
                # 解析路径和轮次信息
                # 格式: [BEST] path/results.json/round X (score=Y)
                parts = line.split('/round ')  # 按"/round "分割
                if len(parts) >= 2:  # 确保格式正确
                    path_part = parts[0].replace('[BEST] ', '')  # 提取路径部分
                    round_part = parts[1].split(' ')[0]  # 提取轮次号
                    
                    # 构建graph.py路径
                    # 从 aflow_scripts\DROP\drop_heavy_noise\results.json/round 20
                    # 转换为 aflow_scripts\DROP\drop_heavy_noise\round_20/graph.py
                    graph_path = path_part.replace('\\results.json', f'\\round_{round_part}\\graph.py')
                    
                    # 从路径中提取噪声类型
                    path_segments = graph_path.split('\\')
                    noise_type = None
                    for segment in path_segments:
                        if 'original' in segment:
                            noise_type = 'original'
                            break
                        elif 'requirements' in segment:
                            noise_type = 'requirements'
                            break
                        elif 'paraphrasing' in segment:
                            noise_type = 'paraphrasing'
                            break
                        elif 'light_noise' in segment:
                            noise_type = 'light_noise'
                            break
                        elif 'moderate_noise' in segment:
                            noise_type = 'moderate_noise'
                            break
                        elif 'heavy_noise' in segment:
                            noise_type = 'heavy_noise'
                            break
                    
                    if noise_type and current_dataset:  # 如果成功识别噪声类型和数据集
                        datasets[current_dataset][noise_type] = graph_path  # 存储路径
    
    return datasets  # 返回解析结果

datasets = parse_best_results("aflow_scripts/best.txt")
print(datasets['DROP'])