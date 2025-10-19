import json  # 导入JSON模块，用于处理JSON格式数据的解析和序列化
import sys  # 导入系统模块，用于访问标准错误输出流
from pathlib import Path  # 导入Path类，用于跨平台的文件路径操作
import textwrap  # 导入文本包装模块，用于处理多行文本的格式化


def iter_jsonl(file_path: str):  # 定义函数，用于迭代读取JSONL文件，参数为文件路径字符串
    """
    逐行读取 JSONL 文件并解析为 Python 对象（dict/list/...）
    - 跳过空行、以 # 或 // 开头的注释行
    - 解析失败时打印警告但不中断
    """
    p = Path(file_path)  # 将字符串路径转换为Path对象，便于进行路径操作
    if not p.exists():  # 检查文件是否存在
        raise FileNotFoundError(f"File not found: {p.resolve()}")  # 如果文件不存在，抛出文件未找到异常，显示文件的绝对路径

    with p.open("r", encoding="utf-8") as f:  # 以UTF-8编码打开文件进行读取，使用with语句确保文件自动关闭
        for lineno, raw in enumerate(f, 1):  # 遍历文件的每一行，enumerate从1开始计数行号，raw为原始行内容
            line = raw.strip()  # 去除行首尾的空白字符（空格、制表符、换行符等）
            if not line or line.startswith("#") or line.startswith("//"):  # 如果行为空或者以#或//开头（注释行）
                continue  # 跳过当前行，继续处理下一行
            try:  # 尝试解析JSON
                yield json.loads(line)  # 将当前行解析为JSON对象并通过yield返回，实现生成器模式
            except json.JSONDecodeError as e:  # 如果JSON解析失败，捕获JSONDecodeError异常
                print(f"[WARN] JSON decode error at line {lineno}: {e}", file=sys.stderr)  # 打印警告信息到标准错误输出，包含行号和错误详情

def json_to_question(data: dict) -> str:  # 定义函数，将JSON数据转换为问题字符串，参数为字典，返回字符串
    """
    将一个 JSON 对象拼接成一整个字符串形式的问题
    """
    parts = []  # 初始化空列表，用于存储问题的各个组成部分
    
    parts.append(textwrap.dedent(f"""  # 使用textwrap.dedent去除多行字符串的公共缩进，添加编程问题的开头说明
    You will complete a programming problem. Please implement the function `{data['entry_point']}` in python.
    """).strip())  # 从data字典中提取entry_point字段作为函数名，strip()去除首尾空白
    
    parts.append("Problem:\n" + data["prompt"].strip() + "\n")  # 添加问题描述部分，从data中提取prompt字段并去除空白
    parts.append("TESTS:\n" + data["test"].strip() + "\n")  # 添加测试用例部分，从data中提取test字段并去除空白
    return "\n\n".join(parts)  # 将parts列表中的各部分用双换行符连接成一个完整的问题字符串

if __name__ == "__main__":  # 检查是否作为主程序直接运行（而不是被导入）
    original_path = "../../ScoreFlow/data/humaneval_original.jsonl"  # 定义原始HumanEval数据集文件路径
    requirement_path = "../../ScoreFlow/data/humaneval_requirements.jsonl"  # 定义需求变化数据集文件路径
    paraphrasing_path = "../../ScoreFlow/data/humaneval_paraphrasing.jsonl"  # 定义释义变化数据集文件路径
    light_noise_path = "../../ScoreFlow/data/mbpp_light_noise.jsonl"  # 定义MBPP轻度噪声数据集文件路径
    moderate_noise_path = "../../ScoreFlow/data/mbpp_moderate_noise.jsonl"  # 定义MBPP中度噪声数据集文件路径
    heavy_noise_path = "../../ScoreFlow/data/humaneval_heavy_noise.jsonl"  # 定义HumanEval重度噪声数据集文件路径
    all_questions = []  # 初始化空列表，用于存储所有处理后的问题
    
    for obj in iter_jsonl(original_path):  # 遍历原始数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(requirement_path):  # 遍历需求变化数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(paraphrasing_path):  # 遍历释义变化数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(light_noise_path):  # 遍历MBPP轻度噪声数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(moderate_noise_path):  # 遍历MBPP中度噪声数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(heavy_noise_path):  # 遍历HumanEval重度噪声数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    light_noise_path = "../../ScoreFlow/data/humaneval_light_noise.jsonl"  # 重新定义HumanEval轻度噪声数据集文件路径（覆盖之前的MBPP路径）
    moderate_noise_path = "../../ScoreFlow/data/humaneval_moderate_noise.jsonl"  # 重新定义HumanEval中度噪声数据集文件路径（覆盖之前的MBPP路径）
    
    for obj in iter_jsonl(light_noise_path):  # 遍历HumanEval轻度噪声数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    for obj in iter_jsonl(moderate_noise_path):  # 遍历HumanEval中度噪声数据集中的每个JSON对象
        all_questions.append({"question": json_to_question(obj)})  # 将JSON对象转换为问题字符串，并包装在question字段中添加到列表
    
    with open("all_questions.jsonl", "w", encoding="utf-8") as f:  # 以UTF-8编码打开输出文件进行写入，使用with语句确保文件自动关闭
        for q in all_questions:  # 遍历所有处理后的问题
            f.write(json.dumps(q, ensure_ascii=False))  # 将问题字典转换为JSON字符串写入文件，ensure_ascii=False保持中文字符不被转义
            f.write("\n")  # 在每个JSON对象后写入换行符，形成JSONL格式