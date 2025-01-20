import subprocess
import os

def convert_to_16k(input_file, output_file=None):
    """
    将音频文件转换为16000Hz采样率
    :param input_file: 输入音频文件路径
    :param output_file: 输出音频文件路径，如果不指定则自动生成
    :return: 输出文件路径
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    # 如果没有指定输出文件，则自动生成输出文件名
    if output_file is None:
        filename, ext = os.path.splitext(input_file)
        output_file = f"{filename}_16k{ext}"
    
    # 构建ffmpeg命令
    command = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-ar', '16000',    # 设置采样率为16000Hz
        '-y',             # 覆盖已存在的文件
        output_file
    ]
    
    try:
        # 执行ffmpeg命令
        subprocess.run(command, check=True, capture_output=True)
        print(f"转换完成: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e.stderr.decode()}")
        raise

if __name__ == "__main__":
    # 示例使用
    input_file = "./asset/test.wav"
    try:
        output_file = convert_to_16k(input_file)
        print(f"文件已转换: {output_file}")
    except Exception as e:
        print(f"转换过程中出错: {e}") 