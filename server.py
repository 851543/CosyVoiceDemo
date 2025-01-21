# 导入必要的库和模块
import os
import sys
from concurrent import futures  # 用于并发处理
import argparse  # 命令行参数解析
import cosyvoice_pb2  # gRPC生成的协议文件
import cosyvoice_pb2_grpc  # gRPC生成的服务文件
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc  # gRPC框架
import torch
import numpy as np

# 设置项目根目录和依赖路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/'.format(ROOT_DIR))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

# 配置日志格式
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    """CosyVoice服务实现类，处理不同模式的语音合成请求"""
    
    def __init__(self, args):
        # 初始化模型，尝试加载CosyVoice或CosyVoice2模型
        try:
            self.cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            try:
                self.cosyvoice = CosyVoice2(args.model_dir)
            except Exception:
                raise TypeError('no valid model_type!')
        logging.info('grpc service initialized')

    def Inference(self, request, context):
        """处理推理请求的主要方法"""
        if request.HasField('sft_request'):
            # 处理SFT（微调）模式请求
            logging.info('get sft inference request')
            model_output = self.cosyvoice.inference_sft(request.sft_request.tts_text, request.sft_request.spk_id)
        elif request.HasField('zero_shot_request'):
            # 处理零样本语音克隆请求
            logging.info('get zero_shot inference request')
            # 将音频数据转换为PyTorch张量并进行归一化
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.zero_shot_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.cosyvoice.inference_zero_shot(request.zero_shot_request.tts_text,
                                                              request.zero_shot_request.prompt_text,
                                                              prompt_speech_16k)
        elif request.HasField('cross_lingual_request'):
            # 处理跨语言语音克隆请求
            logging.info('get cross_lingual inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.cosyvoice.inference_cross_lingual(request.cross_lingual_request.tts_text, prompt_speech_16k)
        else:
            # 处理指令控制模式请求
            logging.info('get instruct inference request')
            model_output = self.cosyvoice.inference_instruct(request.instruct_request.tts_text,
                                                             request.instruct_request.spk_id,
                                                             request.instruct_request.instruct_text)

        # 发送合成结果
        logging.info('send inference response')
        print(model_output)
        for i in model_output:
            response = cosyvoice_pb2.Response()
            # 将浮点数音频转换为16位整数格式
            response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
            yield response


def main():
    # 创建gRPC服务器，配置最大并发数
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    # 注册CosyVoice服务
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    # 绑定服务器地址和端口
    grpcServer.add_insecure_port('127.0.0.1:{}'.format(args.port))
    # 启动服务器
    grpcServer.start()
    logging.info("server listening on 127.0.0.1:{}".format(args.port))
    # 保持服务器运行
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000,
                        help='服务器监听端口')
    parser.add_argument('--max_conc',
                        type=int,
                        default=4,
                        help='最大并发请求数')
    parser.add_argument('--model_dir',
                        type=str,
                        default='D:\Python\Project\CosyVoice\pretrained_models\CosyVoice-300M',
                        help='模型目录路径或modelscope仓库ID')
    args = parser.parse_args()
    main()
