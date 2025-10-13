import json
from pathlib import Path

config = {}

print("""
接下来我将为你配置启动abacusagent所需的设置，请根据提示提供所需的信息。
""")

print("你想将运行abacusagent时产生的文件放在哪个目录？默认为/tmp/abacusagent，按Enter键使用默认值，或输入你想要的路径：")
response = input()
if response.strip() == "":
    output_dir = "/tmp/abacusagent"
else:
    output_dir = response.strip()
config['ABACUSAGENT_WORK_PATH'] = output_dir

config['ABACUSAGENT_SUBMIT_TYPE'] = 'local'

print("请设置运行abacusagent的Host地址，默认为127.0.0.1，按Enter键使用默认值，或输入你希望使用的Host地址：")
response = input()
if response.strip() == "":
    host = "127.0.0.1"
else:
    host = response.strip()
config['ABACUSAGENT_HOST'] = host

print("请设置运行abacusagent的端口号，默认为50001，按Enter键使用默认值，或输入你希望使用的端口号：")
response = input()
if response.strip() == "":
    port = 50001
else:
    port = int(response.strip())
config['ABACUSAGENT_PORT'] = port

print("请设置运行ABACUS的命令，默认为'OMP_NUM_THREADS=1 mpirun -np 4 abacus |tee log'，按Enter键使用默认值，或输入你希望使用的运行ABACUS的命令：")
response = input()
if response.strip() == "":
    command = "OMP_NUM_THREADS=1 mpirun -np 4 abacus |tee log"
else:
    command = response.strip()
config['ABACUS_COMMAND'] = command

pp_path, orb_path = Path("./apns-pseudopotentials-v1"), Path("./apns-orbitals-efficiency-v1")
if Path("./apns-pseudopotentials-v1").exists():
    print(f"ABACUS计算使用的APNS-PP-ORB-V1赝势已被下载到{pp_path.absolute()}，按Enter键使用，或输入你希望使用的赝势的目录：")
    response = input()  
    if response.strip() == "":
        pp_path = ""
    else:
        pp_path = response.strip()
else:
    print(f"未找到默认的APNS-PP-ORB-V1赝势，请输入你希望使用的赝势的目录：")
    response = input()
    if response.strip() == "":
        print("未设置赝势所在目录！")
        pp_path = ""
    else:
        pp_path = response.strip()
config['ABACUS_PP_PATH'] = pp_path

if Path("./apns-orbitals-efficiency-v1").exists():   
    print(f"ABACUS计算使用的APNS-PP-ORB-V1数值原子轨道基组已被下载到{orb_path.absolute()}，按Enter键使用，或输入你希望使用的赝势的目录：")
    response = input()
    if response.strip() == "":
        orb_path = ""
    else:
        orb_path = response.strip()
else:
    print(f"未找到默认的APNS-PP-ORB-V1数值原子轨道基组，请输入你希望使用的数值原子轨道基组的目录：")
    response = input()
    if response.strip() == "":
        print("未设置数值原子轨道基组所在目录！")
        orb_path = ""
    else:
        orb_path = response.strip()
    config['ABACUS_ORB_PATH'] = orb_path

print("请设置LLM模型名称，目前支持直接设置通义千问(openai/qwen-turbo)、豆包(openai/doubao-seed-1-6-250625)和DeepSeek(deepseek/deepseek-chat)，请输入你希望使用的模型名称，默认值为空，需要你自己修改智能体文件，设置LLM模型：")

while True:
    response = input()
    model_name = response.strip()
    
    if model_name == "":
        config['LLM_MODEL'] = ""
        break
    elif model_name in ['openai/qwen-turbo', 'openai/doubao-seed-1-6-250625', 'deepseek/deepseek-chat']:
        config['LLM_MODEL'] = model_name
        break
    else:
        print("未识别的模型名称，请重新输入")

if model_name in ['openai/qwen-turbo', 'openai/qwen-plus', 'openai/qwen-max']:
    config['LLM_BASE_URL'] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
elif model_name in ['openai/doubao-seed-1-6-250625']:
    config['LLM_BASE_URL'] = "https://ark.cn-beijing.volces.com/api/v3"
else:
    config['LLM_BASE_URL'] = ""

print("请设置LLM模型的API Key，默认值为空，需要你自己修改智能体文件，设置API Key：")
response = input()
if response.strip() == "":
    model_api_key = ""
else:
    model_api_key = response.strip()
config['LLM_API_KEY'] = model_api_key

abacusagent_envfile_dir = Path.home() / ".abacusagent"
abacusagent_envfile_dir.mkdir(exist_ok=True)
abacusagent_envfile = abacusagent_envfile_dir / "env.json"
with open(abacusagent_envfile, "w") as f:
    json.dump(config, f, indent=4)
