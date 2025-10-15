#!/bin/bash

# 下载ABACUS-agent-tools
#git clone https://github.com/deepmodelling/ABACUS-agent-tools.git
#cd ABACUS-agent-tools/quick_start

# 提示用户输入conda环境名称，并提供默认值
echo -n "请输入conda环境名称（直接回车使用默认值 abacus_agent）: "
read CONDA_ENV_NAME

# 设置默认值
CONDA_ENV_NAME=${CONDA_ENV_NAME:-my_abacus_agent}

# 创建并激活环境
conda create -n $CONDA_ENV_NAME python=3.11
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# 安装ABACUS-agent-tools
cd ../
pip install . # 依赖安装还需要补充检查
git clone https://gitlab.com/1041176461/ase-abacus.git #ase-abacus
cd ase-abacus
pip install . 
cd ..

pip install google-adk bohr-agent-sdk litellm

cd quick_start

# 下载APNS-PP-ORB-v1赝势和轨道
if [ ! -f "ABACUS-APNS-PPORBs-v1.zip" ]; then
    echo "Downloading dataset..."
    wget https://store.aissquare.com/datasets/af21b5d9-19e6-462f-ada1-532f47f165f2/ABACUS-APNS-PPORBs-v1.zip
else
    echo "Pseudopotential and orbital file already downloaded."
fi
if [ ! -d "apns-orbitals-efficiency-v1" ] || [ ! -d "apns-pseudopotentials-v1" ]; then
    echo "Unzipping dataset..."
    unzip -u ABACUS-APNS-PPORBs-v1.zip
else
    echo "Directory containing pseudopotential and orbital files already exists."
fi

ref_agent_file=$(realpath "./agent.py")
# 设置abacusagent的参数
python3 prepare_abacusagent_env.py
abacusagent --model dp --host 0.0.0.0 --port 50001 > abacusagent.log 2>&1 &

# 创建智能体所需的文件，并启动
if [ ! -d "agent" ]; then
    mkdir agent
fi
pwd
cd agent
abacusagent --create
cp $ref_agent_file abacus-agent/agent.py
adk web --port 50002 --host 0.0.0.0
