# Xinmai AI

本模型基于 `uer/roberta-base-chinese-extractive-qa` 模型进行训练。

## Usage

1. 安装需要的工作环境

    ```bash
    pip -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. 更新 `sample_qa.csv` 文件内容

3. AI 建模与测试

    ```bash
    # training 训练
    python train.py

    # inference 测试
    python inference.py
    ```

## Copyrights

本项目由 中国深圳信迈科技 与 美国硅谷杰圆科技 合作创建，未经允许不得用于任何商业目的。
如需商业化，请联系商务合作。

WeChat ID: xinmai002leo