## 简介

利用 PaddleOCR2Pytorch 尝试将最新出的 `PP-OCRv5` 模型转换成 Pytorch 模型，但不是很成功。

效果：能顺利转换，但转换后的模型在推理时，效果很差，合理推测是部分参数未正确写入torch结构。


## 项目核心结构

- ppocr：paddle原始的模型组件结构

- pytorchocr：torch重新实现的模型组件结构

项目核心是根据 `ppocr`的结构，用`pytorch`重新实现，参数接口一一对应

## 使用方式

以转换`PP-OCRv5_server_rec_infer`模型为例：

1. 创建 PP-OCRv5_server_rec_infer 文件夹

2. 从paddleocr官网下载 PP-OCRv5_server_rec_pretrained.pdparams （预训练模型），放置到PP-OCRv5_server_rec_infer文件夹下

3. 运行转换脚本

```bash
python ./converter/ch_ppocr_v5_rec_server_converter.py --yaml_path ./configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml --src_model_path PP-OCRv5_server_rec_infer --save_path ./PP-OCRv5_server_rec_infer/ch_PP-OCRv5_rec_server_infer.pth
```

## 注意事项
主体代码用Agent生成，需严格鉴别其正确性。
