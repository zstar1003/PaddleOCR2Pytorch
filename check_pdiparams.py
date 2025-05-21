import paddle
import os

# --- 修改这里为你实际的 .pdiparams 文件路径 ---
params_path = r"D:\Code\PaddleOCR2Pytorch\PP-OCRv5_server_rec_infer\PP-OCRv5_server_rec_pretrained.pdparams"
# ---------------------------------------------

if not os.path.exists(params_path):
    print(f"错误: 文件未找到: {params_path}")
else:
    print(f"尝试加载文件: {params_path}")
    try:
        loaded_object = paddle.load(params_path)
        print(f"成功加载! 对象类型: {type(loaded_object)}")

        if isinstance(loaded_object, paddle.Tensor):
            print("文件内容是一个单独的 Paddle Tensor。")
            print(f"  - 张量形状 (Shape): {loaded_object.shape}")
            print(f"  - 张量数据类型 (Dtype): {loaded_object.dtype}")
            # 如果张量不大，可以尝试打印部分数据
            # numpy_array = loaded_object.numpy()
            # print(f"  - 部分数据 (前10个值): {numpy_array.flatten()[:10]}")
        elif isinstance(loaded_object, dict):
            print("文件内容是一个字典 (期望的 state_dict 格式)。")
            print(f"  - 字典中的键数量: {len(loaded_object)}")
            print("  - 前5个键和对应值的类型/形状:")
            for i, (key, value) in enumerate(loaded_object.items()):
                if i < 5:
                    value_shape = (
                        value.shape if hasattr(value, "shape") else "N/A (非张量)"
                    )
                    print(
                        f"    键: '{key}', 值类型: {type(value)}, 值形状: {value_shape}"
                    )
                elif i == 5:
                    print("    ... (以及更多键)")
                    break
        else:
            print(f"文件内容是其他未知类型: {loaded_object}")

    except Exception as e:
        print(f"加载文件时发生错误: {e}")
