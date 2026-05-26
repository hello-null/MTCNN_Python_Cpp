import torch
import numpy as np
from mtcnn.get_nets import PNet, RNet, ONet  # 假设网络定义在此文件中
import onnxruntime as ort

'''
送入 P-Net 的张量形状为 [batch_size, 3, H, W]（其中 H 和 W 是缩放后图像的高和宽）。
经过 P-Net 的卷积层后，输出的两个张量分别为：
    b (offsets)	[batch_size, 4, H', W']	每个滑动窗口位置的 边界框回归偏移量，原始 logits，无激活
        边界框回归偏移量（shape: [batch_size, 4, H', W']）
        b[0, 0, i, j] → tx1：窗口左上角 x 坐标的修正比例（相对于窗口宽度）
        b[0, 1, i, j] → ty1：窗口左上角 y 坐标的修正比例（相对于窗口高度）
        b[0, 2, i, j] → tx2：窗口右下角 x 坐标的修正比例（相对于窗口宽度）
        b[0, 3, i, j] → ty2：窗口右下角 y 坐标的修正比例（相对于窗口高度）
        修正公式（用于将原始的 12×12 固定窗口调整为更贴合真实人脸的框）：
        x1_true = x1 + tx1 * w
        y1_true = y1 + ty1 * h
        x2_true = x2 + tx2 * w
        y2_true = y2 + ty2 * h
    a (probs)	[batch_size, 2, H', W']	每个滑动窗口位置的 人脸/非人脸分类概率，经过 softmax 激活
        人脸分类概率（shape: [batch_size, 2, H', W']）
        例如，a[0, 1, i, j] 表示第 i 行、第 j 列的 12×12 窗口内包含人脸的置信度。



R-Net 结构：相比 P-Net 更深，最后有全连接层，输入固定为 24×24，输出两个分支：
b：边界框回归偏移量，形状 [M, 4]。原始 logits，无激活
a：人脸分类概率，形状 [M, 2]（经过 softmax，两列分别为非人脸、人脸概率）。




O-Net:
输入：[L, 3, 48, 48]。
输出三个张量（注意顺序与 R-Net 不同）：
c：人脸关键点坐标，形状 [L, 10]。原始 logits，无激活
b：边界框回归偏移量，形状 [L, 4]。原始 logits，无激活
a：人脸分类概率，形状 [L, 2]。经过 softmax 激活
landmarks：形状 [L, 10]，每行 10 个值，分别对应 5 个关键点的相对坐标（相对于当前候选框的宽度和高度比例）。
顺序为：(x1, x2, x3, x4, x5, y1, y2, y3, y4, y5)。
offsets 和 probs：含义与前两个阶段相同。
'''




def export_pnet(weights_path="./weights/pnet.npy", output_path="pnet.onnx"):
    # 1. 创建模型并加载权重
    model = PNet()
    weights = np.load(weights_path, allow_pickle=True)[()]
    state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. 定义示例输入 (batch_size=1, 3通道, 任意H,W)
    #    这里用 120x120 作为示例，实际推理时 H,W 可变
    dummy_input = torch.randn(1, 3, 120, 120)

    # 3. 导出 ONNX，设置动态轴
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["bbox_offsets", "face_probs"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "bbox_offsets": {0: "batch", 2: "out_height", 3: "out_width"},
            "face_probs": {0: "batch", 2: "out_height", 3: "out_width"}
        },
        opset_version=11
    )
    print(f"P-Net exported to {output_path}")

def export_rnet(weights_path="./weights/rnet.npy", output_path="rnet.onnx"):
    model = RNet()
    weights = np.load(weights_path, allow_pickle=True)[()]
    state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 24, 24)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["bbox_offsets", "face_probs"],
        dynamic_axes={
            "input": {0: "batch"},
            "bbox_offsets": {0: "batch"},
            "face_probs": {0: "batch"}
        },
        opset_version=11
    )
    print(f"R-Net exported to {output_path}")

def export_onet(weights_path="./weights/onet.npy", output_path="onet.onnx"):
    model = ONet()
    weights = np.load(weights_path, allow_pickle=True)[()]
    state_dict = {k: torch.FloatTensor(v) for k, v in weights.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dummy_input = torch.randn(1, 3, 48, 48)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["landmarks", "bbox_offsets", "face_probs"],
        dynamic_axes={
            "input": {0: "batch"},
            "landmarks": {0: "batch"},
            "bbox_offsets": {0: "batch"},
            "face_probs": {0: "batch"}
        },
        opset_version=11
    )
    print(f"O-Net exported to {output_path}")

def export_onnx():
    # 假设权重文件在当前目录下的 weights/ 文件夹中
    export_pnet("./weights/pnet.npy", "./onnx/pnet.onnx")
    export_rnet("./weights/rnet.npy", "./onnx/rnet.onnx")
    export_onet("./weights/onet.npy", "./onnx/onet.onnx")

def test_onnx(onnx_path, input_shape):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})
    print(f"Test {onnx_path}:")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: shape {out.shape}")

def test_export_onnx():
    # 测试 P-Net (可变输入)
    test_onnx("./onnx/pnet.onnx", (1, 3, 240, 320))
    # 测试 R-Net
    test_onnx("./onnx/rnet.onnx", (10, 3, 24, 24))
    # 测试 O-Net
    test_onnx("./onnx/onet.onnx", (5, 3, 48, 48))


if __name__ == '__main__':
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    print("CUDA版本:", torch.version.cuda)

    # export_onnx()
    test_export_onnx()
    pass