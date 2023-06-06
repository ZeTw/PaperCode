# -*-coding:utf-8-*-
import onnx
import torch, os

from utils.utils import get_classes, get_anchors
# from nets.yolo import YoloBody
from nets.ma_yolo import YoloBody
import onnxruntime
from onnxsim import simplify


def export_onnx(pth, onnx_path, input_shape, is_simplify):
    # classes_path = 'model_data/dior_classes.txt'
    classes_path = 'model_data/rsod_classes.txt'

    # anchors_path = 'model_data/dior_anchors2.txt'
    anchors_path = 'model_data/rsod_anchors.txt'
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model = YoloBody(anchors_mask, num_classes, pretrained=False, phi=2)
    # model = YoloBody(anchors_mask, num_classes, pretrained=False)
    if os.path.exists(pth):
        state_dict = torch.load(pth)
        model.load_state_dict(state_dict)

    model.eval()
    device = torch.device("cuda")
    model = model.to(device)
    input_names = ["input0"]
    output_names = ["output0"]
    inputs = torch.randn(input_shape).to(device)
    torch.onnx._export(model, inputs, onnx_path, export_params=True, verbose=False,
                       keep_initializers_as_inputs=True, input_names=input_names, output_names=output_names,
                       opset_version=11)
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    if is_simplify:
        model_simplified, check = simplify(model_onnx)

        # 确保简化后的模型和原始模型是等价的
        assert check, "The simplified model is not equivalent to the original model."

        # 保存简化后的模型
        onnx.save(model_simplified, onnx_path.split('.')[0] + '_simplified_model.onnx')


if __name__ == "__main__":
    output_onnx = 'onnx_file/RSOD/mayolo_512.onnx'
    # pth = "model_data/best_epoch_weights_800_2.pth"
    pth = "pth/RSOD_pth/rsod_anchor_best.pth"
    input_shape = (1, 3, 512, 512)
    # pth = ""
    export_onnx(pth, output_onnx, input_shape, True)
