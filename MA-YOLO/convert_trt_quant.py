import numpy as np
from PIL import Image
import glob, os
import util_trt
from utils.utils import cvtColor, resize_image, preprocess_input


class DataLoader:
    def __init__(self, input_shape, batch, batch_size, img_type, data_path):
        self.index = 0
        self.length = batch
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_list = glob.glob(os.path.join(data_path, "*." + img_type))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            data_path) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros(
            (self.batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3]), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'

                image = Image.open(self.img_list[i + self.index * self.batch_size])

                # ---------------------------------------------------------#
                #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
                #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
                # ---------------------------------------------------------#
                image = cvtColor(image)
                # ---------------------------------------------------------#
                #   给图像增加灰条，实现不失真的resize
                #   也可以直接resize进行识别
                # ---------------------------------------------------------#
                image_data = resize_image(image, (self.input_shape[-2], self.input_shape[-1]), False)
                # ---------------------------------------------------------#
                #   添加上batch_size维度
                # ---------------------------------------------------------#
                image_data = np.expand_dims(
                    np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

                self.calibration_data[i] = image_data

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


def main():
    # onnx2trt

    # int8_mode = True
    int8_mode = False
    print('*** onnx to tensorrt begin ***')
    # calibration
    onnx_file_path = "onnx_file/yolov4_800_simplified_model.onnx"
    engine_model_path = "trt_file/yolov4_800.trt"
    cache_file = "trt_cache/yolov4_800_cache.cache"
    BATCH_SIZE = 5
    BATCH = 50
    data_path = 'D:\Dataset\DIOR\JPEGImages-trainval'
    img_type = 'jpg'
    input_shape = [1, 3, 800, 800]
    calibration_stream = DataLoader(input_shape, BATCH, BATCH_SIZE, img_type, data_path)

    para_dict = {"int8_mode": int8_mode, "calibrator_stream": calibration_stream, "cache_file": cache_file}

    # fixed_engine,校准产生校准表

    engine_fixed = util_trt.get_engine(onnx_file_path, input_shape, engine_model_path, **para_dict)
    # assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')


if __name__ == '__main__':
    main()
