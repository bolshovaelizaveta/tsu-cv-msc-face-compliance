import cv2
import numpy as np
import onnxruntime as ort

class FaceSegmentationV2:
    def __init__(self, model_path="models/modnet_photographic_portrait_matting.onnx"):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 8
        self.session = ort.InferenceSession(model_path, sess_options)

    def remove_background(self, crop, face_center):
        h, w = crop.shape[:2]
        input_img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_tensor = (input_img.astype(np.float32) - 127.5) / 128.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))[np.newaxis, :, :, :]

        matte = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0][0, 0, :, :]
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

        # Очистка мусора по центру
        matte_uint8 = (matte * 255).astype(np.uint8)
        _, bin_mask = cv2.threshold(matte_uint8, 10, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        
        target_label = labels[face_center[1], face_center[0]]
        if target_label == 0 and num_labels > 1:
            target_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        if target_label > 0:
            matte = matte * (labels == target_label)

        matte = np.clip((matte - 0.2) / 0.8, 0.0, 1.0)
        matte_3ch = np.stack([matte] * 3, axis=-1)

        bg = np.full(crop.shape, 255, dtype=np.uint8)
        return (crop.astype(float) * matte_3ch + bg.astype(float) * (1.0 - matte_3ch)).astype(np.uint8)