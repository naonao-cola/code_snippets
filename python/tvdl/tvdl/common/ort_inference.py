'''
Copyright (C) 2023 TuringVision

OnnxRuntime inference class.
'''
import os
import os.path as osp
import numpy as np

__all__ = ['OrtInference']

PROVIDER_DICT = {
    'tensorrt': 'TensorrtExecutionProvider',
    'cuda': 'CUDAExecutionProvider',
    'openvino': 'OpenVINOExecutionProvider',
    'cpu':'CPUExecutionProvider'
}


class OrtInference:
    '''Basic class for wrap onnxruntime inference code for tsnn models
    '''
    def __init__(self, model_path, devices=['cuda'],
                 sess_options=None,
                 provider_options=None,
                 trt_cache_path=None,
                 ort_batch_size=1):
        """
        Args:
            model_path: onnx model path
            devices: can be any combination of ['cuda', 'tensorrt', 'openvino', 'cpu']
            sess_options: inference session options
            provider_options: provider options
            trt_cache_path: trt engine file cache path
            batch_size: onnxruntime inference batch_size
        """
        self.ort_session = self._init_ort_session(model_path, devices, sess_options,
                                                  provider_options, trt_cache_path)
        # Warmup the model
        input_shape = self.ort_session.get_inputs()[0].shape
        input_shape = [ort_batch_size if isinstance(item, str) else item for item in input_shape]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        self.forward(dummy_input, ort_batch_size)

    def _init_ort_session(self, model_path, devices,
                          sess_options,
                          provider_options,
                          trt_cache_path):
        import onnxruntime as ort
        available_prs = ort.get_available_providers()
        providers = [PROVIDER_DICT[dev] for dev in devices
                        if PROVIDER_DICT[dev] in available_prs]

        # TensorRT cache dir
        if trt_cache_path is None:
            trt_cache_path = osp.abspath(osp.dirname(model_path))
        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'
        os.environ['ORT_TENSORRT_CACHE_PATH'] = trt_cache_path

        if sess_options is None:
            sess_options = ort.SessionOptions()
            if 'openvino' in providers:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            else:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        return ort.InferenceSession(model_path,
                                    providers=providers,
                                    sess_options=sess_options,
                                    provider_options=provider_options)

    def forward(self, images, ort_batch_size=1):
        '''
        images: batch numpy or tensor, shape like:[n,c,h,w]
        ort_batch_size: onnxruntime inference batch_size
        '''
        if not isinstance(images, np.ndarray):
            images = images.numpy()
        real_bs = images.shape[0]
        if images.shape[0] < ort_batch_size:
            input_shape = [ort_batch_size - images.shape[0], images.shape[1], images.shape[2], images.shape[3]]
            add_image = np.random.randn(*input_shape).astype(np.float32)
            images = np.concatenate((images, add_image), axis=0)
        input_name = self.ort_session.get_inputs()[0].name
        inputs = {input_name: images}
        outputs = [self.ort_session.run(None, inputs)[0][:real_bs]]
        return outputs

