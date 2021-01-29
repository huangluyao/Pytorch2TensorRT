import os


trtexec_path = "/home/hly/workspace/library/TensorRT-7.2.1.6/bin/trtexec"
onnx_file = "weights/yolov4.onnx"
save_path = "weights/yolov4_fp32.engine"
workspace =120

run = "%s --onnx=%s --explicitBatch --saveEngine=%s --workspace=%s " %(trtexec_path,
                                                                             onnx_file,
                                                                             save_path,
                                                                             workspace)
print(os.system(run))

print("create engine file done")