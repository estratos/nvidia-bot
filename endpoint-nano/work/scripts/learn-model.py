execfile('/home/work/sources/jetson-inference/python/training/classification/train.py --model-dir=ModelTrainData /home/work/ModelTrainData/')

execfile('/home/work/sources/jetson-inference/python/training/classification/onnx_export.py --model-dir=ModelTrainData')
# check output location and names
