import os

def build_engine(onnx_file_path):
    # This is where you would add the logic to load the ONNX model and build the engine
    print(f"Building engine from: {onnx_file_path}")
    return "engine_built_successfully"

# File path to your ONNX model
file_path = '~/magisterska/jetson-inference/python/training/detection/ssd/models/fruit/ssd-mobilenet.onnx'
file_path = os.path.expanduser(file_path)  # This ensures '~' is correctly interpreted

# Call the function
engine = build_engine(file_path)
print(engine)

