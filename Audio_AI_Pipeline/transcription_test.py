import moonshine_onnx
text = moonshine_onnx.transcribe('output.wav', 'moonshine/tiny')
print(text[0])