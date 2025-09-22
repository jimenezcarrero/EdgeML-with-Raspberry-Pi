import pyaudio
p = pyaudio.PyAudio()
for ii in range(p.get_device_count()):
    print(ii, p.get_device_info_by_index(ii).get('name'))