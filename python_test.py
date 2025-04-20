import ctypes
import os
from ctypes import cdll

algorithm = cdll.LoadLibrary('./api.dll')
calc = algorithm.CalculateSR
calc.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
calc.restype = ctypes.c_double

folder_path = "E:/osusbPack/Songs/2147885 VA - SFR's 7 to 6 Convert Practice Pack 1"

if __name__ == "__main__":
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.osu'):
                file_path = os.path.join(root, file)
                result = calc(file_path.encode(), b'NM')
                print(file, "|", f'{result:.4f}')