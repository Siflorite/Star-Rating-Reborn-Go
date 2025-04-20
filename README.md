# Star-Rating-Reborn-Go
A go translation of the new osu!mania [Star Rating algorithm](https://github.com/sunnyxxy/Star-Rating-Rebirth) of [sunnyxxy](https://github.com/sunnyxxy).  
I am a complete noob in Go and the whole program is directly translated from the 2025-04-15 Python version through DeepSeek.

## Usage
### Excecutable file (.exe)
This file does the same function of the srcalc.exe made by sunnyxxy, simply put it under the folder an run it, it will display the calculated ratings of all .osu files under its directory.  
Currently there are bugs that the executable file will stop working when processing files not legal (e.g. Non-mania maps, maps with empty columns), I'm working on how to fix it. So now please check that the osu files are all "legal" before using it.
### Dynamic Link Library (.dll)
A dll file is also compiled for convenient use in other Programming Languanges such as Python. An example of python using this dll `python_test.py` is given.  
A simple use is as follow:
```python
import ctypes
import os
from ctypes import cdll

algorithm = cdll.LoadLibrary('./api.dll')
calc = algorithm.CalculateSR
calc.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
calc.restype = ctypes.c_double

file_path = "YourOsuFile.osu"

if __name__ == "__main__":
    result = calc(file_path.encode(), b'NM')
    print(result)
```
The DLL file exposes a C interface, so data need to be transformed to C types as aruguments. The C interface can be sen as `double CalculateSR(char* file, char* mod)`, so strings in python need to be transformed to C char pointers. If you use a direct assign, just change "content" to b"content". For a string variable `str`, you need to use `str.encode()` to change it into bytes.
