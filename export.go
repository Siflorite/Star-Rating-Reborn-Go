//go:build ignore
// +build ignore

package main

import "C"
import algorithm "star-rating-rebirth-go/algorithm"

//export CalculateSR
func CalculateSR(filePath *C.char, mod *C.char) C.double {
	goFilePath := C.GoString(filePath)
	goMod := C.GoString(mod)

	result := algorithm.Calculate(goFilePath, goMod)
	return C.double(result)
}

// If you want to generate dll file, delete the annotation "// +build ignore" and add it to main.go, so that the two main functions won't conflict
// Then you need to install gcc (MinGW is recommended for Windows) and add the directory of bin folder to environment
// Finally, run the instruction below
// go build -ldflags "-s -w" -buildmode=c-shared -o api.dll .\export.go
func main() {}
