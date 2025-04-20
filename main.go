package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	algorithm "star-rating-rebirth-go/algorithm"
)

// go build -ldflags "-s -w" -o calcSR.exe .\main.go
func main() {
	algorithm.Walkdir("")
	fmt.Println("Program Finished. Press ENTER key to continue...")
	// 跨平台保持窗口打开
	if runtime.GOOS == "windows" {
		// Windows专用：调用系统命令暂停
		cmd := exec.Command("cmd", "/c", "pause")
		cmd.Stdin = os.Stdin
		cmd.Stdout = os.Stdout
		cmd.Run()
	} else {
		// 其他系统使用标准输入等待
		fmt.Scanln() // 等待回车
	}
}
