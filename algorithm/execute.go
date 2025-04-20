package algorithm

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

type result struct {
	path   string
	rating float64
	err    error
}

// 对指定目录folderPath下面的所有osu文件进行难度测定
// 如果输入参数为""则默认为执行文件目录
func Walkdir(folderPath string) {
	startTime := time.Now()
	if folderPath == "" {
		folderPath, _ = os.Executable()
		folderPath = filepath.Dir(folderPath)
	}
	// 创建通道和等待组
	paths := make(chan string, runtime.NumCPU()*2)
	results := make(chan result)

	var wgWorkers sync.WaitGroup
	var wgCollector sync.WaitGroup

	// 启动结果收集器
	wgCollector.Add(1)
	go func() {
		defer wgCollector.Done()
		total, processed := 0, 0
		for res := range results {
			total++
			if res.err != nil {
				log.Printf("Error: %s → %v", filepath.Base(res.path), res.err)
				continue
			}
			processed++
			fmt.Printf("%-50s | %7.4f\n", shortenFilename(filepath.Base(res.path), 45), res.rating)
		}
		log.Printf("Processed %d/%d files in %v", processed, total, time.Since(startTime))
	}()

	// 启动工作池
	numWorkers := runtime.NumCPU()
	wgWorkers.Add(numWorkers)
	for i := 0; i < numWorkers; i++ {
		go func() {
			defer wgWorkers.Done()
			for path := range paths {
				rating := Calculate(path, "NM")
				results <- result{path, rating, nil}
			}
		}()
	}

	// 遍历目录
	go func() {
		filepath.Walk(folderPath, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				results <- result{path: path, err: err}
				return nil
			}
			if !info.IsDir() && filepath.Ext(path) == ".osu" {
				paths <- path
			}
			return nil
		})
		close(paths) // 关闭路径通道表示没有更多文件
	}()

	// 等待所有工作器完成
	wgWorkers.Wait()
	close(results)     // 安全关闭结果通道
	wgCollector.Wait() // 等待结果收集器完成
}

// 缩短文件名显示长度
func shortenFilename(name string, maxLen int) string {
	if len(name) <= maxLen {
		return name
	}
	return "..." + name[len(name)-maxLen+3:]
}
