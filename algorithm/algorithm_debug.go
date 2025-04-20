package algorithm

import (
	"encoding/json"
	"os"
	"path/filepath"
)

type IntermediateData struct {
	X             float64
	K             int
	T             int
	AllCorners    []float64
	BaseCorners   []float64
	ACorners      []float64
	KeyUsage      map[int][]bool
	ActiveColumns [][]int
	KeyUsage400   map[int][]float64
	Anchor        []float64
	DeltaKs       map[int][]float64
	JBar          []float64
	XBar          []float64
	PBar          []float64
	ABar          []float64
	RBar          []float64
	CStep         []float64
	KsStep        []float64
}

func ExportIntermediate(filePath, mod string, outputDir string) error {
	// 获取所有中间数据
	x, K, T, noteSeq, noteSeqByColumn, lnSeq, tailSeq, _, _ := preprocessFile(filePath, mod)
	allCorners, baseCorners, aCorners := getCorners(T, noteSeq)
	keyUsage := getKeyUsage(K, T, noteSeq, baseCorners)
	activeColumns := getActiveColumns(K, keyUsage)
	keyUsage400 := getKeyUsage400(K, T, noteSeq, baseCorners)
	anchor := computeAnchor(K, keyUsage400, baseCorners)

	deltaKs, jBar := computeJBar(K, T, x, noteSeqByColumn, baseCorners)
	jBar = interpValues(allCorners, baseCorners, jBar)

	xBar := computeXBar(K, T, x, noteSeqByColumn, activeColumns, baseCorners)
	xBar = interpValues(allCorners, baseCorners, xBar)

	lnRep := lnBodiesCountSparse(lnSeq, T)

	pBar := computePBar(K, T, x, noteSeq, lnRep, anchor, baseCorners)
	pBar = interpValues(allCorners, baseCorners, pBar)

	aBar := computeABar(K, T, x, noteSeqByColumn, activeColumns, deltaKs, aCorners, baseCorners)
	aBar = interpValues(allCorners, aCorners, aBar)

	rBar := computeRBar(K, T, x, noteSeqByColumn, tailSeq, baseCorners)
	rBar = interpValues(allCorners, baseCorners, rBar)

	cStep, ksStep := computeCAndKs(K, T, noteSeq, keyUsage, baseCorners)
	cStep = stepInterp(allCorners, baseCorners, cStep)
	ksStep = stepInterp(allCorners, baseCorners, ksStep)

	data := IntermediateData{
		X:             x,
		K:             K,
		T:             T,
		AllCorners:    allCorners,
		BaseCorners:   baseCorners,
		ACorners:      aCorners,
		KeyUsage:      keyUsage,
		ActiveColumns: activeColumns,
		KeyUsage400:   keyUsage400,
		Anchor:        anchor,
		DeltaKs:       deltaKs,
		JBar:          jBar,
		XBar:          xBar,
		PBar:          pBar,
		ABar:          aBar,
		RBar:          rBar,
		CStep:         cStep,
		KsStep:        ksStep,
	}

	// 创建输出目录
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}

	// 生成唯一文件名
	baseName := filepath.Base(filePath)
	outputPath := filepath.Join(outputDir, baseName+".json")

	// 写入JSON文件
	file, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}
