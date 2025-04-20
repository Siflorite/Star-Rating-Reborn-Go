package algorithm

import (
	"fmt"
	"math"
	"sort"
	"star-rating-rebirth-go/osu_parser"
)

type Note struct {
	Column   int
	HeadTime int
	TailTime int
}

// LNRep 长按音符稀疏表示结构
type LNRep struct {
	Points []int     // 时间断点
	CumSum []float64 // 累积和
	Values []float64 // 分段常数值
}

func preprocessFile(filePath, mod string) (
	float64, // x
	int, // K
	int, // T
	[]Note, // noteSeq
	[][]Note, // noteSeqByColumn
	[]Note, // LNSeq
	[]Note, // tailSeq
	[][]Note, // LNSeqByColumn
	error, // error
) {
	parser := osu_parser.NewParser(filePath)
	if err := parser.Process(); err != nil {
		fmt.Printf("Error processing file: %v\n", err)
		return 0, 0, 0, nil, nil, nil, nil, nil, err
	}
	columnCount, columns, starts, ends, types, od := parser.GetParsedData()
	var noteSeq []Note
	for i := range columns {
		k := columns[i] // 列索引
		h := starts[i]  // 起始时间
		t := -1         // 默认单点音符

		// 长按音符处理
		if types[i] == 128 {
			t = ends[i]
		}
		// 处理mod
		switch mod {
		case "DT": // Double Time
			h = int(math.Floor(float64(h) * 2 / 3))
			if t >= 0 {
				t = int(math.Floor(float64(t) * 2 / 3))
			}
		case "HT": // Half Time
			h = int(math.Floor(float64(h) * 4 / 3))
			if t >= 0 {
				t = int(math.Floor(float64(t) * 4 / 3))
			}
		}
		noteSeq = append(noteSeq, Note{
			Column:   k,
			HeadTime: h,
			TailTime: t,
		})
	}

	// 计算击打宽容度x
	x := 0.3 * math.Sqrt((64.5-math.Ceil(od*3))/500)
	x = math.Min(x, 0.6*(x-0.09)+0.09)

	// 排序：先按起始时间，再按列索引
	sort.Slice(noteSeq, func(i, j int) bool {
		if noteSeq[i].HeadTime == noteSeq[j].HeadTime {
			return noteSeq[i].Column < noteSeq[j].Column
		}
		return noteSeq[i].HeadTime < noteSeq[j].HeadTime
	})

	noteSeqByColumn := make([][]Note, columnCount)
	for k := 0; k < columnCount; k++ {
		noteSeqByColumn[k] = make([]Note, 0)
	}
	for _, note := range noteSeq {
		if note.Column >= 0 && note.Column < columnCount {
			noteSeqByColumn[note.Column] = append(noteSeqByColumn[note.Column], note)
		}
	}

	// 处理长按音符
	var LNSeq []Note
	for _, note := range noteSeq {
		if note.TailTime >= 0 {
			LNSeq = append(LNSeq, note)
		}
	}

	// 按尾键时间排序
	tailSeq := make([]Note, len(LNSeq))
	copy(tailSeq, LNSeq)
	sort.Slice(tailSeq, func(i, j int) bool {
		if tailSeq[i].TailTime != tailSeq[j].TailTime {
			return tailSeq[i].TailTime < tailSeq[j].TailTime
		}
		if tailSeq[i].HeadTime != tailSeq[j].HeadTime {
			return tailSeq[i].HeadTime < tailSeq[j].HeadTime
		}
		return tailSeq[i].Column < tailSeq[j].Column
	})

	// 长按按列分组
	LNSeqByColumn := make([][]Note, columnCount)
	for k := 0; k < columnCount; k++ {
		LNSeqByColumn[k] = make([]Note, 0)
	}
	for _, note := range LNSeq {
		if note.Column >= 0 && note.Column < columnCount {
			LNSeqByColumn[note.Column] = append(LNSeqByColumn[note.Column], note)
		}
	}

	// 计算总时长T
	maxTime := 0
	for _, note := range noteSeq {
		if note.HeadTime > maxTime {
			maxTime = note.HeadTime
		}
		if note.TailTime > maxTime {
			maxTime = note.TailTime
		}
	}
	T := maxTime + 1
	return x, columnCount, T, noteSeq, noteSeqByColumn, LNSeq, tailSeq, LNSeqByColumn, nil
}

// getCorners 生成用于插值的关键时间点集合
// 返回：
// allCorners - 所有关键时间点（合并后的）
// baseCorners - 基础关键时间点
// aCorners - 用于A计算的关键时间点
func getCorners(T int, noteSeq []Note) ([]float64, []float64, []float64) {
	baseSet := make(map[int]struct{})
	for _, note := range noteSeq {
		baseSet[note.HeadTime] = struct{}{}
		if note.TailTime >= 0 {
			baseSet[note.TailTime] = struct{}{}
		}
	}

	// 扩展基础时间点
	tempBase := make([]int, 0, len(baseSet)*3)
	for s := range baseSet {
		tempBase = append(tempBase, s)
		tempBase = append(tempBase, s+501)
		tempBase = append(tempBase, s-499)
		tempBase = append(tempBase, s+1)
	}
	for _, s := range tempBase {
		baseSet[s] = struct{}{}
	}
	baseSet[0] = struct{}{}
	baseSet[T] = struct{}{}

	// 过滤并排序基础时间点
	baseCorners := filterAndSort(baseSet, T)

	// 处理A时间点集合
	aSet := make(map[int]struct{})
	for _, note := range noteSeq {
		aSet[note.HeadTime] = struct{}{}
		if note.TailTime >= 0 {
			aSet[note.TailTime] = struct{}{}
		}
	}

	// 扩展A时间点
	tempA := make([]int, 0, len(aSet)*2)
	for s := range aSet {
		tempA = append(tempA, s)
		tempA = append(tempA, s+1000)
		tempA = append(tempA, s-1000)
	}
	for _, s := range tempA {
		aSet[s] = struct{}{}
	}
	aSet[0] = struct{}{}
	aSet[T] = struct{}{}

	// 过滤并排序A时间点
	aCorners := filterAndSort(aSet, T)

	// 合并所有时间点
	allSet := make(map[int]struct{})
	for _, s := range baseCorners {
		allSet[int(s)] = struct{}{}
	}
	for _, s := range aCorners {
		allSet[int(s)] = struct{}{}
	}

	// 转换为float64并排序
	allCorners := make([]float64, 0, len(allSet))
	for s := range allSet {
		allCorners = append(allCorners, float64(s))
	}
	sort.Float64s(allCorners)

	return allCorners, baseCorners, aCorners
}

func filterAndSort(timeSet map[int]struct{}, T int) []float64 {
	filtered := make([]float64, 0, len(timeSet))
	for s := range timeSet {
		if s >= 0 && s <= T {
			filtered = append(filtered, float64(s))
		}
	}
	sort.Float64s(filtered)
	return filtered
}

func getKeyUsage(K int, T int, noteSeq []Note, baseCorners []float64) map[int][]bool {
	keyUsage := make(map[int][]bool, K)
	for k := 0; k < K; k++ {
		keyUsage[k] = make([]bool, len(baseCorners))
	}
	for _, note := range noteSeq {
		k := note.Column
		h := note.HeadTime
		t := note.TailTime

		// 计算时间区间
		startTime := math.Max(float64(h)-150, 0)
		var endTime float64
		if t < 0 {
			endTime = float64(h) + 150
		} else {
			endTime = math.Min(float64(t)+150, float64(T-1))
		}

		// 二分查找索引
		leftIdx := sort.SearchFloat64s(baseCorners, startTime)
		rightIdx := sort.SearchFloat64s(baseCorners, endTime)

		// 设置区间为true
		for i := leftIdx; i < rightIdx; i++ {
			if i < len(baseCorners) {
				keyUsage[k][i] = true
			}
		}
	}
	return keyUsage
}

func getKeyUsage400(K int, T int, noteSeq []Note, baseCorners []float64) map[int][]float64 {
	keyUsage400 := make(map[int][]float64, K)
	for k := 0; k < K; k++ {
		keyUsage400[k] = make([]float64, len(baseCorners))
	}

	for _, note := range noteSeq {
		k := note.Column
		h := note.HeadTime
		t := note.TailTime

		startTime := math.Max(float64(h), 0)
		var endTime float64
		if t < 0 {
			endTime = float64(h)
		} else {
			endTime = math.Min(float64(t), float64(T-1))
		}

		// 计算四个边界索引
		left400Idx := sort.SearchFloat64s(baseCorners, startTime-400)
		leftIdx := sort.SearchFloat64s(baseCorners, startTime)
		rightIdx := sort.SearchFloat64s(baseCorners, endTime)
		right400Idx := sort.SearchFloat64s(baseCorners, endTime+400)

		// 中间区间计算
		duration := math.Min(endTime-startTime, 1500)
		for i := leftIdx; i < rightIdx; i++ {
			if i < len(baseCorners) {
				keyUsage400[k][i] += 3.75 + duration/150
			}
		}

		// 左衰减区间
		for i := left400Idx; i < leftIdx; i++ {
			if i >= 0 && i < len(baseCorners) {
				delta := baseCorners[i] - startTime
				keyUsage400[k][i] += 3.75 - 3.75/(400*400)*delta*delta
			}
		}

		// 右衰减区间
		for i := rightIdx; i < right400Idx; i++ {
			if i < len(baseCorners) {
				delta := math.Abs(baseCorners[i] - endTime)
				keyUsage400[k][i] += 3.75 - 3.75/(400*400)*delta*delta
			}
		}
	}
	return keyUsage400
}

func getActiveColumns(K int, keyUsage map[int][]bool) [][]int {
	if len(keyUsage) == 0 {
		return nil
	}
	numPoints := len(keyUsage[0])
	activeColumns := make([][]int, numPoints)

	for i := 0; i < numPoints; i++ {
		cols := make([]int, 0, K)
		for k := 0; k < K; k++ {
			if keyUsage[k][i] {
				cols = append(cols, k)
			}
		}
		activeColumns[i] = cols
	}
	return activeColumns
}

func computeAnchor(K int, keyUsage400 map[int][]float64, baseCorners []float64) []float64 {
	anchor := make([]float64, len(baseCorners))

	for idx := range baseCorners {
		// 收集当前时间点的所有键值计数
		counts := make([]float64, K)
		for k := 0; k < K; k++ {
			counts[k] = keyUsage400[k][idx]
		}

		// 降序排序
		sort.Slice(counts, func(i, j int) bool {
			return counts[i] > counts[j]
		})

		// 过滤非零值
		var nonzeroCounts []float64
		for _, v := range counts {
			if v > 1e-9 { // 考虑浮点精度误差
				nonzeroCounts = append(nonzeroCounts, v)
			}
		}

		// 计算walk指标
		if len(nonzeroCounts) > 1 {
			var walk, maxWalk float64
			for i := 0; i < len(nonzeroCounts)-1; i++ {
				ratio := nonzeroCounts[i+1] / nonzeroCounts[i]
				term := nonzeroCounts[i] * (1 - 4*math.Pow(0.5-ratio, 2))
				walk += term
				maxWalk += nonzeroCounts[i]
			}

			if maxWalk > 1e-9 {
				anchor[idx] = walk / maxWalk
			}
		}

		// 应用最终变换公式
		diff := anchor[idx] - 0.18
		cubicTerm := 5 * math.Pow(anchor[idx]-0.22, 3)
		anchor[idx] = 1 + math.Min(diff, cubicTerm)
	}

	return anchor
}

// -----Helper methods--------

// cumulativeSum 计算累积积分 (对应Python的cumulative_sum)
// x: 已排序的时间点切片
// f: 对应区间的函数值切片
func cumulativeSum(x []float64, f []float64) []float64 {
	n := len(x)
	F := make([]float64, n)
	for i := 1; i < n; i++ {
		dx := x[i] - x[i-1]
		F[i] = F[i-1] + f[i-1]*dx
	}
	return F
}

// queryCumSum 查询累积值 (对应Python的query_cumsum)
func queryCumSum(q float64, x []float64, F []float64, f []float64) float64 {
	if len(x) == 0 {
		return 0.0
	}
	if q <= x[0] {
		return 0.0
	}
	if q >= x[len(x)-1] {
		return F[len(F)-1]
	}

	// 二分查找对应区间
	i := sort.SearchFloat64s(x, q) - 1
	if i < 0 {
		i = 0
	}
	return F[i] + f[i]*(q-x[i])
}

// smoothOnCorners 滑动窗口平滑 (对应Python的smooth_on_corners)
func smoothOnCorners(x []float64, f []float64, window float64, scale float64, mode string) []float64 {
	F := cumulativeSum(x, f)
	g := make([]float64, len(x))

	for i, s := range x {
		// 计算窗口边界
		a := math.Max(s-window, x[0])
		b := math.Min(s+window, x[len(x)-1])

		// 计算积分值
		val := queryCumSum(b, x, F, f) - queryCumSum(a, x, F, f)

		// 处理不同模式
		if mode == "avg" {
			if b-a > 1e-9 { // 避免除以0
				g[i] = val / (b - a)
			}
		} else {
			g[i] = scale * val
		}
	}
	return g
}

// interpValues 线性插值 (对应Python的interp_values)
func interpValues(newX []float64, oldX []float64, oldVals []float64) []float64 {
	result := make([]float64, len(newX))
	for i, x := range newX {
		// 找到插入位置
		idx := sort.SearchFloat64s(oldX, x)
		if idx == 0 {
			result[i] = oldVals[0]
		} else if idx == len(oldX) {
			result[i] = oldVals[len(oldVals)-1]
		} else {
			// 线性插值计算
			x0 := oldX[idx-1]
			x1 := oldX[idx]
			y0 := oldVals[idx-1]
			y1 := oldVals[idx]
			t := (x - x0) / (x1 - x0)
			result[i] = y0 + t*(y1-y0)
		}
	}
	return result
}

// stepInterp 阶梯插值 (对应Python的step_interp)
func stepInterp(newX []float64, oldX []float64, oldVals []float64) []float64 {
	result := make([]float64, len(newX))
	for i, x := range newX {
		// 查找右边界索引
		idx := sort.Search(len(oldX), func(j int) bool { return oldX[j] > x })
		// 减1得到最后一个≤x的索引
		idx--
		// 边界保护
		if idx < 0 {
			idx = 0
		} else if idx >= len(oldVals) {
			idx = len(oldVals) - 1
		}
		result[i] = oldVals[idx]
	}
	return result
}

// rescaleHigh 难度重缩放 (对应Python的rescale_high)
func rescaleHigh(sr float64) float64 {
	if sr <= 9 {
		return sr
	}
	return 9 + (sr-9)*(1.0/1.2)
}

// findNextNoteInColumn 查找下一个音符 (对应Python的find_next_note_in_column)
func findNextNoteInColumn(note Note, times []int, noteSeqByColumn [][]Note) Note {
	k := note.Column
	// 在当前列的时间序列中二分查找
	idx := sort.SearchInts(times, note.HeadTime)
	if idx+1 < len(noteSeqByColumn[k]) {
		return noteSeqByColumn[k][idx+1]
	}
	// 返回默认值
	return Note{Column: 0, HeadTime: 1e9, TailTime: 1e9}
}

// -----End of Helper methods--------

// 辅助函数：整数最小值
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// 辅助函数：边界限制
func clamp(val, min, max int) int {
	if val < min {
		return min
	}
	if val > max {
		return max
	}
	return val
}

// lnBodiesCountSparse 对应Python的LN_bodies_count_sparse_representation
func lnBodiesCountSparse(lnSeq []Note, T int) LNRep {
	diff := make(map[int]float64) // 时间点变化量

	// 第一步：遍历所有长按音符，记录变化量
	for _, note := range lnSeq {
		h := note.HeadTime
		t := note.TailTime
		if t < 0 { // 过滤无效长按
			continue
		}

		// 计算三个关键时间点
		t0 := minInt(h+60, t)
		t1 := minInt(h+120, t)
		diff[t0] += 1.3
		diff[t1] += (-1.3 + 1) // 净变化-0.3
		diff[t] -= 1
	}

	// 第二步：生成有序时间点集合
	pointsSet := make(map[int]struct{})
	pointsSet[0] = struct{}{}
	pointsSet[T] = struct{}{}
	for t := range diff {
		pointsSet[t] = struct{}{}
	}

	// 转换为有序切片
	points := make([]int, 0, len(pointsSet))
	for t := range pointsSet {
		points = append(points, t)
	}
	sort.Ints(points)

	// 第三步：构建分段值和累积和
	values := make([]float64, 0, len(points)-1)
	cumSum := make([]float64, 1, len(points)) // cumSum[0] = 0
	current := 0.0

	for i := 0; i < len(points)-1; i++ {
		t := points[i]
		// 应用当前时间点的变化量
		if delta, exists := diff[t]; exists {
			current += delta
		}

		// 应用变换公式
		val := math.Min(current, 2.5+0.5*current)
		values = append(values, val)

		// 计算区间长度并更新累积和
		segLength := float64(points[i+1] - points[i])
		cumSum = append(cumSum, cumSum[len(cumSum)-1]+segLength*val)
	}

	return LNRep{
		Points: points,
		CumSum: cumSum,
		Values: values,
	}
}

// lnSum 对应Python的LN_sum
func lnSum(a, b int, rep LNRep) float64 {
	points := rep.Points
	cumSum := rep.CumSum
	values := rep.Values

	// 处理空数据情况
	if len(points) == 0 || len(values) == 0 {
		return 0.0
	}

	// 查找区间索引（bisect_right实现）
	i := sort.Search(len(points), func(idx int) bool { return points[idx] > a }) - 1
	j := sort.Search(len(points), func(idx int) bool { return points[idx] > b }) - 1

	// 边界保护
	i = clamp(i, 0, len(values)-1)
	j = clamp(j, 0, len(values)-1)

	var total float64
	if i == j { // 同一区间
		total = float64(b-a) * values[i]
	} else { // 跨区间计算
		// 第一段部分
		total += float64(points[i+1]-a) * values[i]
		// 中间完整区间
		total += cumSum[j] - cumSum[i+1]
		// 最后部分
		total += float64(b-points[j]) * values[j]
	}
	return total
}

// computeJBar 对应Python的compute_Jbar函数
func computeJBar(K int, T int, x float64, noteSeqByColumn [][]Note, baseCorners []float64) (map[int][]float64, []float64) {
	// 初始化数据结构
	Jks := make(map[int][]float64, K)     // 各列J值
	deltaKs := make(map[int][]float64, K) // 各列delta值

	// 辅助函数：jack_nerfer
	jackNerfer := func(delta float64) float64 {
		return 1 - 7e-5*math.Pow(0.15+math.Abs(delta-0.08), -4)
	}

	// 初始化各列存储
	for k := 0; k < K; k++ {
		Jks[k] = make([]float64, len(baseCorners))
		deltaKs[k] = make([]float64, len(baseCorners))
		for i := range deltaKs[k] {
			deltaKs[k][i] = 1e9 // 初始化为大值
		}
	}

	// 第一步：填充各列的J值和delta值
	for k := 0; k < K; k++ {
		notes := noteSeqByColumn[k]
		for i := 0; i < len(notes)-1; i++ {
			start := float64(notes[i].HeadTime)
			end := float64(notes[i+1].HeadTime)

			// 查找区间索引
			leftIdx := sort.SearchFloat64s(baseCorners, start)
			rightIdx := sort.SearchFloat64s(baseCorners, end)
			if leftIdx >= rightIdx {
				continue
			}

			// 计算delta和相关值
			delta := 0.001 * (end - start)
			val := (1 / delta) * math.Pow(delta+0.11*math.Pow(x, 0.25), -1)
			jVal := val * jackNerfer(delta)

			// 填充区间内的所有位置
			for idx := leftIdx; idx < rightIdx; idx++ {
				Jks[k][idx] = jVal
				deltaKs[k][idx] = delta
			}
		}
	}

	// 第二步：对各列进行平滑处理
	JBarKs := make(map[int][]float64, K)
	for k := 0; k < K; k++ {
		JBarKs[k] = smoothOnCorners(baseCorners, Jks[k], 500, 0.001, "sum")
	}

	// 第三步：聚合计算JBar
	JBar := make([]float64, len(baseCorners))
	for i := range baseCorners {
		var numerator, denominator float64

		// 遍历所有列收集数据
		for k := 0; k < K; k++ {
			v := math.Max(JBarKs[k][i], 0)
			w := 1 / deltaKs[k][i]

			numerator += math.Pow(v, 5) * w
			denominator += w
		}

		// 处理分母为零的情况
		if denominator < 1e-9 {
			JBar[i] = 0
		} else {
			JBar[i] = math.Pow(numerator/denominator, 1.0/5)
		}
	}

	return deltaKs, JBar
}

func computeXBar(K int, T int, x float64, noteSeqByColumn [][]Note, activeColumns [][]int, baseCorners []float64) []float64 {
	// 交叉系数矩阵
	crossMatrix := [][]float64{
		{-1},
		{0.075, 0.075},
		{0.125, 0.05, 0.125},
		{0.125, 0.125, 0.125, 0.125},
		{0.175, 0.25, 0.05, 0.25, 0.175},
		{0.175, 0.25, 0.175, 0.175, 0.25, 0.175},
		{0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225},
		{0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225},
		{0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275},
		{0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275},
		{0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325},
	}

	crossCoeff := crossMatrix[K]
	n := len(baseCorners)

	// 初始化存储结构
	XKs := make(map[int][]float64, K+1)
	fastCross := make(map[int][]float64, K+1)
	for k := 0; k <= K; k++ {
		XKs[k] = make([]float64, n)
		fastCross[k] = make([]float64, n)
	}

	// 处理每个键位组合
	for k := 0; k <= K; k++ {
		var notesInPair []Note

		// 合并相邻列音符
		switch {
		case k == 0:
			notesInPair = noteSeqByColumn[0]
		case k == K:
			notesInPair = noteSeqByColumn[K-1]
		default:
			// 手动实现有序合并
			notesInPair = mergeSortedNotes(noteSeqByColumn[k-1], noteSeqByColumn[k])
		}

		// 处理音符对
		for i := 1; i < len(notesInPair); i++ {
			prev := notesInPair[i-1]
			curr := notesInPair[i]
			start := float64(prev.HeadTime)
			end := float64(curr.HeadTime)

			// 查找索引范围
			idxStart := sort.SearchFloat64s(baseCorners, start)
			idxEnd := sort.SearchFloat64s(baseCorners, end)
			if idxStart >= idxEnd {
				continue
			}

			// 计算delta和相关值
			delta := 0.001 * (end - start)
			val := 0.16 * math.Pow(math.Max(x, delta), -2)

			// 检查激活状态
			if (!isActive(k-1, activeColumns[idxStart], activeColumns[idxEnd])) ||
				(!isActive(k, activeColumns[idxStart], activeColumns[idxEnd])) {
				val *= (1 - crossCoeff[k])
			}

			// fmt.Printf("k=%d i=%d start=%.2f end=%.2f idx=%d:%d val=%.6f\n", k, i, start, end, idxStart, idxEnd, val)

			// 填充区间值
			for idx := idxStart; idx < idxEnd; idx++ {
				XKs[k][idx] = val
				fastCrossTerm := 0.4*math.Pow(math.Max(math.Max(delta, 0.06), 0.75*x), -2) - 80
				fastCross[k][idx] = math.Max(fastCrossTerm, 0)
			}
		}
	}

	// 计算XBase
	XBase := make([]float64, n)
	for i := 0; i < n; i++ {
		// 第一求和项
		sum1 := 0.0
		for k := 0; k <= K; k++ {
			sum1 += XKs[k][i] * crossCoeff[k]
		}
		// 第二求和项
		sum2 := 0.0
		for k := 0; k < K; k++ {
			term := math.Sqrt(fastCross[k][i] * crossCoeff[k] *
				fastCross[k+1][i] * crossCoeff[k+1])
			sum2 += term
		}
		XBase[i] = sum1 + sum2
	}
	// 平滑处理
	return smoothOnCorners(baseCorners, XBase, 500, 0.001, "sum")
}

// 辅助函数：判断列是否激活
func isActive(col int, cols ...[]int) bool {
	for _, c := range cols {
		for _, v := range c {
			if v == col {
				return true
			}
		}
	}
	return false
}

// 辅助函数：合并有序音符序列
func mergeSortedNotes(a, b []Note) []Note {
	merged := make([]Note, 0, len(a)+len(b))
	i, j := 0, 0

	for i < len(a) && j < len(b) {
		if a[i].HeadTime <= b[j].HeadTime {
			merged = append(merged, a[i])
			i++
		} else {
			merged = append(merged, b[j])
			j++
		}
	}

	merged = append(merged, a[i:]...)
	merged = append(merged, b[j:]...)
	return merged
}

func computePBar(K int, T int, x float64, noteSeq []Note, lnRep LNRep, anchor []float64, baseCorners []float64) []float64 {
	// 辅助函数：streamBooster
	streamBooster := func(delta float64) float64 {
		ratio := 7.5 / delta
		if ratio > 160 && ratio < 360 {
			return 1 + 1.7e-7*(ratio-160)*math.Pow(ratio-360, 2)
		}
		return 1
	}

	n := len(baseCorners)
	pStep := make([]float64, n)

	for i := 0; i < len(noteSeq)-1; i++ {
		hl := noteSeq[i].HeadTime
		hr := noteSeq[i+1].HeadTime
		deltaTime := hr - hl

		// 处理同时出现的音符
		if deltaTime < 1 {
			spike := 1000 * math.Pow(0.02*(4/x-24), 0.25)
			leftIdx := sort.SearchFloat64s(baseCorners, float64(hl))
			rightIdx := sort.SearchFloat64s(baseCorners, float64(hl)+1e-9) // 模拟side='right'

			for idx := leftIdx; idx < rightIdx && idx < n; idx++ {
				pStep[idx] += spike
			}
			continue
		}

		// 常规情况处理
		leftIdx := sort.SearchFloat64s(baseCorners, float64(hl))
		rightIdx := sort.SearchFloat64s(baseCorners, float64(hr))
		if leftIdx >= rightIdx {
			continue
		}

		delta := 0.001 * float64(deltaTime)
		v := 1 + 6*0.001*lnSum(hl, hr, lnRep)
		bVal := streamBooster(delta)

		// 计算增量
		var inc float64
		if delta < 2*x/3 {
			baseTerm := 0.08 / x * (1 - 24/x*math.Pow(delta-x/2, 2))
			inc = (1 / delta) * math.Pow(baseTerm, 0.25) * math.Max(bVal, v)
		} else {
			baseTerm := 0.08 / x * (1 - 24/x*math.Pow(x/6, 2))
			inc = (1 / delta) * math.Pow(baseTerm, 0.25) * math.Max(bVal, v)
		}

		// 应用增量
		maxInc := math.Max(inc, inc*2-10)
		for idx := leftIdx; idx < rightIdx && idx < n; idx++ {
			pStep[idx] += math.Min(inc*anchor[idx], maxInc)
		}
	}

	return smoothOnCorners(baseCorners, pStep, 500, 0.001, "sum")
}

func computeABar(K int, T int, x float64, noteSeqByColumn [][]Note, activeColumns [][]int, deltaKs map[int][]float64, aCorners []float64, baseCorners []float64) []float64 {
	// 初始化dks存储
	dks := make(map[int][]float64)
	for k := 0; k < K-1; k++ {
		dks[k] = make([]float64, len(baseCorners))
	}

	// 第一步：计算dks值
	for i := range baseCorners {
		cols := activeColumns[i]
		for j := 0; j < len(cols)-1; j++ {
			k0 := cols[j]
			k1 := cols[j+1]
			if k0 >= K-1 { // 确保k0在有效范围
				continue
			}

			delta0 := deltaKs[k0][i]
			delta1 := deltaKs[k1][i]
			diff := math.Abs(delta0 - delta1)
			maxDelta := math.Max(delta0, delta1)
			adjustment := 0.4 * math.Max(maxDelta-0.11, 0)

			dks[k0][i] = diff + adjustment
		}
	}

	// 第二步：计算A_step
	aStep := make([]float64, len(aCorners))
	for i := range aCorners {
		aStep[i] = 1.0
	}

	for i, s := range aCorners {
		// 查找对应的baseCorners索引
		idx := sort.SearchFloat64s(baseCorners, s)
		if idx >= len(baseCorners) {
			idx = len(baseCorners) - 1
		}

		cols := activeColumns[idx]
		for j := 0; j < len(cols)-1; j++ {
			k0 := cols[j]
			k1 := cols[j+1]
			if k0 >= len(dks) || idx >= len(dks[k0]) {
				continue
			}

			dVal := dks[k0][idx]
			delta0 := deltaKs[k0][idx]
			delta1 := deltaKs[k1][idx]
			maxDelta := math.Max(delta0, delta1)

			if dVal < 0.02 {
				term := math.Min(0.75+0.5*maxDelta, 1)
				aStep[i] *= term
			} else if dVal < 0.07 {
				term := math.Min(0.65+5*dVal+0.5*maxDelta, 1)
				aStep[i] *= term
			}
		}
	}

	// 第三步：平滑处理
	return smoothOnCorners(aCorners, aStep, 250, 1.0, "avg")
}

func computeRBar(K int, T int, x float64, noteSeqByColumn [][]Note, tailSeq []Note, baseCorners []float64) []float64 {
	n := len(baseCorners)
	IArr := make([]float64, n)
	RStep := make([]float64, n)

	// 构建按列分组的时间索引
	timesByColumn := make(map[int][]int)
	for colIdx, notes := range noteSeqByColumn {
		var times []int
		for _, note := range notes {
			times = append(times, note.HeadTime)
		}
		timesByColumn[colIdx] = times
	}

	// 计算I列表
	IList := make([]float64, 0, len(tailSeq))
	for _, note := range tailSeq {
		k := note.Column
		hi := note.HeadTime
		ti := note.TailTime

		// 查找下一个音符
		nextNote := findNextNoteInColumn(note, timesByColumn[k], noteSeqByColumn)
		hj := nextNote.HeadTime

		// 计算I_h和I_t
		Ih := 0.001 * math.Abs(float64(ti-hi-80)) / x
		It := 0.001 * math.Abs(float64(hj-ti-80)) / x

		// 计算I值
		denominator := 2 + math.Exp(-5*(Ih-0.75)) + math.Exp(-5*(It-0.75))
		IList = append(IList, 2/denominator)
	}

	// 填充IArr和RStep
	for i := 0; i < len(tailSeq)-1; i++ {
		tStart := tailSeq[i].TailTime
		tEnd := tailSeq[i+1].TailTime

		// 查找索引范围
		leftIdx := sort.SearchFloat64s(baseCorners, float64(tStart))
		rightIdx := sort.SearchFloat64s(baseCorners, float64(tEnd))
		if leftIdx >= rightIdx {
			continue
		}

		// 更新IArr
		for idx := leftIdx; idx < rightIdx && idx < n; idx++ {
			IArr[idx] = 1 + IList[i]
		}

		// 计算R值
		deltaR := 0.001 * float64(tEnd-tStart)
		iListI := IList[i]
		iListI1 := IList[i+1]
		rVal := 0.08 * math.Pow(deltaR, -0.5) / x * (1 + 0.8*(iListI+iListI1))

		// 更新RStep
		for idx := leftIdx; idx < rightIdx && idx < n; idx++ {
			RStep[idx] = rVal
		}
	}

	return smoothOnCorners(baseCorners, RStep, 500, 0.001, "sum")
}

func computeCAndKs(K int, T int, noteSeq []Note, keyUsage map[int][]bool, baseCorners []float64) ([]float64, []float64) {
	// 第一部分：计算C_step
	noteHitTimes := make([]int, 0, len(noteSeq))
	for _, note := range noteSeq {
		noteHitTimes = append(noteHitTimes, note.HeadTime)
	}
	sort.Ints(noteHitTimes)

	cStep := make([]float64, len(baseCorners))
	for i, s := range baseCorners {
		// 转换为整数毫秒时间
		center := int(math.Round(s))
		low := center - 500
		high := center + 500

		// 二分查找边界
		left := sort.SearchInts(noteHitTimes, low)
		right := sort.SearchInts(noteHitTimes, high)
		cStep[i] = float64(right - left)
	}

	// 第二部分：计算Ks_step
	ksStep := make([]float64, len(baseCorners))
	for i := range baseCorners {
		count := 0
		for k := 0; k < K; k++ {
			if keyUsage[k][i] {
				count++
			}
		}
		if count < 1 {
			ksStep[i] = 1
		} else {
			ksStep[i] = float64(count)
		}
	}

	return cStep, ksStep
}

// 辅助函数：计算指定索引范围内的D平均值

func Calculate(filePath, mod string) float64 {
	// lnSeqByColumn is not used
	x, K, T, noteSeq, noteSeqByColumn, lnSeq, tailSeq, _, err := preprocessFile(filePath, mod)
	if err != nil {
		return 0
	}

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

	// === Final Computations ===
	// 准备计算结果数组
	nAll := len(allCorners)
	sAll := make([]float64, nAll)
	tAll := make([]float64, nAll)
	dAll := make([]float64, nAll)

	// 计算S_all, T_all, D_all
	for i := 0; i < nAll; i++ {
		// S_all计算
		aTerm := math.Pow(aBar[i], 3/ksStep[i])
		jTerm := math.Min(jBar[i], 8+0.85*jBar[i])
		sPart1 := 0.4 * math.Pow(aTerm*jTerm, 1.5)

		aTerm2 := math.Pow(aBar[i], 2.0/3)
		pTerm := 0.8*pBar[i] + rBar[i]*35/(cStep[i]+8)
		sPart2 := 0.6 * math.Pow(aTerm2*pTerm, 1.5)

		sAll[i] = math.Pow(sPart1+sPart2, 2.0/3)

		// T_all计算
		tNumerator := math.Pow(aBar[i], 3/ksStep[i]) * xBar[i]
		tDenominator := xBar[i] + sAll[i] + 1
		tAll[i] = tNumerator / tDenominator

		// D_all计算
		dAll[i] = 2.7*math.Sqrt(sAll[i])*math.Pow(tAll[i], 1.5) + sAll[i]*0.27
	}

	// 计算有效权重
	gaps := make([]float64, nAll)
	if nAll > 1 {
		gaps[0] = (allCorners[1] - allCorners[0]) / 2
		gaps[nAll-1] = (allCorners[nAll-1] - allCorners[nAll-2]) / 2
		for i := 1; i < nAll-1; i++ {
			gaps[i] = (allCorners[i+1] - allCorners[i-1]) / 2
		}
	}

	effectiveWeights := make([]float64, nAll)
	for i := range effectiveWeights {
		effectiveWeights[i] = cStep[i] * gaps[i]
	}

	// 创建排序结构体
	type dataPoint struct {
		d         float64
		weight    float64
		origIndex int
	}
	sortedData := make([]dataPoint, nAll)
	for i := 0; i < nAll; i++ {
		sortedData[i] = dataPoint{
			d:         dAll[i],
			weight:    effectiveWeights[i],
			origIndex: i,
		}
	}

	averageD := func(data []dataPoint, indices []int) float64 {
		sum := 0.0
		count := 0
		for _, idx := range indices {
			if idx < len(data) {
				sum += data[idx].d
				count++
			}
		}
		if count == 0 {
			return 0
		}
		return sum / float64(count)
	}

	// 按D值排序
	sort.Slice(sortedData, func(i, j int) bool {
		return sortedData[i].d < sortedData[j].d
	})

	// 计算累积权重
	cumWeights := make([]float64, nAll+1)
	for i := 0; i < nAll; i++ {
		cumWeights[i+1] = cumWeights[i] + sortedData[i].weight
	}
	totalWeight := cumWeights[nAll]
	if totalWeight < 1e-9 {
		totalWeight = 1e-9
	}

	// 计算目标分位数
	targetPercentiles := []float64{0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815}
	indices := make([]int, len(targetPercentiles))
	for i, p := range targetPercentiles {
		target := p * totalWeight
		indices[i] = sort.Search(len(cumWeights), func(j int) bool {
			return cumWeights[j] >= target
		}) - 1
		if indices[i] < 0 {
			indices[i] = 0
		}
	}

	// 计算百分位数平均值
	percentile93 := averageD(sortedData, indices[:4])
	percentile83 := averageD(sortedData, indices[4:8])

	// 计算加权平均值
	var weightedSum, weightSum float64
	for _, dp := range sortedData {
		weightedSum += math.Pow(dp.d, 5) * dp.weight
		weightSum += dp.weight
	}
	weightedMean := math.Pow(weightedSum/weightSum, 1.0/5)

	// 最终SR计算
	SR := 0.88*percentile93*0.25 + 0.94*percentile83*0.2 + weightedMean*0.55

	// 计算音符总数调整
	totalNotes := float64(len(noteSeq))
	for _, note := range lnSeq {
		if note.TailTime > note.HeadTime {
			duration := math.Min(float64(note.TailTime-note.HeadTime), 1000)
			totalNotes += 0.5 * duration / 200
		}
	}
	SR *= totalNotes / (totalNotes + 60)

	// 最终调整
	SR = rescaleHigh(SR)
	SR *= 0.975

	return SR
}
