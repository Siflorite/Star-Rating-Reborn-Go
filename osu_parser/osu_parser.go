package osu_parser

import (
	"bufio"
	"os"
	"strconv"
	"strings"
)

func stringToInt(s string) int {
	f, _ := strconv.ParseFloat(s, 64)
	return int(f)
}

type Parser struct {
	filePath     string
	od           float64
	columnCount  int
	columns      []int
	noteStarts   []int
	noteEnds     []int
	noteTypes    []int
	noteStartsLn []int
}

func NewParser(filePath string) *Parser {
	return &Parser{
		filePath:    filePath,
		od:          -1,
		columnCount: -1,
	}
}

func (p *Parser) Process() error {
	file, err := os.Open(p.filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		p.readMetadata(scanner, line)
		p.readOverallDifficulty(line)
		p.readColumnCount(line)
		if p.columnCount != -1 {
			p.readNote(scanner, line)
		}
	}
	return scanner.Err()
}

func (p *Parser) readMetadata(scanner *bufio.Scanner, line string) {
	if strings.Contains(line, "[Metadata]") {
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "Source:") {
				break
			}
		}
	}
}

func (p *Parser) readOverallDifficulty(line string) {
	if strings.Contains(line, "OverallDifficulty:") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) > 1 {
			p.od, _ = strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		}
	}
}

func (p *Parser) readColumnCount(line string) {
	if strings.Contains(line, "CircleSize:") {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) > 1 {
			value := strings.TrimSpace(parts[1])
			if value == "0" {
				value = "10"
			}
			p.columnCount = stringToInt(value)
		}
	}
}

func (p *Parser) readNote(scanner *bufio.Scanner, line string) {
	if strings.Contains(line, "[HitObjects]") {
		for scanner.Scan() {
			objectLine := scanner.Text()
			p.parseHitObject(objectLine)
		}
	}
}

func (p *Parser) parseHitObject(objectLine string) {
	params := strings.Split(objectLine, ",")
	if len(params) < 6 {
		return
	}

	// Parse column
	xPos, _ := strconv.Atoi(params[0])
	columnWidth := 512 / p.columnCount
	column := xPos / columnWidth
	p.columns = append(p.columns, column)

	// Parse note start time
	noteStart, _ := strconv.Atoi(params[2])
	p.noteStarts = append(p.noteStarts, noteStart)

	// Parse note type
	noteType, _ := strconv.Atoi(params[3])
	p.noteTypes = append(p.noteTypes, noteType)

	// Parse note end time
	endParts := strings.Split(params[5], ":")
	noteEnd, _ := strconv.Atoi(endParts[0])
	p.noteEnds = append(p.noteEnds, noteEnd)
}

func (p *Parser) GetParsedData() (int, []int, []int, []int, []int, float64) {
	return p.columnCount,
		p.columns,
		p.noteStarts,
		p.noteEnds,
		p.noteTypes,
		p.od
}
