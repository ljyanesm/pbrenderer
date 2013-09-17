#pragma once
class LYCell
{
public:
	int cellId;
	int cellStart;
	int numVertex;
	int cellEnd;
	// The end of the cell is defined by cellStart+numVertex

	LYCell(void);
	~LYCell(void);
};

