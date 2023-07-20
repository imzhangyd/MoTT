/**
 * This is a simple implementation of the Hungarian/Munkres-Kuhn algorithm for the rectangulat assignment problem
 *  
 * @version February 3, 2012
 * 
 * @author Nicolas Chenouard
 *
 */

public class HungarianMatching
{
	final double[][] costs;
	boolean[][] starMat;
	boolean[][] primeMat;

	boolean[] columnCoverage;
	boolean[] rowCoverage;
	final int numRows;
	final int numColumns;
	boolean feasibleAssociation;

	/**
	 * * Create the optimizer
	 * @param costs table of assignment costs. The number of lines has to be less are equal to the number of columns (i.e., costs.length <= costs[0]. length == true). 
	 */
	public HungarianMatching(double[][] costs)
	{
		// subtract the smallest value of each row to the row
		this.costs = costs;
		numRows = costs.length;
		numColumns = costs[0].length;

		primeMat = new boolean[numRows][numColumns];
		starMat = new boolean[numRows][numColumns];
		rowCoverage = new boolean[numRows];
		columnCoverage = new boolean[numColumns];
		for (int i = 0; i < numRows; i++)
		{
			double minCost = costs[i][0];
			for (int j = 1; j < numColumns; j ++)
				if (costs[i][j] < minCost)
					minCost = costs[i][j];
			for (int j = 0; j < numColumns; j ++)
				costs[i][j]-=minCost;
		}                                    
		feasibleAssociation = false;
	}

	int numStep;
	int step4_row;
	int step4_col;
	/**
	 * Build the optimal assignment
	 * 
	 * @return a table indicating in each row element is assigned to each column element
	 * */
	public boolean[][] optimize() throws Exception
	{
		step1();
		numStep = 2;
		while(!feasibleAssociation)
		{
			switch (numStep) {
			case 2:
				step2();
				break;
			case 3:
				step3();
				break;
			case 4:
				step4(step4_row, step4_col);
				break;
			case 5:
				step5();
				break;
			}
		}
		
		return starMat;
	}

	private void step1() throws Exception
	{
		for (int i = 0; i<numRows; i++)
		{
			for (int j = 0; j< numColumns; j++)
			{
				if (costs[i][j]==0)
				{
					if (!columnCoverage[j])
					{
						starMat[i][j] = true;
						columnCoverage[j] = true;
						break;
					}
				}
			}
		}
	}

	private void step2() throws Exception
	{
		int cntColumnCoverage = 0;
		for (int j = 0; j< numColumns; j++)
		{
			for (int i = 0; i<numRows; i++)
			{
				if (starMat[i][j])
				{
					columnCoverage[j] = true;
					cntColumnCoverage++;
					break;
				}
			}
		}
		feasibleAssociation = (cntColumnCoverage==numRows);
		numStep = 3;
	}

	private void step3() throws Exception
	{
		boolean zerosFound = true;
		while (zerosFound)
		{
			zerosFound = false;
			for (int j = 0; j<numColumns; j++)
			{
				if (!columnCoverage[j])
				{
					for (int i = 0; i < numRows; i++)
					{
						if ((!rowCoverage[i]) && (costs[i][j]==0))
						{
							primeMat[i][j] = true;
							boolean foundStarCol = false;
							for (int j2 = 0; j2 < numColumns; j2++)
							{
								if (starMat[i][j2])
								{
									foundStarCol = true;
									columnCoverage[j2] = false;
									break;
								}
							}
							if (!foundStarCol)
							{
								step4_col = j;
								step4_row = i;
								numStep = 4;
								return;
							}
							else
							{
								rowCoverage[i] = true;
								zerosFound = true;
								break; // go to next column
							}
						}
					}
				}
			}
		}
		numStep = 5;
	}

	private void step4(int row, int col) throws Exception
	{
		boolean[][] starMat2 = new boolean[numRows][numColumns];
		for (int i = 0; i < numRows; i++)
			System.arraycopy(starMat[i], 0, starMat2[i], 0, numColumns);
		starMat2[row][col] = true;

		int starCol = col;
		int starRow = -1;
		for (int i = 0; i < numRows; i++)
		{
			if (starMat[i][starCol])
			{
				starRow = i;
				break; // there is only one starred zero per column
			}
		}
		while (starRow >= 0)
		{
			// unstar the starred zero
			starMat2[starRow][starCol] = false;
			// find a starred prime
			int primeRow = starRow;
			int primeCol = -1;
			for (int j = 0; j < numColumns; j++)
			{
				if (primeMat[primeRow][j])
				{
					primeCol = j;
					break;
				}
			}
			// star the primed zero
			starMat2[primeRow][primeCol] = true;
			// find a starred zero in the column
			starCol = primeCol;
			starRow = -1;
			for (int i = 0; i < numRows; i++)
			{
				if (starMat[i][starCol])
				{
					starRow = i;
					break; // there is only one starred zero per column
				}
			}
		}
		// update star matrix
		starMat = starMat2;
		// clear prime matrix and coverred rows		
		for (int i = 0; i<numRows; i++)
		{
			for (int j = 0; j< numColumns; j++)
			{
				primeMat[i][j] = false;
			}
			rowCoverage[i] = false;
		}		
		numStep = 2;
	}


	private void step5() throws Exception
	{
		// find the smallest uncovered element
		double minUncoveredCost = Double.MAX_VALUE;
		for (int j = 0; j< numColumns; j++)
		{
			if (!columnCoverage[j])
				for (int i = 0; i<numRows; i++)
				{
					if (!rowCoverage[i])
					{
						if (minUncoveredCost > costs[i][j])
							minUncoveredCost = costs[i][j];
					}
				}
		}
		// add the min cost to each covered row
		for (int i = 0; i<numRows; i++)
			if (rowCoverage[i])
				for (int j = 0; j< numColumns; j++)
					costs[i][j]+=minUncoveredCost;
		// subtract the min cost to each uncovered column
		for (int j = 0; j< numColumns; j++)
			if (!columnCoverage[j])
				for (int i = 0; i<numRows; i++)
					costs[i][j] -= minUncoveredCost;
		numStep = 3;
	}
}