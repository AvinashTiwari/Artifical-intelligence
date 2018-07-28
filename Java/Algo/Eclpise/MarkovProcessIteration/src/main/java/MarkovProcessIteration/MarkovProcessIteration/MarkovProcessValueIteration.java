package MarkovProcessIteration.MarkovProcessIteration;

public class MarkovProcessValueIteration {
	
	private double v[][];
	private double vNext[][];
	private double r[][];
	private char pi[][];
	private double delta =0;
	private int n;
	
	public MarkovProcessValueIteration()
	{
		pi = new char[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLS];
		v = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLS];
		vNext = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLS];
		r = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLS];

		initVraibale();
	}
	
	
	
	private void initVraibale() {
		// TODO Auto-generated method stub
		
		for(int rowIndex =0 ; rowIndex < MarkovProcessConstant.NUM_ROWS;rowIndex++ ){
			for(int columnIndex = 0; columnIndex < MarkovProcessConstant.NUM_COLS; columnIndex++){
				r[rowIndex][columnIndex] = MarkovProcessConstant.STATE_REWARD;
			}
		}
		
		r[0][3] =1;
		r[1][3] =-1;
		r[1][1] = 0;
	}



	public void run(){
		
	}

}
