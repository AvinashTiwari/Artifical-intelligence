package MarkovProcessIteration.MarkovProcessIteration;

public class MarkovProcessValueIteration {
	
	//this is the V(s) value function
    private double v[][]; 
    //V'(s) used in updates
    private double vNext[][];
    //we have to store the rewards
    private double r[][]; 
    //we have to store the policy itself
    private char pi[][];
    //we track the error delta=|V(t+1)-V(t)|
    private double delta = 0;
    private int n;
    
    public MarkovProcessValueIteration() {
    	//policy: in the end we want to have the characters - where to go U(up),D(down),L(left),R(right)
    	pi = new char[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLUMNS]; 
    	//the V value function values
    	v = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLUMNS];
    	//initialize V'
        vNext = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLUMNS];
        //reward function R(s',s)
        r = new double[MarkovProcessConstant.NUM_ROWS][MarkovProcessConstant.NUM_COLUMNS];

        initializeVariables();
	}
     
    private void initializeVariables() {
		//we just have to initialize the R(s',s) rewards 
    	for (int rowIndex=0; rowIndex<MarkovProcessConstant.NUM_ROWS; rowIndex++) {
            for (int columnIndex=0; columnIndex<MarkovProcessConstant.NUM_COLUMNS; columnIndex++) {
                r[rowIndex][columnIndex] = MarkovProcessConstant.STATE_REWARD;
            }
        }
    	
    	//initialize the +1 and -1 states + we have an unreachable state
    	r[0][3] =+100; 
        r[1][3] =-100; 
        r[1][1] =0;
	}

	public void run() {
    	
        do {
            copyArray(vNext, v);
            n++;
            delta = 0;
            
            for (int rowIndex=0; rowIndex<MarkovProcessConstant.NUM_ROWS; rowIndex++) {
                for (int columnIndex=0; columnIndex<MarkovProcessConstant.NUM_COLUMNS; columnIndex++) {
                	//manipulates vNext
                    update(rowIndex, columnIndex);
                    double error = Math.abs(vNext[rowIndex][columnIndex] - v[rowIndex][columnIndex]);
                   
                    //we make sure every state s it will converge |V(s)-V'(s)|<delta
                    //we just track the maximum error (as many error terms as the num of states)
                    //if the max error is smaller than epsilon -> the algorithm has converged !!!
                    if (error > delta)
                        delta = error;
                }
            }
        } while (delta > MarkovProcessConstant.EPLSION && n < MarkovProcessConstant.NUMER_OF_ITERATION);
         
        //the error is small: we have found the approximated V*(s) and pi*(s) functions
        printResults();
    }
     
    private void printResults() {
    	
    	//display the V(s) value-function values
        System.out.println("The V(s) values after " + n + " iterations:\n");
        for (int rowIndex=0; rowIndex<MarkovProcessConstant.NUM_ROWS; rowIndex++) {
            for (int columnIndex=0; columnIndex<MarkovProcessConstant.NUM_COLUMNS; columnIndex++) {
                System.out.printf("% 6.5f\t", v[rowIndex][columnIndex]);
            }
            System.out.print("\n");
        }
        
        pi[0][3]='+';
        pi[1][3]='-';
        pi[1][1]='@';
        
        //display the pi(s) policy-function: prints out what action to do in every state
        System.out.println("\nBest policy:\n");
        for (int rowIndex=0; rowIndex<MarkovProcessConstant.NUM_ROWS; rowIndex++) {
            for (int columnIndex=0; columnIndex<MarkovProcessConstant.NUM_COLUMNS; columnIndex++) {
                System.out.print(pi[rowIndex][columnIndex] + "   ");
            }
            System.out.print("\n");
        }    
	}

	public void update(int row, int col) {
         
        double actions[] = new double[4]; 
     
        //+1,-1 or obstacle state - use that value
        if ((row==0 && col==3) || (row==1 && col==3) || (row==1 && col==1)) {
            vNext[row][col] = r[row][col];
        } else {
        	//we calculate the P(s'|s,a)*V(s') values
        	actions[0] = MarkovProcessConstant.ACTION_PROB*goUp(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goLeft(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goRight(row,col);
            actions[1] = MarkovProcessConstant.ACTION_PROB*goDown(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goLeft(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goRight(row,col);
            actions[2] = MarkovProcessConstant.ACTION_PROB*goLeft(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goDown(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goUp(row,col);
            actions[3] = MarkovProcessConstant.ACTION_PROB*goRight(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goDown(row,col) + MarkovProcessConstant.ACTION_MISS_PROB*goUp(row,col);
            
            //we want to find the optimal action so finding the max
            int best = findMaxIndex(actions);
             
            //this is how we calculate V with Bellman-equation - V(s)=R(s)+max[gamma*sum(P(s'|s,a)*V(s'))]
            vNext[row][col] = r[row][col] + MarkovProcessConstant.GAMMA * actions[best];
             
            //update policy (argmax implementation)
            switch(best) {
            	case 0: 
            		pi[row][col]='U';
            		break;
            	case 1:
            		pi[row][col]='D';
            		break;
            	case 2:
            		pi[row][col]='L';
            		break;
            	case 3:
            		pi[row][col]='R';
            		break;
            }
        }
    }
     
	//returns the index of the optimal action
    public int findMaxIndex(double actions[]) {
        
    	int maxIndex=0;
        
        for (int i=1; i<actions.length; i++)
        	if( actions[i] > actions[maxIndex])
        		maxIndex = i;
        
        return maxIndex;
    }
     
    public double goUp(int row, int col) {
        //check whether it is possible to go up
    	//if can not: stay at the same state
        if ((row==0) || (row==2 && col==1))
            return v[row][col];
        
        //go up if it is possible
        return v[row-1][col];
    }
 
    public double goDown(int row, int col) {
    	//check whether it is possible to go down
    	//if can not: stay at the same state
        if ((row==MarkovProcessConstant.NUM_ROWS-1) || (row==0 && col==1))
            return v[row][col];
        
        //go down if it is possible
        return v[row+1][col];
    }
 
    public double goLeft(int row, int col) {
    	//check whether it is possible to go left
    	//if can not: stay at the same state
        if ((col==0) || (row==1 && col==2))
            return v[row][col];
        
        //go left if it is possible
        return v[row][col-1];
    }
 
    public double goRight(int row, int col) {
    	//check whether it is possible to go right
    	//if can not: stay at the same state
        if ((col==MarkovProcessConstant.NUM_COLUMNS-1) || (row==1 && col==0))
            return v[row][col];
        
        //go right if it is possible
        return v[row][col+1];
    }
     
    public void copyArray(double[][] sourceArray, double[][] destionationArray) {
        for (int i=0; i<sourceArray.length; i++) {
            for (int j=0; j<sourceArray[i].length; j++) {
                destionationArray[i][j] = sourceArray[i][j];
            }
        }
    }

}
