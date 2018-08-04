package QLearning.QLearning;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class QLearning {

	// this R[][] stores the reward for every state (and action)
	private double[][] R = {
			{Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE},
			{Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.EXIT_REWARD},
			{Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.MIN_VALUE},
			{Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE},
			{Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.EXIT_REWARD},
			{Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.MIN_VALUE,Constants.MIN_VALUE,Constants.STANDARD_REWARD,Constants.EXIT_REWARD},
	};
	
	// this is the Q(s,a) function
    private double[][] Q;
    private Random random;
    
    public QLearning() {
    	this.random = new Random();
    	// Q values for going from s to s'
    	this.Q = new double[Constants.NUM_OF_STATES][Constants.NUM_OF_STATES];
	}
    
    public void run() {
    	
    	// episode: a full iteration when the agent starts from a random state and finds the terminal state
    	for(int epsisodeCounter=0;epsisodeCounter<Constants.NUM_OF_EPISODES;++epsisodeCounter) {
    		int state = random.nextInt(Constants.NUM_OF_STATES);
    		// we do not want to start with the terminal state
    		if(state==5) continue;
    		simulate(state);
    	}
    }

	private void simulate(int state) {
		
		// a single episode: the agent finds a path from state s to the terminal state
		
		do {
			
			// get available actions (so available next states)
			List<Integer> possibleNextStates = availableStates(state);
			
			// choose a random next state
			int nextState = possibleNextStates.get(random.nextInt(possibleNextStates.size()));
			
			// the max Q value concerning the next state
			double maxQ = findMaxQ(nextState);
			
			// Q learning equation: Q[s][a] = Q[s][a] + alpha ( R[s][a] + gamma (max Q[s'][a']) - Q[s][a] )
			Q[state][nextState] = Q[state][nextState] + Constants.ALPHA*(R[state][nextState]+Constants.GAMMA*maxQ-Q[state][nextState]);
 			
			// consider the next state: the agent considers the next state until it reaches the terminal one
 			state = nextState;
			
		} while( state != 5);
	}
	
	// finding the max Q value for the next state
	private double findMaxQ(int nextState) {
		
		double maxQ = Constants.MIN_VALUE;
		
		// thats why we use Q[][] it is easy to find the max Q value with
		// a simple linerat search O(N)
		for(int i=0;i<this.Q.length;++i) {
			if( this.Q[nextState][i] > maxQ)
				maxQ = this.Q[nextState][i];
		}
		
		return maxQ;
	}
	
	private List<Integer> availableStates(int state) {
		
		List<Integer> possibleNextStates = new ArrayList<>();
		
		// get the available states: the R[][] matrix given row contains all the
		// possible next states !!!
		for(int colIndex=0;colIndex<this.R.length;++colIndex) {
			if( this.R[state][colIndex] > Constants.MIN_VALUE )  {
				possibleNextStates.add(colIndex);
			}
		}
		
		return possibleNextStates;
	}

	public void showResult() {
		
		for(int i=0;i<this.Q.length;++i) {
			for(int j=0;j<this.Q.length;++j) {
				System.out.printf("%.1f ", this.Q[i][j]);
			}
			System.out.println();
		}
	}
	
	public void showPolicy() {
		
		// we consider every single state as a starting state
		// until we find the terminal state: we go in the direction
		// of the max Q value (this is why we calculate the Q values)
		
		for(int i=0;i<Constants.NUM_OF_STATES;i++) {
			
			int state = i;
			System.out.print("Policy: " + state);
			
			while( state!= 5 ) {
				
				int maxQState = 0;
				double maxQ = 0;
				
				for(int j=0;j<Constants.NUM_OF_STATES;j++) {
					if( Q[state][j] > maxQ ) {
						maxQ = Q[state][j];
						maxQState = j;
					}
				}
				
				System.out.print(" -> " + maxQState);
				state = maxQState;
			}
			System.out.println();
		}
	}
}