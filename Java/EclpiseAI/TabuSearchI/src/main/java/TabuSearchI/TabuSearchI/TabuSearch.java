package TabuSearchI.TabuSearchI;

import java.util.List;

public class TabuSearch {

	private State[][] states;
	private TabuList tabuList;
	private NeighborSolutionHandler neighborSolutionHandler;
	
	public TabuSearch(State[][] states) {
		this.states = states;
		this.tabuList = new TabuList();
		this.neighborSolutionHandler = new NeighborSolutionHandler();
	}
	
	public State solve(State initialSolution) {
		
		State bestState = initialSolution;
		State currentState = initialSolution;
		
		int iterationCounter = 0;
		
		//we make a predefined number of iterations
		while(iterationCounter<Constants.NUM_ITERATIONS) {
			
			//get all the available (reachable) states in the neighborhood
			List<State> candidateNeighbors = currentState.getNeighbors();
			//get the tabu list
			List<State> solutionsTabu = tabuList.getTabuItems();
			
			//get the best neighbor (lowest f(x) value) AND make sure it is not in the tabu list
			State bestNeighborFound = neighborSolutionHandler.getBestNeighbor(states, candidateNeighbors, solutionsTabu);
			
			//we are looking for a minimum in this case
			if (bestNeighborFound.getZ()<bestState.getZ()) {
				bestState = bestNeighborFound;
			}
			
			//we add it to the tabu list because we considered this item
			tabuList.add(currentState);
			
			//hop to the next state
			currentState = bestNeighborFound;
			
			iterationCounter++;
		}
		
		//solution of the algorithm
		return bestState;
	}
}
