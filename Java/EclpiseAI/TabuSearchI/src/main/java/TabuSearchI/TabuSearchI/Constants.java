package TabuSearchI.TabuSearchI;

public class Constants {

	private Constants() {
		
	}
	
	//because the interval is [-10:10] with 0.1 steps
	public static final int NUM_VALUES = 200;
	//number of iterations in the search
	public static final int NUM_ITERATIONS = 100000;
	//tabu tenure (size of the tabu list or queue)
	public static final int TABU_TENURE = 400;
}
