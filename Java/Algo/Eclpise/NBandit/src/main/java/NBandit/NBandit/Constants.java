package NBandit.NBandit;

public class Constants {

	private Constants() {
		
	}
	
	// epsilon greedy strategy the agent visit new states with epsilon probability
	public final static double EPSILON = 0.1;
	// number of bandits
	public final static int NUM_OF_BANDITS = 3;
	// number of iterations
	public final static int NUM_OF_EPISODES = 10000;
}
