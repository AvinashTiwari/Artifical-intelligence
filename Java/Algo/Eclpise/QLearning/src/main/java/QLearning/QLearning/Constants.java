package QLearning.QLearning;

public class Constants {

	private Constants() {

	}

	public static final double MIN_VALUE = -1e5;
	// number of states
	public static final int NUM_OF_STATES = 6;
	// reward in non-terminal states (used to initialize R[][])
	public static final double STANDARD_REWARD = -0.1;
	public static final double EXIT_REWARD = 100;
	// gamma discount factor: how to deal with future rewards [0,1]
	public static final double GAMMA = 0.9;
	// learning rate
	public static final double ALPHA = 0.1;
	// number of iterations
	public static final int NUM_OF_EPISODES = 100000;
    
}
