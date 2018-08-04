package QLearning.QLearning;

public class App  {  
 
    public static void main(String[] args) {
    	QLearning algorithm = new QLearning();
    	algorithm.run();
    	algorithm.showResult();
    	algorithm.showPolicy();
	}
}
