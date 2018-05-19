package learn.Avinash.JavaAI;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class StochasticSearch {
	Random random = new Random();
	private static final double START_X = -1;
	private static final double END_X = 2;


	private double f(double x) {
		return (x - 1) * (x - 1);
	}

	public void stochasticSearch() {
		
		double startpointX = START_X;
		double min = f(startpointX);
		double minX =  START_X;
		
		for(int i = 0 ; i < 100000; i++){
			double RandomX = ThreadLocalRandom.current().nextDouble(START_X, END_X);
			
			if(f(RandomX) < min){
				min = f(RandomX);
				minX = RandomX;
			}
		}
		
		System.out.println("The minium is f(x) = " +min  + " x point " + minX );
	}

}
