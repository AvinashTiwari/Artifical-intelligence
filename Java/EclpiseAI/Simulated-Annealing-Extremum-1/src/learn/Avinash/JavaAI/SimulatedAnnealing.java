package learn.Avinash.JavaAI;

import java.util.Random;

public class SimulatedAnnealing {
	private Random randomGenrator;
	private double current_coordinate_X;
	private double next_coordinate_X;
	private double best_coordinate_X;

	public void findOptinum(){
		double temperature = Constants.MAX_TEMPURATURE;
		while(temperature > Constants.MIN_TEMPURATURE){
			next_coordinate_X = getRandomX();
			double actualEnergy = getEngery(current_coordinate_X);
			double newEnery = getEngery(next_coordinate_X);
			if(acceptanceProbablity(actualEnergy,newEnery, temperature) > Math.random()){
				current_coordinate_X = next_coordinate_X;
			}
			
			if(f(current_coordinate_X) < f(best_coordinate_X))
			{
				best_coordinate_X = current_coordinate_X;
			}
			
			temperature *= 1 - Constants.COOLING_RATE; 
		}
		
		System.out.println( best_coordinate_X  + " " + f(best_coordinate_X));
	}
	
	private double getRandomX() {
		// TODO Auto-generated method stub
		return this.randomGenrator.nextDouble()*(Constants.MAX_COORDINATE- Constants.MIN_COORDINATE ) - Constants.MIN_COORDINATE;
	}

	public SimulatedAnnealing() {
		this.randomGenrator = new Random();
	}

	public double getEngery(double x) {
		return f(x);
	}

	public double f(double x) {
		return (x - 0.3) * (x - 0.3) * (x - 0.3) - 5 * x + x - 2;
	}

	public double acceptanceProbablity(double engery, double newengery, double temprature) {
		if (newengery < engery) {
			return 1;
		}
		return Math.exp((engery-newengery)/temprature);
	}

}
