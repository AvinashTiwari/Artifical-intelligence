package learn.Avinash.JavaAI;

public class BruteForce {
	private static final double START_X = -1;
	private static final double END_X = 2;

	public double f(double x) {
		return -1 * (x - 1) * (x - 1) + 2;
	}
	
	public void bruteForceSearch(){
		double startPointX = START_X;
		double max = f(startPointX);
		double dx = 0.01;
		double maxX = START_X;
		
		for(double i= startPointX; i < END_X; i+=dx){
			if(f(i) > max){
				max =f(i);
				maxX = i;
			}
		}
		
		System.out.println("The Minium value f(x) = " + max + " and x " + maxX);
	}
}
