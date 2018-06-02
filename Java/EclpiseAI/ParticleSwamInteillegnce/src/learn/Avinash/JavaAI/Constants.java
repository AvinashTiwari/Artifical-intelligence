package learn.Avinash.JavaAI;

public class Constants {
	private Constants() {
	}
	
	public final static int NUM_DIMENSIONS = 2; 
	public final static int NUM_PARTICALS = 5;
	public final static int MAX_ITERATIONS = 1000;
	public final static double MIN = -10.0; 
	public final static double MAX = 10.0;
	public final static double w = 0.729; // inertia weight
	public final static double c1 = 1.49445; // cognitive/local weight
	public final static double c2 = 1.49445; // social/global weight
	
	public static double f(double[] data){
		return Math.exp((data[0]*data[0] + data[1]*data[1]));
	}

}
