package TabuSearchI.TabuSearchI;

public class CostFunction {

	//we can define the function we want to optimize (find minimum or maximum)
	public static double f(double x, double y) {
		return Math.exp(-x*x-y*y)*Math.sin(x);
	}
}
