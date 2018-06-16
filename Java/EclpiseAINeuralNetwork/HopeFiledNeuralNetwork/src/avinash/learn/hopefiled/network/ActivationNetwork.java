package avinash.learn.hopefiled.network;

public class ActivationNetwork {

	public static int stepFunction(double x){
		if(x >= 0){
			return 1;
		}
		
		return -1;
	}
}
