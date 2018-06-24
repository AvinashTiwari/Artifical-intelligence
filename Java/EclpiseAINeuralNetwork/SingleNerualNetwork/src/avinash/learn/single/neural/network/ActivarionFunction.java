package avinash.learn.single.neural.network;

public class ActivarionFunction {

	public static int stepFunction(float activation){
		if(activation >= 1)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	} 
}
