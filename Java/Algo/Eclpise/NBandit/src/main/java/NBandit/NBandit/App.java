package NBandit.NBandit;

public class App {

	public static void main(String[] args) {
		
		NArmedBanditProblem banditProblem = new NArmedBanditProblem();
		banditProblem.run();
		banditProblem.showStatistics();
	}
}
