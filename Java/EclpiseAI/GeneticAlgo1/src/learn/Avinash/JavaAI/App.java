package learn.Avinash.JavaAI;

public class App {

	public static void main(String[] args) {
		GeneticAlgo genAlgo = new GeneticAlgo();

		Popualtion population = new Popualtion(100);
		population.initialize();

		int genrationCounter = 0;
		while (population.getFitnessIndivual().getFitness() != Constants.MAX_FITNESS) {
			++genrationCounter;
			System.out.println(
					"Generator : " + genrationCounter + " fittest is" + population.getFitnessIndivual().getFitness());
		System.out.println(population.getFitnessIndivual() + " \n" );
		population = genAlgo.evolvePopulation(population);
		}
		
		System.out.println("Solution found !! " + population.getFitnessIndivual());
	}

}
