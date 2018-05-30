package learn.Avinash.JavaAI;

import java.util.Random;

public class GeneticAlgo {
	private Random randomGenrator;

	public GeneticAlgo() {
		this.randomGenrator = new Random();
	}

	public Popualtion evolvePopulation(Popualtion popualtion) {
		Popualtion newPopualtion = new Popualtion(popualtion.size());
		for (int i = 0; i < popualtion.size(); i++) {
			Indiviual firstIndiviual = randomSelection(popualtion);
			Indiviual secondirstIndiviual = randomSelection(popualtion);
			Indiviual newIndiviual = crossOver(firstIndiviual, secondirstIndiviual);
			newPopualtion.saveIndivual(i, newIndiviual);

		}

		for (int i = 0; i < newPopualtion.size(); i++) {
			mutate(newPopualtion.getIndiviual(i));
		}

		return newPopualtion;
	}

	private void mutate(Indiviual indiviual) {
		for (int i = 0; i < Constants.CHROMOZOME_LENGTH; i++) {
			if (Math.random() <= Constants.MUTATION_RATE) {
				int gene = randomGenrator.nextInt(10);
				indiviual.setGene(i, gene);
			}
		}
	}

	private Indiviual crossOver(Indiviual firstIndiviual, Indiviual secondirstIndiviual) {
		Indiviual newSolution = new Indiviual();
		for (int i = 0; i < Constants.CHROMOZOME_LENGTH; ++i) {
			if (Math.random() <= Constants.CROSS_OVER_RATE) {
				newSolution.setGene(i, firstIndiviual.getGene(i));
			} else {
				newSolution.setGene(i, secondirstIndiviual.getGene(i));

			}
		}
		return newSolution;
	}

	private Indiviual randomSelection(Popualtion popualtion) {

		Popualtion newPopualtion = new Popualtion(Constants.TOURNAMENT_SIZE);
		for (int i = 0; i < Constants.TOURNAMENT_SIZE; i++) {
			int randomIndex = (int) (Math.random() * popualtion.size());
			newPopualtion.saveIndivual(i, popualtion.getIndiviual(randomIndex));
		}

		Indiviual fittestIndivual = newPopualtion.getFitnessIndivual();

		return fittestIndivual;
	}

}
