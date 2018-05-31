package learn.Avinash.JavaAI;

import java.util.Random;

public class Indiviual {
	private int[] genes;
	private int fitness;
	private Random random;

	public Indiviual() {
		this.genes = new int[Constants.CHROMOZOME_LENGTH];
		this.random = new Random();
	}

	public void genrateIndiviual() {
		for (int i = 0; i < Constants.CHROMOZOME_LENGTH; ++i) {
			int gene = random.nextInt(10);
			genes[i] = gene;
		}
	}

	public int getFitness() {
		if (fitness == 0) {
			for (int i = 0; i < Constants.CHROMOZOME_LENGTH; ++i) {
				if (getGene(i) == Constants.SOLUTION_SEQUENCE[i]) {
					this.fitness++;
				}
			}

		}

		return fitness;
	}

	public int getGene(int i) {
		// TODO Auto-generated method stub
		return this.genes[i];
	}
	
	public void setGene(int index, int givenvalue){
		this.genes[index] = givenvalue;
		this.fitness = 0;
	}

	@Override
	public String toString() {
		String s = "";
		
		for(int i=0 ; i < Constants.CHROMOZOME_LENGTH; i++){
			s +=getGene(i);
		}
		
		return s;
	}
	
	
}
