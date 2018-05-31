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
			int gene = random.nextInt(2);
			genes[i] = gene;
		}
	}

	public double f(double x){
		return Math.sin(x) * ((x-2) * (x-2)) + 3;
	}
	
	public double getFitness() {
		double geneInDouble = geneToDoubles();
		
		return f(geneInDouble);

	
	}
	
	public  double geneToDoubles(){
		int base =1;
		double geneInDouble =10;
		for(int i =0; i < Constants.GENE_LENGTH; i++){
			if(this.genes[i] == 1){
				geneInDouble +=  base;
			}
			
			base  = base *2;
		}
		
		geneInDouble = geneInDouble/102.4f;
		
		return geneInDouble;
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
