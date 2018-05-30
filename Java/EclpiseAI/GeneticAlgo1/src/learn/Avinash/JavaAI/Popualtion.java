package learn.Avinash.JavaAI;

public class Popualtion {
	private Indiviual indiviuals[];
	
	public Popualtion(int populationSize){
		indiviuals = new Indiviual[populationSize];
	}
  
	public void initialize(){
		for(int i =0; i < indiviuals.length; ++i ){
			Indiviual newIndiviual = new Indiviual();
			newIndiviual.genrateIndiviual();
			saveIndivual(i, newIndiviual);
			
		}
	}

	public Indiviual getIndiviual(int index){
		return this.indiviuals[index];
	} 
	
	public Indiviual getFitnessIndivual(){
		Indiviual fitness = indiviuals[0];
		for(int i=1; i < indiviuals.length; ++i){
			if(getIndiviual(i).getFitness()>= fitness.getFitness()){
				fitness = getIndiviual(i);
			}
		}
		
		return fitness;
	}
	
	public int size(){
		return this.indiviuals.length;
	}
	
	public void saveIndivual(int i, Indiviual newIndiviual) {
 
		this.indiviuals[i] = newIndiviual;
	}
}
