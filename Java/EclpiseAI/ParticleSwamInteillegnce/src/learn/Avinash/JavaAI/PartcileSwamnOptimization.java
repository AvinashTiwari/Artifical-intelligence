package learn.Avinash.JavaAI;

public class PartcileSwamnOptimization {
	private double[] globalBestSolution;
	private Particle[] particleSwarm;
	private int epochs;

	public PartcileSwamnOptimization() {
		this.globalBestSolution = new double[Constants.NUM_DIMENSIONS];
		this.particleSwarm = new Particle[Constants.NUM_PARTICALS];
		generateRandomSolution();
	}

	public void solve() {

		for (int i = 0; i < Constants.NUM_PARTICALS; ++i) {

			double[] x = initializeLocation();
			double[] v = initializeVelocity();

			this.particleSwarm[i] = new Particle(x, v);
			this.particleSwarm[i].checkBestSolution(this.globalBestSolution);
		}

		while (this.epochs < Constants.MAX_ITERATIONS) {

			for (Particle actualParticle : this.particleSwarm) {

				for (int i = 0; i < actualParticle.getVelocity().length; ++i) {

					double rp = Math.random();
					double rg = Math.random();

					actualParticle.getVelocity()[i] = Constants.w * actualParticle.getVelocity()[i]
							+ Constants.c1 * rp * (actualParticle.getBestPosition()[i] - actualParticle.getPosition()[i])
							+ Constants.c2 * rg * (this.globalBestSolution[i] - actualParticle.getPosition()[i]);
				}

				for (int i = 0; i < actualParticle.getPosition().length; ++i) {

					actualParticle.getPosition()[i] += actualParticle.getVelocity()[i];

					if (actualParticle.getPosition()[i] < Constants.MIN) {
						actualParticle.getPosition()[i] = Constants.MIN;
					} else if (actualParticle.getPosition()[i] > Constants.MAX) {
						actualParticle.getPosition()[i] = Constants.MAX;
					}
				}
				
				if (Constants.f(actualParticle.getPosition()) < Constants.f(actualParticle.getBestPosition())) {
					actualParticle.setBestPosition(actualParticle.getPosition());
				}
				
				if (Constants.f(actualParticle.getBestPosition()) < Constants.f(this.globalBestSolution)) {
					System.arraycopy(actualParticle.getBestPosition(), 0, this.globalBestSolution, 0, actualParticle.getBestPosition().length);
				}	
			}
		
			++this.epochs;
		}
	}

	private double[] initializeVelocity() {

		double vx = random(-(Constants.MAX - Constants.MIN), Constants.MAX - Constants.MIN);
		double vy = random(-(Constants.MAX - Constants.MIN), Constants.MAX- Constants.MIN);

		double[] newVelocity = new double[] { vx, vy };

		return newVelocity;
	}

	private double[] initializeLocation() {

		double x = random(Constants.MIN, Constants.MAX);
		double y = random(Constants.MIN, Constants.MAX);

		double[] newLocation = new double[] { x, y };

		return newLocation;
	}

	private void generateRandomSolution() {
		for (int i = 0; i < Constants.NUM_DIMENSIONS; ++i) {
			double randCoordinate = random(Constants.MIN, Constants.MAX);
			this.globalBestSolution[i] = randCoordinate;
		}
	}

	private double random(double min, double max) {
		return min + (max - min) * Math.random();
	}

	public void showSolution() {
		System.out.println("Solution of PSO problem!");
		System.out.println("Best solution -> x: " + this.globalBestSolution[0]+ " - y:" + this.globalBestSolution[1]);
		System.out.println("Value f(x,y)=" + Constants.f(globalBestSolution));
	}
}
