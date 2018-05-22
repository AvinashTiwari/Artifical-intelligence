package TabuSearchI.TabuSearchI;

import java.util.ArrayList;
import java.util.List;

public class State {

	private double x;
	private double y;
	private double z; //f(x,y)
	private List<State> neighbors;

	public State(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
		this.neighbors = new ArrayList<>();
	}

	public double getX() {
		return x;
	}

	public void setX(double x) {
		this.x = x;
	}

	public double getY() {
		return y;
	}

	public void setY(double y) {
		this.y = y;
	}

	public double getZ() {
		return z;
	}

	public void setZ(double z) {
		this.z = z;
	}
	
	public void addNeigbor(State state) {
		this.neighbors.add(state);
	}
	
	public List<State> getNeighbors() {
		return neighbors;
	}

	@Override
	public String toString() {
		return "(" + this.x + ";" + this.y + ";" + this.z + ")";
	}
}