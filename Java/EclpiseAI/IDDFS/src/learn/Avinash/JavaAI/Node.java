package learn.Avinash.JavaAI;

import java.util.*;

public class Node {

	private String name;
	private int depthLevel = 0;
	private List<Node> adjacenciesList;
	
	public Node(String name){
		this.name = name;
		this.adjacenciesList = new ArrayList<>();
	}
	
	public void addNeighbour(Node vertex){
		this.adjacenciesList.add(vertex);
	}
	
	public int getDepthLevel() {
		return depthLevel;
	}

	public void setDepthLevel(int depthLevel) {
		this.depthLevel = depthLevel;
	}
	
	public String getName(){
		return this.name;
	}

	public List<Node> getAdjacenciesList() {
		return adjacenciesList;
	}

	@Override
	public String toString() {
		return this.name;
	}
}
