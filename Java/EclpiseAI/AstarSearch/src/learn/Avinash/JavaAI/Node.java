package learn.Avinash.JavaAI;

import java.util.ArrayList;
import java.util.List;

public class Node implements Comparable<Node> {

	private String value;
	private double gScore;
	private double fScore = 0;
	private double x;
	private double y;
	private List<Edge> adjacenciesList;
	private Node parentNode;

	public Node(String value) {
		this.value = value;
		this.adjacenciesList = new ArrayList<>();
	}

	public double getgScore() {
		return gScore;
	}
	
	public void addNeighbour(Edge edge){
		this.adjacenciesList.add(edge);
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

	public void setgScore(double gScore) {
		this.gScore = gScore;
	}

	public double getfScore() {
		return fScore;
	}

	public void setfScore(double fScore) {
		this.fScore = fScore;
	}

	public Node getParentNode() {
		return parentNode;
	}

	public String getValue() {
		return value;
	}

	public void setValue(String value) {
		this.value = value;
	}

	public void setParentNode(Node parentNode) {
		this.parentNode = parentNode;
	}

	public List<Edge> getAdjacenciesList() {
		return adjacenciesList;
	}

	@Override
	public String toString() {
		return value;
	}

	@Override
	public int compareTo(Node otherNode) {
		return Double.compare(this.fScore, otherNode.getfScore());
	}
}
