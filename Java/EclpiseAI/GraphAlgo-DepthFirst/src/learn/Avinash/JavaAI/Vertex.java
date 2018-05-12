package learn.Avinash.JavaAI;

import java.util.ArrayList;
import java.util.List;

public class Vertex {
	private String name;
	private boolean visited;
	private List<Vertex> neighbourList;

	public Vertex(String name) {
		this.name = name;
		this.neighbourList = new ArrayList<>();

	}

	public void addNeighbour(Vertex vertex) {
		this.neighbourList.add(vertex);
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public boolean isVisited() {
		return visited;
	}

	public void setVisited(boolean visited) {
		this.visited = visited;
	}

	public List<Vertex> getNeighbourList() {
		return neighbourList;
	}

	public void setNeighbourList(List<Vertex> neighbourList) {
		this.neighbourList = neighbourList;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		return this.name;
	}

}
