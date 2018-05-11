package learn.Avinash.JavaAI;

import java.util.*;

public class Vertex {
	private int data;
	private boolean visited;
	private List<Vertex> neighbourList;

	public Vertex(int data) {
		this.data = data;
		neighbourList = new ArrayList<>();
	}

	public int getData() {
		return data;
	}

	public void setData(int data) {
		this.data = data;
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
		return ""+this.data;	}
	
	public void addNeigbourVertex(Vertex vertex)
	{
		this.neighbourList.add(vertex);
		
	}
}
