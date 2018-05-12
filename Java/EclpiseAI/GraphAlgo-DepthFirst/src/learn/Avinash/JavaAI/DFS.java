package learn.Avinash.JavaAI;

import java.util.List;
import java.util.Stack;

public class DFS {
	private Stack<Vertex> stack;

	public DFS() {
		stack = new Stack<>();
	}

	public void dfs(List<Vertex> vertex) {
		for (Vertex v : vertex) {
			if (!v.isVisited()) {
				v.setVisited(true);
				dfsWithStack(v);
			}
		}
	}

	private void dfsWithStack(Vertex v) {
		this.stack.add(v);
		v.setVisited(true);

		while (!stack.isEmpty()) {
			Vertex actualVertex = this.stack.pop();
			System.out.println(actualVertex + " ");
			
			for(Vertex getNeigbour : actualVertex.getNeighbourList())
			{
				if(!getNeigbour.isVisited())
				{
					getNeigbour.setVisited(true);
					this.stack.push(getNeigbour);
				}
			}
		}
	}
}
