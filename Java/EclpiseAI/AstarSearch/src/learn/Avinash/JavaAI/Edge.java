package learn.Avinash.JavaAI;

public class Edge {
	
	public final double cost;
	public final Node targetNode;

	public Edge(Node targetNode, double cost) {
		this.targetNode = targetNode;
		this.cost = cost;
	}

	public double getCost() {
		return cost;
	}

	public Node getTargetNode() {
		return targetNode;
	}
}
