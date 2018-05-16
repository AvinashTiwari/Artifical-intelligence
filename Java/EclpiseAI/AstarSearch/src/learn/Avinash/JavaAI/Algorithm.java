package learn.Avinash.JavaAI;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class Algorithm {

	public void aStarSearch(Node sourceNode, Node goalNode) {

		Set<Node> exploredNodes = new HashSet<Node>();

		PriorityQueue<Node> unexploredNodesQueue = new PriorityQueue<Node>();
		sourceNode.setgScore(0);
		unexploredNodesQueue.add(sourceNode);
		boolean found = false;

		while ( !unexploredNodesQueue.isEmpty() && !found ) {

			Node currentNode = unexploredNodesQueue.poll();
			exploredNodes.add(currentNode);

			if (currentNode.getValue().equals(goalNode.getValue())) {
				found = true;
			}

			for (Edge e : currentNode.getAdjacenciesList()) {
				Node childNode = e.getTargetNode();
				double cost = e.getCost();
				double tempGScore = currentNode.getgScore() + cost;
				double tempFScore = tempGScore + heuristic(childNode, goalNode);

				if( exploredNodes.contains(childNode) && (tempFScore >= childNode.getfScore()) ) {
					continue;
				} else if ( !unexploredNodesQueue.contains(childNode) || (tempFScore < childNode.getfScore()) ) {

					childNode.setParentNode(currentNode);
					childNode.setgScore(tempGScore);
					childNode.setfScore(tempFScore);

					if (unexploredNodesQueue.contains(childNode)) {
						unexploredNodesQueue.remove(childNode);
					}

					unexploredNodesQueue.add(childNode);
				}
			}
		}
	}
	
	public List<Node> printPath(Node targetNode) {

		List<Node> pathList = new ArrayList<Node>();

		for (Node node = targetNode; node != null; node = node.getParentNode()) {
			pathList.add(node);
		}

		Collections.reverse(pathList);

		return pathList;
	}
	
	// Manhattan heuristic/distance !!!
	public double heuristic(Node node1, Node node2){
		return Math.abs( node1.getX() - node2.getX() ) + Math.abs( node2.getY() - node2.getY() );
	}
}
