package learn.Avinash.JavaAI;

import java.util.Stack;

public class Algorithm {

	private Node targetVertex;
	private volatile boolean isTargetFound;
	
	public Algorithm(Node targetVertex){
		this.targetVertex = targetVertex;
	}
	
	public void runDeepeningSearch(Node rootVertex) {
		
		int depth = 0;
		
		while(!isTargetFound){
			System.out.println();
			dfs(rootVertex,depth);
			depth++;
		}	
	}
	
	public void dfs(Node sourceVertex, int depthLevel){
		
		Stack<Node> stack = new Stack<>();
		sourceVertex.setDepthLevel(0);
		stack.push(sourceVertex);
		
		while( !stack.isEmpty() ){
			
			Node actualNode = stack.pop();
			System.out.print(actualNode+" ");
			
			if( actualNode.getName().equals(this.targetVertex.getName()) ){
				System.out.println("\nNode found...");
				this.isTargetFound = true;
				return;
			}
			
			if( actualNode.getDepthLevel() >= depthLevel ){
				continue;
			}
			
			for(Node node : actualNode.getAdjacenciesList()){
					node.setDepthLevel(actualNode.getDepthLevel()+1);
					stack.push(node);
			}	
		}	
	}
}
