package learn.Avinash.JavaAI;

public class App {

	public static void main(String[] args) {
		BFS bfs = new BFS();
		Vertex vertex1 = new Vertex(1);
		Vertex vertex2 = new Vertex(2);
		Vertex vertex3 = new Vertex(3);
		Vertex vertex4 = new Vertex(4);
		Vertex vertex5 = new Vertex(5);
		
		vertex1.addNeigbourVertex(vertex2);
		vertex1.addNeigbourVertex(vertex4);

		vertex4.addNeigbourVertex(vertex5);

		vertex2.addNeigbourVertex(vertex3);
		
		bfs.bfs(vertex1);

	}

}
