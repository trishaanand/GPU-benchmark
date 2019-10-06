#ifndef _KERNELS_
#define _KERNELS_

//Structure to hold a node information
typedef struct
{
	int starting;
	int reverse_starting;
	int no_of_edges;
	int no_of_reverse_edges;
}Node;

typedef struct {
    int in_vertex;
	int out_vertex;
}Edge;


__global__ void update_pagerank_arrays (float* g_pagerank, float* g_pagerank_new, int *no_of_nodes ) {
   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < *no_of_nodes) {
        //for the next iteration update the pagerank array with the new calculated values
        g_pagerank[tid] = g_pagerank_new[tid];
    }

    // if(tid == 5) {
    //     printf("New pagerank is : ");
    //     for (int i=0; i<6; i++) {
    //         printf("    %f", g_pagerank[i]);
    //     }
    //     printf("\n");
    // }
}

__global__ void edgelist( Node* g_graph_nodes, Edge* g_graph_edges, int* no_of_edges, float* g_pagerank, float* g_pagerank_new) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( tid<*no_of_edges ) {
        
        Edge current_edge = g_graph_edges[tid];

		float new_rank = 0.0f;
		int in_vertex = current_edge.in_vertex;
		int out_vertex = current_edge.out_vertex;
		int degree = g_graph_nodes[in_vertex].no_of_edges;
		
		if (degree != 0) new_rank = g_pagerank[in_vertex] / degree;
        
        atomicAdd(&g_pagerank_new[out_vertex], new_rank);

    }	
}

__global__ void vertex_push( Node* g_graph_nodes,
    Edge* g_graph_edges, 
    int* no_of_nodes,
    int* g_neighbours,
    float* g_pagerank, 
    float* g_pagerank_new) {
   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( tid < *no_of_nodes ) {
        Node current_node = g_graph_nodes[tid];
		float new_rank = 0.0f;
		int degree = current_node.no_of_edges;
		if (degree!=0) 
			new_rank = g_pagerank[tid] / degree;

		int starting = current_node.starting;
		int max = starting + current_node.no_of_edges;
		for (int i=starting; i<max; i++ ) {
			atomicAdd(&g_pagerank_new[g_neighbours[i]], new_rank);
		}
    }	
}

__global__ void vertex_pull( Node* g_graph_nodes,
    Edge* g_graph_edges, 
    int* no_of_nodes,
    int* g_reverse_neighbours,
    float* g_pagerank, 
    float* g_pagerank_new) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( tid < *no_of_nodes ) {
        //initialize to the last page rank seen by the vertex
		float new_rank = 0.0f; 

		Node current_node = g_graph_nodes[tid];

		int starting = current_node.reverse_starting;
		int max = starting + current_node.no_of_reverse_edges;
		
		for (int i=starting; i<max; i++ ) {
			int neighbour_index = g_reverse_neighbours[i];
			Node neighbour_node = g_graph_nodes[neighbour_index];

			int degree = neighbour_node.no_of_edges;
			if (degree != 0) new_rank += (g_pagerank[neighbour_index])/degree;
		}
		g_pagerank_new[tid] += new_rank;
    } 
}

   

#endif
