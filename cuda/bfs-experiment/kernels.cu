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


__global__ void edgelist( Node* g_graph_nodes, Edge* g_graph_edges, char* g_graph_visited, int* no_of_edges, char* g_over, int * g_depth, int* g_level) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid == 8046) {
    //     printf("for tid %d, in vertex is %d, out vertex is %d,  glevel of in vertex is %d\n", tid, g_graph_edges[tid].in_vertex, g_graph_edges[tid].out_vertex, 
    //                 g_level[g_graph_edges[tid].in_vertex]);
    // }
    if( (tid<*no_of_edges) && (g_graph_visited[g_graph_edges[tid].out_vertex] != true) &&(g_level[g_graph_edges[tid].in_vertex]==*g_depth)) {
        // printf("Inside the if condition for tid %d where in_vertex is %d\n", tid, g_graph_edges[tid].in_vertex);
        int new_depth = *g_depth + 1;
        // g_graph_visited[g_graph_edges[tid].in_vertex] = true;
        g_graph_visited[g_graph_edges[tid].out_vertex] = true;
        // if (g_level[g_graph_edges[tid].out_vertex] > new_depth) {
		// 	g_level[g_graph_edges[tid].out_vertex] = new_depth;
		// 	*g_over = true;
		// }
        if (atomicMin(&g_level[g_graph_edges[tid].out_vertex], new_depth) > new_depth) {
            //atomic min returns the old value and updates the first argument with the minimum.
            //if the depth seen by a node is higher than the current new depth, we should try more depths
            // printf("Update: for tid %d , updating out vertex %d when in vertex is %d. New level is of out vertex is %d\n", tid, 
            // g_graph_edges[tid].out_vertex, g_graph_edges[tid].in_vertex, g_level[g_graph_edges[tid].out_vertex]);
            *g_over=true;
        }
    }	
}

__global__ void reverse_edgelist( Node* g_graph_nodes,
    Edge* g_graph_edges, 
    char* g_graph_visited, 
    int* no_of_edges,
    char* g_over,
    int * g_depth,
    int* g_level) {
        
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( (tid<*no_of_edges) && (g_graph_visited[g_graph_edges[tid].in_vertex] != true) && (g_level[g_graph_edges[tid].out_vertex]==*g_depth)) {

        int new_depth = *g_depth + 1;
        g_graph_visited[g_graph_edges[tid].in_vertex] = true;
        // if (g_level[g_graph_edges[tid].in_vertex] > new_depth) {
		// 	g_level[g_graph_edges[tid].in_vertex] = new_depth;
		// 	*g_over = true;
		// }
        if (atomicMin(&g_level[g_graph_edges[tid].in_vertex], new_depth) > new_depth) {
            //atomic min returns the old value
            //if the depth seen by a node is higher than the current new depth, we should try more depths
            *g_over=true;
        }
    }	
}

__global__ void vertex_push( Node* g_graph_nodes,
    Edge* g_graph_edges, 
    int* no_of_nodes,
    char* g_over,
    int * g_depth,
    int* g_level,
    int* g_neighbours,
    char* g_graph_visited) {
   
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if( (tid<*no_of_nodes) && (g_level[tid]==*g_depth) ) {

        int new_depth = *g_depth + 1;
        int starting = g_graph_nodes[tid].starting;
        int max = starting + g_graph_nodes[tid].no_of_edges;
        // printf("At %d. starting : %d, max : %d, neighbours array is [%d %d %d %d %d %d %d]\n", tid, starting, max, g_neighbours[0], g_neighbours[1],
        //         g_neighbours[2], g_neighbours[3], g_neighbours[4], g_neighbours[5], g_neighbours[6]);
        for (int i=starting; i<max; i++ ) {
            int neighbour_index = g_neighbours[i];
            // printf("\nAt %d. Before checking vertex %d , graph_visited %c , level %d\n", tid, neighbour_index, g_graph_visited[neighbour_index], g_level[neighbour_index]);
            if (g_graph_visited[neighbour_index] != true) {
                g_graph_visited[neighbour_index] = true;
                // if (g_level[neighbour_index] > new_depth) {
                //     g_level[neighbour_index] = new_depth;
                //     *g_over = true;
                // }
                if (atomicMin(&g_level[neighbour_index], new_depth) > new_depth) {
                    //atomic min returns the old value
                    //if the depth seen by a node is higher than the current new depth, we should try more depths
                    // printf("At %d. Updating vertex %d to depth %d. \n", tid, neighbour_index, g_level[neighbour_index]);
                    *g_over=true;
                }
            }
        }
    }	
}

__global__ void vertex_pull(  Node* g_graph_nodes,
    Edge* g_graph_edges, 
    int* no_of_nodes,
     char* g_over,
     int *g_depth,
     int* g_level,
      int* g_reverse_neighbours,
     char* g_graph_visited) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int new_depth = *g_depth + 1;
    
    if( (tid < *no_of_nodes) && (g_graph_visited[tid]!=true)) {
        int starting = g_graph_nodes[tid].reverse_starting;
        int max = starting + g_graph_nodes[tid].no_of_reverse_edges;

        // printf("At %d. starting : %d, max : %d\n", tid, starting, max);
        for (int i=starting; i<max; i++ ) {
            int neighbour_index = g_reverse_neighbours[i];
            if (g_level[neighbour_index] == *g_depth) {
                g_level[tid] = new_depth;
                *g_over = true;
                g_graph_visited[tid] = true;
                // break; 
            }
        }
    } 
}

   

#endif
