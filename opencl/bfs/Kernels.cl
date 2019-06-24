/* ============================================================
//--cambine: kernel funtion of Breadth-First-Search
//--author:	created by Jianbin Fang
//--date:	06/12/2010
============================================================ */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct{
	int starting;
	int no_of_edges;
} Node;
//Structure to hold edge information
typedef struct{
	int in_vertex;
	int out_vertex;
} Edge;
//--7 parameters
__kernel void BFS_1( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					__global int* g_cost, 
					const  int no_of_nodes){
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_graph_mask[tid]){
		g_graph_mask[tid]=false;
		for(int i=g_graph_nodes[tid].starting; i<(g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++){
			int id = g_graph_edges[i].out_vertex;
			if(!g_graph_visited[id]){
				g_cost[id]=g_cost[tid]+1;
				g_updating_graph_mask[id]=true;
				}
			}
	}	
}

//--5 parameters
__kernel void BFS_2(__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					__global char* g_over,
					const  int no_of_nodes
					) {
	int tid = get_global_id(0);
	if( tid<no_of_nodes && g_updating_graph_mask[tid]){

		g_graph_mask[tid]=true;
		g_graph_visited[tid]=true;
		*g_over=true;
		g_updating_graph_mask[tid]=false;
	}
}

//--10 parameters
__kernel void edgelist( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					const  int no_of_edges,
					__global char* g_over,
					__global int * g_depth,
					__global int* g_level) {

	int tid = get_global_id(0);

	if( (tid<no_of_edges) && (g_level[g_graph_edges[tid].in_vertex]==*g_depth)){

		int new_depth = *g_depth + 1;
		
		if (atomic_min(&g_level[g_graph_edges[tid].out_vertex], new_depth) > new_depth) {
			 //atomic min returns the old value
			//if the depth seen by a node is higher than the current new depth, we should try more depths
			*g_over=true;
		}
	}	
}

//--10 parameters
__kernel void reverse_edgelist( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					__global char* g_graph_mask, 
					__global char* g_updating_graph_mask, 
					__global char* g_graph_visited, 
					const  int no_of_edges,
					__global char* g_over,
					__global int * g_depth,
					__global int* g_level) {

	int tid = get_global_id(0);

	if( (tid<no_of_edges) && (g_level[g_graph_edges[tid].out_vertex]==*g_depth)){

		int new_depth = *g_depth + 1;
		
		if (atomic_min(&g_level[g_graph_edges[tid].in_vertex], new_depth) > new_depth) {
			 //atomic min returns the old value
			//if the depth seen by a node is higher than the current new depth, we should try more depths
			*g_over=true;
		}
	}	
}

//--7 parameters
__kernel void vertex_push( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					const  int no_of_nodes,
					__global char* g_over,
					__global int * g_depth,
					__global int* g_level,
					const __global int* g_neighbours) {

	int tid = get_global_id(0);

	if( (tid<no_of_nodes) && (g_level[tid]==*g_depth) ){

		int new_depth = *g_depth + 1;
		int starting = g_graph_nodes[tid].starting;
		int max = starting + g_graph_nodes[tid].no_of_edges;
		for (int i=starting; i<max; i++ ) {
			if (atomic_min(&g_level[g_neighbours[i]], new_depth) > new_depth) {
				//atomic min returns the old value
				//if the depth seen by a node is higher than the current new depth, we should try more depths
				*g_over=true;
			}
		}
	}	
}

//--7 parameters
__kernel void vertex_pull( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					const  int no_of_nodes,
					__global char* g_over,
					__global int * g_depth,
					__global int* g_level,
					const __global int* g_reverse_neighbours) {

	int tid = get_global_id(0);
	int new_depth = *g_depth + 1;
	if( (tid<no_of_nodes) && (g_level[tid] > new_depth) ){

		int starting = g_graph_nodes[tid].starting;
		int max = starting + g_graph_nodes[tid].no_of_edges;
		for (int i=starting; i<max; i++ ) {
			if (g_level[g_reverse_neighbours[i]] == g_depth) {
				g_level[tid] = new_depth;
				*g_over = true;
				break;
			}
		}
	}	
}

