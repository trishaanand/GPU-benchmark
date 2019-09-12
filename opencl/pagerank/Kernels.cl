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

inline void atomicAdd_g_f(volatile __global float *addr, float val)
   {
       union {
           unsigned int u32;
           float        f32;
       } next, expected, current;
   	current.f32    = *addr;
       do {
   	   expected.f32 = current.f32;
           next.f32     = expected.f32 + val;
   		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                               expected.u32, next.u32);
       } while( current.u32 != expected.u32 );
   }

//Apparently less correct. Older implementattion
inline void AtomicAdd_g_f(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, 
            newVal.intVal) != prevVal.intVal);
}

void float_atomic_add(__global float *loc, const float f) {
	private float old = *loc;
	private float sum = old + f;
	while(atomic_cmpxchg((__global int*)loc, *((int*)&old), *((int*)&sum)) !=*((int*)&old)){
		old = *loc;
		sum = old + f;
	}
}

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
					const  int no_of_edges,
					__global float* g_pagerank,
					__global float* g_pagerank_new) {

	int tid = get_global_id(0);

	if(tid<no_of_edges) {

		float new_rank = 0.0f;
		int in_vertex = g_graph_edges[tid].in_vertex;
		int out_vertex = g_graph_edges[tid].out_vertex;
		int degree = g_graph_nodes[in_vertex].no_of_edges;
		
		if (degree != 0) new_rank = g_pagerank[in_vertex] / degree;
        float_atomic_add(&g_pagerank_new[out_vertex], new_rank);
		// printf("After atomic add : new updated value for out_vertex : %d is : %f\n", out_vertex, g_pagerank_new[out_vertex]);
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

//--6 parameters
__kernel void vertex_push( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					const  int no_of_nodes,
					const __global int* g_neighbours,
					__global float* g_pagerank,
					__global float* g_pagerank_new ) {

	int tid = get_global_id(0);

	if (tid<no_of_nodes) {

		float new_rank = 0.0f;
		int degree = g_graph_nodes[tid].no_of_edges;
		if (degree!=0) new_rank = g_pagerank[tid] / degree;

		int starting = g_graph_nodes[tid].starting;
		int max = starting + g_graph_nodes[tid].no_of_edges;
		// printf("For node %d, starting is : %d and max is : %d\n", tid, starting, max);
		for (int i=starting; i<max; i++ ) {
			float_atomic_add(&g_pagerank_new[g_neighbours[i]], new_rank);
			// printf("After atomic add : new updated value for out_vertex : %d is : %f\n", g_neighbours[i], g_pagerank_new[g_neighbours[i]]);
		}
	}	
}

//--6 parameters
__kernel void vertex_pull( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					const  int no_of_nodes,
					const __global int* g_neighbours,
					__global float* g_pagerank,
					__global float* g_pagerank_new ) {

	int tid = get_global_id(0);

	if (tid<no_of_nodes) {

		//initialize to the last page rank seen by the vertex
		float new_rank = g_pagerank[tid]; 
	
		int starting = g_graph_nodes[tid].starting;
		int max = starting + g_graph_nodes[tid].no_of_edges;
		
		for (int i=starting; i<max; i++ ) {
			int neighbour_index = g_neighbours[i];
			int degree = g_graph_nodes[neighbour_index].no_of_edges;
			if (degree != 0) new_rank += g_pagerank[neighbour_index]/degree;
		}
		g_pagerank_new[tid] = new_rank;
	}	
}

