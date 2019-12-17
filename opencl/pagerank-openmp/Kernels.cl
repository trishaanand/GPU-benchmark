#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
//Structure to hold a node information
typedef struct{
	int starting;
	int reverse_starting;
	int no_of_edges;
	int no_of_reverse_edges;
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

__kernel void update_pagerank_arrays (__global float* g_pagerank,
									  __global float* g_pagerank_new,
									  const int no_of_nodes ) {
	int tid = get_global_id(0);

	if (tid < no_of_nodes) {
		//for the next iteration update the pagerank array with the new calculated values
		g_pagerank[tid] = g_pagerank_new[tid];
		//reset the new page rank array to 0 for the next iteration.
		// g_pagerank_new[tid] = 0.0;
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
		
		Edge current_edge = g_graph_edges[tid];

		float new_rank = 0.0f;
		int in_vertex = current_edge.in_vertex;
		int out_vertex = current_edge.out_vertex;
		int degree = g_graph_nodes[in_vertex].no_of_edges;
		
		if (degree != 0) new_rank = g_pagerank[in_vertex] / degree;
        float_atomic_add(&g_pagerank_new[out_vertex], new_rank);
		// printf("After atomic add : new updated value for out_vertex : %d is : %f\n", out_vertex, g_pagerank_new[out_vertex]);
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
		Node current_node = g_graph_nodes[tid];
		float new_rank = 0.0f;
		int degree = current_node.no_of_edges;
		if (degree!=0) 
			new_rank = g_pagerank[tid] / degree;

		int starting = current_node.starting;
		int max = starting + current_node.no_of_edges;
		// printf("For node %d, starting is : %d and max is : %d\n", tid, starting, max);
		for (int i=starting; i<max; i++ ) {
			float_atomic_add(&g_pagerank_new[g_neighbours[i]], new_rank);
			// if(g_neighbours[i]==0) {
			// 	printf("%d : Added rank of vertex %d which was %f\n", g_neighbours[i], tid, new_rank);
			// }
		}
	}	
}

//--6 parameters
__kernel void vertex_pull( const __global Node* g_graph_nodes,
					const __global Edge* g_graph_edges, 
					const  int no_of_nodes,
					const __global int* g_reverse_neighbours,
					__global float* g_pagerank,
					__global float* g_pagerank_new ) {

	int tid = get_global_id(0);

	if (tid<no_of_nodes) {

		//initialize to the last page rank seen by the vertex
		float new_rank = 0.0f; 

		Node current_node = g_graph_nodes[tid];

		int starting = current_node.reverse_starting;
		int max = starting + current_node.no_of_reverse_edges;
		
		// if (tid == 0) 
		// 	printf("For %d, starting is : %d and max is %d\n\n", tid, starting, max);

		for (int i=starting; i<max; i++ ) {
			int neighbour_index = g_reverse_neighbours[i];
			Node neighbour_node = g_graph_nodes[neighbour_index];

			int degree = neighbour_node.no_of_edges;
			if (degree != 0) new_rank += (g_pagerank[neighbour_index])/degree;
			// if(tid==0) {
			// 	printf("%d : Added rank of vertex %d which was %f\n", tid, neighbour_index, neighbour_rank);
			// }
		}
		g_pagerank_new[tid] += new_rank;
	}	
}

