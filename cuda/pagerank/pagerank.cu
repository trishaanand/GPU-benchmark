#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <array>
#include <sys/time.h>

#include "kernels.cu"

//In open CL program ->
// #define MAX_THREADS_PER_BLOCK 256

#define MAX_THREADS_PER_BLOCK 512
int num_of_blocks;
int num_of_threads_per_block;

#define NUM_ITERATIONS 50

int edge_compare(const void *lhs, const void *rhs) {
	int l = ((Edge *)lhs)->in_vertex;
	int r = ((Edge *)rhs)->in_vertex;
	
	return (l - r);
}

int edge_compare_reverse(const void *lhs, const void *rhs) {
	int l = ((Edge *)lhs)->out_vertex;
	int r = ((Edge *)rhs)->out_vertex;
	
	return (l - r);
}

void run_pagerank_gpu_edgelist(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, Edge *h_graph_edges, double* time_taken)
					throw(std::string) {

	printf("In the edgelist function\n");
	float *h_pagerank = (float *) malloc (no_of_nodes*sizeof(float));
	float *h_pagerank_new = (float *) malloc (no_of_nodes*sizeof(float));
	for (int i=0; i< no_of_nodes; i++) {
		h_pagerank[i] = 0.25;
		h_pagerank_new[i] = 0.0;
	}
	
	printf("Before calling any cuda functions\n");
	//--1 transfer data from host to device

	float *d_pagerank;
	cudaMalloc( (void**) &d_pagerank, sizeof(float)*no_of_nodes) ;
	cudaMemcpy( d_pagerank, h_pagerank, sizeof(float)*no_of_nodes, cudaMemcpyHostToDevice) ;

	float *d_pagerank_new;
	cudaMalloc( (void**) &d_pagerank_new, sizeof(float)*no_of_nodes) ;
	cudaMemcpy( d_pagerank_new, h_pagerank_new, sizeof(float)*no_of_nodes, cudaMemcpyHostToDevice) ;

	Node *d_graph_nodes;
	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;

	Edge *d_graph_edges;
	cudaMalloc( (void**) &d_graph_edges, sizeof(Edge)*edge_list_size) ;
	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(Edge)*edge_list_size, cudaMemcpyHostToDevice) ;

	int *d_no_of_edges;
	cudaMalloc( (void**) &d_no_of_edges, sizeof(int)) ;
	cudaMemcpy( d_no_of_edges, &edge_list_size, sizeof(int), cudaMemcpyHostToDevice) ;

	int *d_no_of_nodes;
	cudaMalloc( (void**) &d_no_of_nodes, sizeof(int)) ;
	cudaMemcpy( d_no_of_nodes, &no_of_nodes, sizeof(int), cudaMemcpyHostToDevice) ;

	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	// int device;
	// cudaGetDevice(&device);
	// struct cudaDeviceProp properties;
	// cudaGetDeviceProperties(&properties, device);
	// printf("using %d multiprocessors\n",properties.multiProcessorCount);
	// printf("max threads per processor: %d\n",properties.maxThreadsPerMultiProcessor);
	// printf("runing with dim3 num_of_blocks %d, num_of_threads_per_block %d\n", num_of_blocks, num_of_threads_per_block);

	try{
		int i = 0;
		int j=0;
		struct timeval t1, t2;
		double elapsedTime;
		// start timer
		gettimeofday(&t1, NULL);
		while(i < NUM_ITERATIONS){

			//Initialize the d_pagerank and d_pagerank_new arrays before invoking the kernel
			//Not required for the first iteration
			if (i!=0) {
				update_pagerank_arrays<<< grid, threads, 0 >>> (d_pagerank, d_pagerank_new, d_no_of_nodes);
			}
			cudaError err = cudaGetLastError();
			if ( cudaSuccess != err )
			{
				fprintf( stderr, "cudaCheckError() for kernel launch of update_pagerank_array failed with error : %s\n",
							cudaGetErrorString( err ) );
				exit( -1 );
			}
			
			cudaDeviceSynchronize();

			edgelist<<< grid, threads, 0 >>>( 	d_graph_nodes,
												d_graph_edges, 
												d_no_of_edges,
												d_pagerank,
												d_pagerank_new);
			err = cudaGetLastError();
			if ( cudaSuccess != err )
			{
				fprintf( stderr, "cudaCheckError() for pagerank kernel launch failed with error : %s\n",
							cudaGetErrorString( err ) );
				exit( -1 );
			}
			cudaDeviceSynchronize(); 
			i++;
		}
		// stop timer
		gettimeofday(&t2, NULL);

		// compute and print the elapsed time in millisec
		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000000000.0;      // sec to ns
		elapsedTime += (t2.tv_usec - t1.tv_usec) * 1000.0;   // us to ns
		printf("Kernel time : %f ns\n", elapsedTime);
		
		// cudaError err = cudaMemcpy((void *) h_pagerank_new, (void *) d_pagerank_new, no_of_nodes*sizeof(float), cudaMemcpyDeviceToHost);
		// if (cudaSuccess != err) {
		// 	fprintf(stderr, "memcopy new pagerank_new from device to host failed with error %s\n", cudaGetErrorString(err));
		// 	exit(-1);
		// }
		// printf("New pageranks are : \n");
		// int max = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	printf("%d : %f, ", i, h_pagerank_new[i]);
		// }
		
		//--4 release cuda resources.
		cudaFree(d_graph_nodes);
		cudaFree(d_graph_edges);
		cudaFree(d_no_of_edges);
		cudaFree(d_no_of_nodes);
		cudaFree(d_pagerank);
		cudaFree(d_pagerank_new);
	}
	catch(std::string msg){		
		cudaFree(d_graph_nodes);
		cudaFree(d_graph_edges);
		cudaFree(d_no_of_edges);
		cudaFree(d_no_of_nodes);
		cudaFree(d_pagerank);
		cudaFree(d_pagerank_new);
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

// void run_pagerank_gpu_vertex_push(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int * h_neighbours, double *time_taken, char *h_graph_visited)
// 								throw(std::string) {

// 	//int number_elements = height*width;
// 	int h_depth = -1;
// 	char h_over;
	
// 	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
// 	for (int i=0; i< no_of_nodes; i++) {
// 		h_level[i] = INT_MAX;	
// 		h_graph_visited[i] = false;
// 	}
// 	h_level[0] = 0;
	
// 	//--1 transfer data from host to device
	
// 	Node *d_graph_nodes;
// 	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
// 	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
// 	Edge *d_graph_edges;
// 	cudaMalloc( (void**) &d_graph_edges, sizeof(Edge)*edge_list_size) ;
// 	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(Edge)*edge_list_size, cudaMemcpyHostToDevice) ;
	
// 	char *d_graph_visited;
// 	cudaMalloc( (void**) &d_graph_visited, sizeof(char)*no_of_nodes) ;
// 	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice) ;

// 	int *d_neighbours;
// 	cudaMalloc( (void**) &d_neighbours, sizeof(int)*edge_list_size) ;
// 	cudaMemcpy( d_neighbours, h_neighbours, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;
	
// 	char *d_over;
// 	cudaMalloc( (void**) &d_over, sizeof(char)) ;
	
// 	int *d_depth;
// 	cudaMalloc( (void**) &d_depth, sizeof(int)) ;
	
// 	int *d_level;
// 	cudaMalloc( (void**) &d_level, sizeof(int)*no_of_nodes) ;
// 	cudaMemcpy( d_level, h_level, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
// 	int *d_no_of_nodes;
// 	cudaMalloc( (void**) &d_no_of_nodes, sizeof(int)) ;
// 	cudaMemcpy( d_no_of_nodes, &no_of_nodes, sizeof(int), cudaMemcpyHostToDevice) ;
	
// 	dim3  grid( num_of_blocks, 1, 1);
// 	dim3  threads( num_of_threads_per_block, 1, 1);
	
// 	// int device;
// 	// cudaGetDevice(&device);
// 	// struct cudaDeviceProp properties;
// 	// cudaGetDeviceProperties(&properties, device);
// 	// printf("\n\nusing %d multiprocessors\n",properties.multiProcessorCount);
// 	// printf("max threads per processor: %d\n",properties.maxThreadsPerMultiProcessor);
// 	// printf("runing with dim3 num_of_blocks %d, num_of_threads_per_block %d\n\n", num_of_blocks, num_of_threads_per_block);
	
// 	try{
// 		//First run pagerank once with level 0 for all nodes who dont have reverse neighbours
// 		h_depth = -1;
// 		struct timeval t1, t2;
// 		double elapsedTime;
// 		// start timer
// 		gettimeofday(&t1, NULL);
// 		do{
// 			h_over = false;
// 			h_depth = h_depth + 1;
// 			// printf("\nNew iterations : traversing current depth %d \n", h_depth);
// 			cudaMemcpy( d_depth, &h_depth, sizeof(int), cudaMemcpyHostToDevice) ;
// 			cudaMemcpy( d_over, &h_over, sizeof(char), cudaMemcpyHostToDevice) ;
			
// 			vertex_push<<< grid, threads, 0 >>>( d_graph_nodes,
// 												d_graph_edges, 
// 												d_no_of_nodes,
// 												d_over,
// 												d_depth,
// 												d_level,
// 												d_neighbours,
// 												d_graph_visited);
// 			cudaError err = cudaGetLastError();
// 			if ( cudaSuccess != err )
// 			{
// 				fprintf( stderr, "cudaCheckError() for kernel launch failed with error : %s\n",
// 							cudaGetErrorString( err ) );
// 				exit( -1 );
// 			}
// 			cudaDeviceSynchronize(); 
	
// 			cudaMemcpy( &h_over, d_over, sizeof(char), cudaMemcpyDeviceToHost) ;
// 		}while(h_over);
// 		// stop timer
// 		gettimeofday(&t2, NULL);
	
// 		// compute and print the elapsed time in millisec
// 		// compute and print the elapsed time in millisec
// 		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000000000.0;      // sec to ns
// 		elapsedTime += (t2.tv_usec - t1.tv_usec) * 1000.0;   // us to ns
// 		printf("Kernel time : %f ns\n", elapsedTime);
		
// 		// printf("No of iterations : %d\n",h_depth);
	
// 		// cudaMemcpy(h_level, d_level, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
// 		// // cudaDeviceSynchronize(); 
// 		// printf("New depths are : \n");
// 		// int max = 0;
// 		// for (int i=0; i<no_of_nodes; i++) {
// 		// 	printf("%d : %d, ", i, h_level[i]);
// 		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
// 		// }
// 		// printf("\nMaximum depth seen is %d\n",max);
	
// 		//--4 release cuda resources.
// 		cudaFree(d_graph_nodes);
// 		cudaFree(d_graph_edges);
// 		cudaFree(d_graph_visited);
// 		cudaFree(d_no_of_nodes);
// 		cudaFree(d_neighbours);
// 		cudaFree(d_over);
// 		cudaFree(d_depth);
// 		cudaFree(d_level);
// 	}
// 	catch(std::string msg){		
// 		cudaFree(d_graph_nodes);
// 		cudaFree(d_graph_edges);
// 		cudaFree(d_graph_visited);
// 		cudaFree(d_no_of_nodes);
// 		cudaFree(d_neighbours);
// 		cudaFree(d_over);
// 		cudaFree(d_depth);
// 		cudaFree(d_level);
// 		std::string e_str = "in run_transpose_gpu -> ";
// 		e_str += msg;
// 		throw(e_str);
// 	}
// 	return ;
// }

// void run_pagerank_gpu_vertex_pull(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int *h_reverse_neighbours, double *time_taken, char *h_graph_visited)
// 								throw(std::string) {

// 	//int number_elements = height*width;
// 	int h_depth = -1;
// 	char h_over;
	
// 	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
// 	for (int i=0; i< no_of_nodes; i++) {
// 		h_level[i] = INT_MAX;	
// 		h_graph_visited[i] = false;
// 	}
// 	h_level[0] = 0;
	
// 	//--1 transfer data from host to device
	
// 	Node *d_graph_nodes;
// 	cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
// 	cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
// 	Edge *d_graph_edges;
// 	cudaMalloc( (void**) &d_graph_edges, sizeof(Edge)*edge_list_size) ;
// 	cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(Edge)*edge_list_size, cudaMemcpyHostToDevice) ;
	
// 	char *d_graph_visited;
// 	cudaMalloc( (void**) &d_graph_visited, sizeof(char)*no_of_nodes) ;
// 	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(char)*no_of_nodes, cudaMemcpyHostToDevice) ;

// 	int *d_reverse_neighbours;
// 	cudaMalloc( (void**) &d_reverse_neighbours, sizeof(int)*edge_list_size) ;
// 	cudaMemcpy( d_reverse_neighbours, h_reverse_neighbours, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) ;
	
// 	char *d_over;
// 	cudaMalloc( (void**) &d_over, sizeof(char)) ;
	
// 	int *d_depth;
// 	cudaMalloc( (void**) &d_depth, sizeof(int)) ;
	
// 	int *d_level;
// 	cudaMalloc( (void**) &d_level, sizeof(int)*no_of_nodes) ;
// 	cudaMemcpy( d_level, h_level, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
// 	int *d_no_of_nodes;
// 	cudaMalloc( (void**) &d_no_of_nodes, sizeof(int)) ;
// 	cudaMemcpy( d_no_of_nodes, &no_of_nodes, sizeof(int), cudaMemcpyHostToDevice) ;
	
// 	dim3  grid( num_of_blocks, 1, 1);
// 	dim3  threads( num_of_threads_per_block, 1, 1);
	
// 	// int device;
// 	// cudaGetDevice(&device);
// 	// struct cudaDeviceProp properties;
// 	// cudaGetDeviceProperties(&properties, device);
// 	// printf("\n\nusing %d multiprocessors\n",properties.multiProcessorCount);
// 	// printf("max threads per processor: %d\n",properties.maxThreadsPerMultiProcessor);
// 	// printf("runing with dim3 num_of_blocks %d, num_of_threads_per_block %d\n\n", num_of_blocks, num_of_threads_per_block);
	
// 	try{
// 		//First run pagerank once with level 0 for all nodes who dont have reverse reverse_neighbours
// 		h_depth = -1;
// 		struct timeval t1, t2;
// 		double elapsedTime;
// 		// start timer
// 		gettimeofday(&t1, NULL);
// 		do{
// 			h_over = false;
// 			h_depth = h_depth + 1;
// 			// printf("\nNew iterations : traversing current depth %d \n", h_depth);
// 			cudaMemcpy( d_depth, &h_depth, sizeof(int), cudaMemcpyHostToDevice) ;
// 			cudaMemcpy( d_over, &h_over, sizeof(char), cudaMemcpyHostToDevice) ;
			
// 			vertex_pull<<< grid, threads, 0 >>>( d_graph_nodes,
// 												d_graph_edges, 
// 												d_no_of_nodes,
// 												d_over,
// 												d_depth,
// 												d_level,
// 												d_reverse_neighbours,
// 												d_graph_visited);
// 			cudaError err = cudaGetLastError();
// 			if ( cudaSuccess != err )
// 			{
// 				fprintf( stderr, "cudaCheckError() for kernel launch failed with error : %s\n",
// 							cudaGetErrorString( err ) );
// 				exit( -1 );
// 			}
// 			cudaDeviceSynchronize(); 
	
// 			cudaMemcpy( &h_over, d_over, sizeof(char), cudaMemcpyDeviceToHost) ;
// 		}while(h_over);
// 		// stop timer
// 		gettimeofday(&t2, NULL);
	
// 		// compute and print the elapsed time in millisec
// 		// compute and print the elapsed time in millisec
// 		elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000000000.0;      // sec to ns
// 		elapsedTime += (t2.tv_usec - t1.tv_usec) * 1000.0;   // us to ns
// 		printf("Kernel time : %f ns\n", elapsedTime);
		
// 		// printf("No of iterations : %d\n",h_depth);
	
// 		// cudaMemcpy(h_level, d_level, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);
// 		// // cudaDeviceSynchronize(); 
// 		// printf("New depths are : \n");
// 		// int max = 0;
// 		// for (int i=0; i<no_of_nodes; i++) {
// 		// 	printf("%d : %d, ", i, h_level[i]);
// 		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
// 		// }
// 		// printf("\nMaximum depth seen is %d\n",max);
	
// 		//--4 release cuda resources.
// 		cudaFree(d_graph_nodes);
// 		cudaFree(d_graph_edges);
// 		cudaFree(d_graph_visited);
// 		cudaFree(d_no_of_nodes);
// 		cudaFree(d_reverse_neighbours);
// 		cudaFree(d_over);
// 		cudaFree(d_depth);
// 		cudaFree(d_level);
// 	}
// 	catch(std::string msg){		
// 		cudaFree(d_graph_nodes);
// 		cudaFree(d_graph_edges);
// 		cudaFree(d_graph_visited);
// 		cudaFree(d_no_of_nodes);
// 		cudaFree(d_reverse_neighbours);
// 		cudaFree(d_over);
// 		cudaFree(d_depth);
// 		cudaFree(d_level);
// 		std::string e_str = "in run_transpose_gpu -> ";
// 		e_str += msg;
// 		throw(e_str);
// 	}
// 	return ;
// }

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}

long read_and_return_no_of_nodes(char *filename) {
	long no_of_nodes = 0;
	std::ifstream fin;
	fin.open(filename);
	std::string line;
	int max = 0;
	int min = INT_MAX;
	// int min = 0;
	
	while (std::getline(fin, line)) {
		int node_index = std::stol(line);
		if (node_index > max) max = node_index;
		if (node_index < min) min = node_index;
	}

	no_of_nodes = max - min + 1;

	return no_of_nodes;
}

int read_and_return_no_of_edges(char *filename) 
{
	int no_of_edges = 0;
	std::ifstream fin;
	fin.open(filename);
	std::string line;
	
	while (std::getline(fin, line)) no_of_edges++;

	return no_of_edges; 
}

int main(int argc, char * argv[])
{
	long no_of_nodes;
	int edge_list_size;
	FILE *fp;
	Node* h_graph_nodes;
	char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	try{
		/* For now, read the input files directly instead of reading from i/o*/
		// char *input_fe = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.e";
		// char *input_fv = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.v";

		char *input_fe = "trisha-file.e";
		char *input_fv = "trisha-file.v";
	
		// char *input_fe = "/home/tanand/rodinia_3.1/graph500-10-superconnected.e";
		
		no_of_nodes = read_and_return_no_of_nodes(input_fv);
		// no_of_nodes = 1025;
		printf("Number of nodes read are : %d\n", no_of_nodes);
		edge_list_size = read_and_return_no_of_edges(input_fe);
		printf("Number of edges read are : %d\n", edge_list_size);
		
		//Read in Graph from a file
		fp = fopen(input_fe,"r");
		if(!fp){
		  printf("Error Reading EdgeGraph file\n");
		  return 0;
		}
		int source = 0;


		// allocate host memory
		h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		for (int i=0; i < no_of_nodes; i++) {
			h_graph_nodes[i].no_of_edges = 0;
			h_graph_nodes[i].starting = -1;
			h_graph_nodes[i].reverse_starting = -1;
			h_graph_nodes[i].no_of_reverse_edges = 0;
		}
		h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);
		for (int i=0; i<no_of_nodes; i++) {
			h_graph_visited[i] = false;
		}
	
		int start, edgeno;   
		
		Edge* h_graph_edges = (Edge*) malloc(sizeof(Edge)*edge_list_size);
		int neighbour_index = 0;
		for(int i=0; i < edge_list_size ; i++){
			int in_index, out_index;
			float cost; //for datagen
			fscanf(fp, "%d", &in_index);
			fscanf(fp, "%d", &out_index);
			// fscanf(fp, "%f", &cost); //only for datagen - delete for others
			h_graph_edges[i].in_vertex = in_index;
			h_graph_edges[i].out_vertex = out_index;
			//Update the number of neighbours of the node with index in_index;
			h_graph_nodes[in_index].no_of_edges++;
			// std::cout<<h_graph_edges[i].in_vertex<<" "<<h_graph_edges[i].out_vertex<<", read values are : "<<in_index<<" "<<out_index<<endl;
		}


		//Call edgelist and reverse edgelist here //

		
		//compute neighbours array for vertex push
		int qsort_size = sizeof(h_graph_edges) / sizeof(h_graph_edges[0]);
		qsort((void *) h_graph_edges, edge_list_size, sizeof(Edge), edge_compare);

		int* neighbours = (int *)malloc(edge_list_size * sizeof(int));


		int node_index = -1;
		for (int i=0; i < edge_list_size; i++) {
			// printf("%d, %d\n", h_graph_edges[i].in_vertex, h_graph_edges[i].out_vertex);
			if ((i==0) || (node_index != h_graph_edges[i].in_vertex)) {
				node_index = h_graph_edges[i].in_vertex;
				h_graph_nodes[node_index].starting = i;
				// printf("For %d, starting is %d in neighbours array\n", node_index, i);
			}
			neighbours[i] = h_graph_edges[i].out_vertex;
		}
		
		// printf("Neighbours array : \n");
		// for (int i=0; i < edge_list_size; i++) {
		// 	printf ("%d \n", neighbours[i]);
		// }

		// compute reverse neighbours (parents) for vertex pull
		qsort((void *) h_graph_edges, edge_list_size, sizeof(Edge), edge_compare_reverse);

		int* reverse_neighbours = (int *)malloc(edge_list_size * sizeof(int));

		node_index = -1;
		for (int i=0; i < edge_list_size; i++) {
			// printf("%d, %d\n", h_graph_edges[i].in_vertex, h_graph_edges[i].out_vertex);
			if ((i==0) || (node_index != h_graph_edges[i].out_vertex)) {
				// if(i!=0) std::cout<<node_index<<": starting-"<<h_graph_nodes[node_index].starting<<", reverse-starting-"<<h_graph_nodes[node_index].reverse_starting<<", num reverses-"<<h_graph_nodes[node_index].no_of_reverse_edges<<endl;
				node_index = h_graph_edges[i].out_vertex;
				h_graph_nodes[node_index].reverse_starting = i;
			}
			h_graph_nodes[node_index].no_of_reverse_edges++;
			reverse_neighbours[i] = h_graph_edges[i].in_vertex;
			// if (node_index == 0) std::cout<<reverse_neighbours[i]<<endl;
		}

		if(fp)
			fclose(fp);    
		double time_taken = 0;
		//---------------------------------------------------------
		//--gpu entry
		num_of_blocks = 1;
		// num_of_threads_per_block = no_of_nodes;
		num_of_threads_per_block = edge_list_size;
		//Make execution Parameters according to the number of edges
		//Distribute threads across multiple Blocks if necessary
		if(edge_list_size>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(edge_list_size/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}

		// int device;
		// cudaGetDevice(&device);
		// struct cudaDeviceProp properties;
		// cudaGetDeviceProperties(&properties, device);
		// printf("using %d multiprocessors\n",properties.multiProcessorCount);
		// printf("max threads per processor: %d\n",properties.maxThreadsPerMultiProcessor);
		// printf("runing with dim3 num_of_blocks %d, num_of_threads_per_block %d\n", num_of_blocks, num_of_threads_per_block);
		
		std::cout<<"\nEdgelist Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
			run_pagerank_gpu_edgelist(no_of_nodes, h_graph_nodes,edge_list_size,h_graph_edges, &time_taken);	
			
		// std::cout<<std::endl<<"Reverse Edgelist Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
		// 	run_pagerank_gpu_reverse_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_visited, &time_taken);	

		num_of_blocks = 1;
		// num_of_threads_per_block = no_of_nodes;
		num_of_threads_per_block = no_of_nodes;
		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}

		// std::cout<<std::endl<<"Vertex Push Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
		// 	run_pagerank_gpu_vertex_push(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, neighbours, &time_taken, h_graph_visited);
		// std::cout<<std::endl<<"Vertex Pull Implementation"<<std::endl;
		// for (int i=0; i<5; i++)	
		// 	run_pagerank_gpu_vertex_pull(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, reverse_neighbours, &time_taken, h_graph_visited);	
		
		//release host memory		
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);

	}
	catch(std::string msg){
		std::cout<<"--cambine: exception in main ->"<<msg<<std::endl;
		//release host memory
		free(h_graph_nodes);
		free(h_graph_mask);
		free(h_updating_graph_mask);
		free(h_graph_visited);		
	}
		
    return 0;
}
