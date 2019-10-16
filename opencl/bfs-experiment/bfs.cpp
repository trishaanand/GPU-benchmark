//--by Jianbin Fang

#define __CL_ENABLE_EXCEPTIONS
#include <cstdlib>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include <climits>
#include <sstream>
#include <algorithm>

#ifdef  PROFILING
#include "timer.h"
#endif

#include "CLHelper.h"
#include "util.h"

#define MAX_THREADS_PER_BLOCK 256

//Structure to hold a node information
struct Node
{
	int starting;
	int reverse_starting;
	int no_of_edges;
	int no_of_reverse_edges;
};

struct Edge {
	int in_vertex;
	int out_vertex;
};

int bfs_starting_node = 0;
int work_group_size;

bool edge_compare(Edge lhs, Edge rhs) {
	return (lhs.in_vertex < rhs.in_vertex);
}

bool edge_compare_reverse(Edge lhs, Edge rhs) {
	return (lhs.out_vertex < rhs.out_vertex);
}

//----------------------------------------------------------
//--bfs on cpu
//--programmer:	jianbin
//--date:	26/01/2011
//--note: width is changed to the new_width
//----------------------------------------------------------
void run_bfs_cpu(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		Edge *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, int *h_cost_ref){
	char stop;
	int k = 0;
	do{
		//if no thread changes this value then the loop stops
		stop=false;
		for(int tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
				h_graph_mask[tid]=false;
				for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++){
					int id = h_graph_edges[i].out_vertex;	//--cambine: node id is connected with node tid
					if(!h_graph_visited[id]){	//--cambine: if node id has not been visited, enter the body below
						h_cost_ref[id]=h_cost_ref[tid]+1;
						h_updating_graph_mask[id]=true;
					}
				}
			}		
		}

  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop);
}
//----------------------------------------------------------
//--breadth first search on GPUs - rodinia
//----------------------------------------------------------
void run_bfs_gpu_rodinia(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		Edge *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, int *h_cost) 
					throw(std::string){

	//int number_elements = height*width;
	char h_over;
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, \
			d_graph_visited, d_cost, d_over;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_graph_mask = _clMallocRW(no_of_nodes*sizeof(char), h_graph_mask);
		d_updating_graph_mask = _clMallocRW(no_of_nodes*sizeof(char), h_updating_graph_mask);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);


		d_cost = _clMallocRW(no_of_nodes*sizeof(int), h_cost);
		d_over = _clMallocRW(sizeof(char), &h_over);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_graph_mask, no_of_nodes*sizeof(char), h_graph_mask);	
		_clMemcpyH2D(d_updating_graph_mask, no_of_nodes*sizeof(char), h_updating_graph_mask);	
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);	
		_clMemcpyH2D(d_cost, no_of_nodes*sizeof(int), h_cost);	
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		do{
			h_over = false;
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			//--kernel 0
			int kernel_id = 0;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, d_cost);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			//--kernel 1
			kernel_id = 1;
			kernel_idx = 0;			
			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			
			//work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);			
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			}while(h_over);
			
		_clFinish();
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInSeconds();
#endif
		//--3 transfer data from device to host
		_clMemcpyD2H(d_cost,no_of_nodes*sizeof(int), h_cost);
		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_cost);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		printf("CL init or following device mallocs failed. Check whether the program is on a machine with GPU\n");
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_cost);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--breadth first search on GPUs - edgelist
//----------------------------------------------------------
void run_bfs_gpu_edgelist(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		Edge *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, double* time_taken)
					throw(std::string){

	//int number_elements = height*width;
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;	
		h_graph_visited[i] = false;
	}
	h_level[bfs_starting_node] = 0;
	// h_level[bfs_starting_node] = true;
	// std::cout<<"Before cl_mem and init"<<std::endl;
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, \
			d_graph_visited, d_over, d_depth, d_level;
	try{
		//--1 transfer data from host to device
		_clInit();	
		// std::cout<<"After cl_mem and init. Before clMallocs"<<std::endl;
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);

		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);	
		// _clFinish();
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer, iteration_timer;
		double kernel_time = 0.0;
		double iteration_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		//First run BFS once with level 0 for all nodes who dont have reverse neighbours
		h_depth = -1;
		
		do{
			iteration_timer.reset();
			iteration_timer.start();
			h_over = false;
			h_depth = h_depth + 1;
			
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
			
			// printf("In the iteration %d\n", h_depth);

			int kernel_id = 2;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, &edge_list_size, sizeof(int));
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, d_depth);
			_clSetArgs(kernel_id, kernel_idx++, d_level);
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, edge_list_size, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			// _clFinish();
			iteration_timer.stop();
			iteration_time = iteration_timer.getTimeInNanoSeconds();
			std::cout<<"Iteration "<<h_depth<<", time(ns) : "<<iteration_time<<std::endl;
		}while(h_over);
		
		// _clFinish();
		// std::cout<<"Number of starting points : "<<num_of_starting_points<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
#endif
		//--3 transfer data from device to host
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// std::cout<<"New depths are : "<<std::endl;
		// int max = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	std::cout<<i<<" : "<<h_level[i]<<", ";
		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
		// }
		// std::cout<<std::endl;
		// std::cout<<"Maximum depth seen is "<<max<<std::endl;

		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--breadth first search on GPUs - reverse edgelist
//----------------------------------------------------------
void run_bfs_gpu_reverse_edgelist(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		Edge *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, double* time_taken)
					throw(std::string){

	//int number_elements = height*width;
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;	
	}
	h_level[bfs_starting_node] = 0;
	
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, \
			d_graph_visited, d_over, d_depth, d_level;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_graph_mask = _clMalloc(no_of_nodes*sizeof(char), h_graph_mask);
		d_updating_graph_mask = _clMallocRW(no_of_nodes*sizeof(char), h_updating_graph_mask);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);

		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_graph_mask, no_of_nodes*sizeof(char), h_graph_mask);	
		_clMemcpyH2D(d_updating_graph_mask, no_of_nodes*sizeof(char), h_updating_graph_mask);	
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);	
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer, iteration_timer;
		double kernel_time = 0.0;
		double iteration_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		//First run BFS once with level 0 for all nodes who dont have reverse neighbours
		h_depth = -1;
		
		do{
			iteration_timer.reset();
			iteration_timer.start();
			h_over = false;
			h_depth = h_depth + 1;
			
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
			
			int kernel_id = 3;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			_clSetArgs(kernel_id, kernel_idx++, &edge_list_size, sizeof(int));
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, d_depth);
			_clSetArgs(kernel_id, kernel_idx++, d_level);
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, edge_list_size, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			iteration_timer.stop();
			iteration_time = iteration_timer.getTimeInNanoSeconds();
			std::cout<<"Iteration "<<h_depth<<", time(ns) : "<<iteration_time<<std::endl;
		}while(h_over);
		// std::cout<<"First bfs, no of iterations : "<<h_depth<<std::endl;

		// //Update the h_graph_visited array
		// _clMemcpyD2H(d_graph_visited,no_of_nodes*sizeof(char), h_graph_visited);
		// //Update the h_level array
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);

		// //Now run BFS for all the graph nodes which were disconnected from the above.
		// int num_of_starting_points = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	if (h_graph_visited[i] != true) {
		// 		num_of_starting_points++;
		// 		//Start BFS from the ith node
		// 		h_level[i] = 0;
		// 		h_graph_visited[i] = true;
		// 		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
		// 		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		// 		h_depth = -1;
				
		// 		do{
		// 			h_over = false;
		// 			h_depth = h_depth + 1;
					
		// 			_clMemcpyH2D(d_over, sizeof(char), &h_over);
		// 			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
					
		// 			int kernel_id = 3;
		// 			int kernel_idx = 0;
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_mask);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_updating_graph_mask);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
		// 			_clSetArgs(kernel_id, kernel_idx++, &edge_list_size, sizeof(int));
		// 			_clSetArgs(kernel_id, kernel_idx++, d_over);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_depth);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_level);
					
		// 			//int work_items = no_of_nodes;
		// 			_clInvokeKernel(kernel_id, edge_list_size, work_group_size);
					
		// 			_clMemcpyD2H(d_over,sizeof(char), &h_over);
		// 		}while(h_over);
		// 		//Update the h_graph_visited array
		// 		_clMemcpyD2H(d_graph_visited,no_of_nodes*sizeof(char), h_graph_visited);
		// 		//Update the h_level array
		// 		_clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// 	}
		// }

		_clFinish();
		// std::cout<<"Number of starting points : "<<num_of_starting_points<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
#endif
		//--3 transfer data from device to host
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// std::cout<<"New depths are : "<<std::endl;
		// int max = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	std::cout<<i<<" : "<<h_level[i]<<", ";
		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
		// }
		// std::cout<<std::endl;
		// std::cout<<"Maximum depth seen is "<<max<<std::endl;

		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_graph_mask);
		_clFree(d_updating_graph_mask);
		_clFree(d_graph_visited);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--breadth first search on GPUs - vertex push
//----------------------------------------------------------
void run_bfs_gpu_vertex_push(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int * h_neighbours, double *time_taken, char *h_graph_visited)
								throw(std::string) {
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;	
	}
	h_level[bfs_starting_node] = 0;
	
	cl_mem d_graph_nodes, d_graph_edges, d_over, d_depth, d_level, d_neighbours, d_graph_visited;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_neighbours = _clMalloc(edge_list_size*sizeof(int), h_neighbours);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);
		
		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		_clMemcpyH2D(d_neighbours, edge_list_size*sizeof(int), h_neighbours);
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer, iteration_timer;
		double kernel_time = 0.0;
		double iteration_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		//First run BFS once with level 0 for all nodes who dont have reverse neighbours
		h_depth = -1;
		
		do{
			iteration_timer.reset();
			iteration_timer.start();
			h_over = false;
			h_depth = h_depth + 1;
			
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
			//--kernel 0
			int kernel_id = 4;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, d_depth);
			_clSetArgs(kernel_id, kernel_idx++, d_level);
			_clSetArgs(kernel_id, kernel_idx++, d_neighbours);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			iteration_timer.stop();
			iteration_time = iteration_timer.getTimeInNanoSeconds();
			std::cout<<"Iteration "<<h_depth<<", time(ns) : "<<iteration_time<<std::endl;
		}while(h_over);
			
		// //Now run BFS for all the graph nodes which were disconnected from the above.
		// int num_of_starting_points = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	if (h_graph_visited[i] != true) {
		// 		num_of_starting_points++;
		// 		//Start BFS from the ith node
		// 		h_level[i] = 0;
		// 		h_graph_visited[i] = true;
		// 		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
		// 		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		// 		h_depth = -1;
				
		// 		do{
		// 			h_over = false;
		// 			h_depth = h_depth + 1;
					
		// 			_clMemcpyH2D(d_over, sizeof(char), &h_over);
		// 			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
		// 			//--kernel 0
		// 			int kernel_id = 4;
		// 			int kernel_idx = 0;
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
		// 			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
		// 			_clSetArgs(kernel_id, kernel_idx++, d_over);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_depth);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_level);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_neighbours);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
					
		// 			//int work_items = no_of_nodes;
		// 			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
					
		// 			_clMemcpyD2H(d_over,sizeof(char), &h_over);
		// 		}while(h_over);

		// 		//Update the h_graph_visited array
		// 		_clMemcpyD2H(d_graph_visited,no_of_nodes*sizeof(char), h_graph_visited);
		// 		//Update the h_level array
		// 		_clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// 	}
		// }

		_clFinish();
		// std::cout<<"Num iterations : "<<h_depth<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
			
#endif
		//--3 transfer data from device to host
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// std::cout<<"New depths are : "<<std::endl;
		// int max = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	std::cout<<i<<" : "<<h_level[i]<<", ";
		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
		// }
		// std::cout<<std::endl;
		// std::cout<<"Maximum depth seen is "<<max<<std::endl;

		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;	
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--breadth first search on GPUs - vertex pull
//----------------------------------------------------------
void run_bfs_gpu_vertex_pull(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int *h_reverse_neighbours, double *time_taken, char *h_graph_visited)
								throw(std::string) {
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;	
	}
	h_level[bfs_starting_node] = 0;
	
	cl_mem d_graph_nodes, d_graph_edges, d_over, d_depth, d_level, d_reverse_neighbours, d_graph_visited;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_reverse_neighbours = _clMalloc(edge_list_size*sizeof(int), h_reverse_neighbours);
		d_graph_visited = _clMallocRW(no_of_nodes*sizeof(char), h_graph_visited);
		
		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		_clMemcpyH2D(d_reverse_neighbours, edge_list_size*sizeof(int), h_reverse_neighbours);
		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer, iteration_timer;
		double kernel_time = 0.0;
		double iteration_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		//First run BFS once with level 0 for all nodes who dont have reverse neighbours
		h_depth = -1;
		
		do{
			iteration_timer.reset();
			iteration_timer.start();
			h_over = false;
			h_depth = h_depth + 1;
			if (!h_over)
				// std::cout<<"Start of iteration, h_over is "<<h_over<<", and h_depth is "<<h_depth<<std::endl;
			_clMemcpyH2D(d_over, sizeof(char), &h_over);
			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
			//--kernel 0
			int kernel_id = 5;
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
			_clSetArgs(kernel_id, kernel_idx++, d_over);
			_clSetArgs(kernel_id, kernel_idx++, d_depth);
			_clSetArgs(kernel_id, kernel_idx++, d_level);
			_clSetArgs(kernel_id, kernel_idx++, d_reverse_neighbours);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			// if (h_over)
			// 	std::cout<<"End of iteration, h_over is "<<h_over<<std::endl;
			// else
			// 	std::cout<<"End of do-while loop"<<std::endl;
			iteration_timer.stop();
			iteration_time = iteration_timer.getTimeInNanoSeconds();
			std::cout<<"Iteration "<<h_depth<<", time(ns) : "<<iteration_time<<std::endl;
		}while(h_over);

		// //Update the h_graph_visited array
		// _clMemcpyD2H(d_graph_visited,no_of_nodes*sizeof(char), h_graph_visited);
		// //Update the h_level array
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);

		// //Now run BFS for all the graph nodes which were disconnected from the above.
		// int num_of_starting_points = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	if (h_graph_visited[i] != true) {
		// 		num_of_starting_points++;
		// 		//Start BFS from the ith node
		// 		// std::cout<<"Going to run BFS again because node "<<i<<" has not been visited yet"<<std::endl;
		// 		h_level[i] = 0;
		// 		h_graph_visited[i] = true;
		// 		_clMemcpyH2D(d_graph_visited, no_of_nodes*sizeof(char), h_graph_visited);
		// 		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		// 		h_depth = -1;
				
		// 		do{
		// 			h_over = false;
		// 			h_depth = h_depth + 1;
		// 			// if (!h_over)
		// 			// 	std::cout<<"Start of iteration, h_over is "<<h_over<<", and h_depth is "<<h_depth<<std::endl;
		// 			_clMemcpyH2D(d_over, sizeof(char), &h_over);
		// 			_clMemcpyH2D(d_depth, sizeof(int), &h_depth);
		// 			//--kernel 0
		// 			int kernel_id = 5;
		// 			int kernel_idx = 0;
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
		// 			_clSetArgs(kernel_id, kernel_idx++, &no_of_nodes, sizeof(int));
		// 			_clSetArgs(kernel_id, kernel_idx++, d_over);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_depth);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_level);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_reverse_neighbours);
		// 			_clSetArgs(kernel_id, kernel_idx++, d_graph_visited);
					
		// 			//int work_items = no_of_nodes;
		// 			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
					
		// 			_clMemcpyD2H(d_over,sizeof(char), &h_over);
		// 			// if (h_over)
		// 			// 	std::cout<<"End of iteration, h_over is "<<h_over<<std::endl;
		// 			// else
		// 			// 	std::cout<<"End of do-while loop"<<std::endl;
		// 		}while(h_over);

		// 		//Update the h_graph_visited array
		// 		_clMemcpyD2H(d_graph_visited,no_of_nodes*sizeof(char), h_graph_visited);
		// 		//Update the h_level array
		// 		_clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// 	}
		// }
		// std::cout<<"Number of starting points : "<<num_of_starting_points<<std::endl;
		_clFinish();
		
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
			
#endif
		// _clMemcpyD2H(d_level,no_of_nodes*sizeof(int), h_level);
		// std::cout<<"New depths are : "<<std::endl;
		// int max = 0;
		// for (int i=0; i<no_of_nodes; i++) {
		// 	std::cout<<i<<" : "<<h_level[i]<<", ";
		// 	if (h_level[i] != INT_MAX && h_level[i]>max) max = h_level[i];
		// }
		// std::cout<<std::endl;
		// std::cout<<"Maximum depth seen is "<<max<<std::endl;


		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;	
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		_clFree(d_over);
		_clRelease();
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}

long read_and_return_no_of_nodes(char *filename) {
	long no_of_nodes = 0;
	ifstream fin;
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
	ifstream fin;
	fin.open(filename);
	std::string line;
	
	while (std::getline(fin, line)) no_of_edges++;

	return no_of_edges; 
}
//----------------------------------------------------------
//--cambine:	main function
//--author:		created by Jianbin Fang
//--date:		25/01/2011
//----------------------------------------------------------
int main(int argc, char * argv[])
{
	long no_of_nodes;
	int edge_list_size;
	FILE *fp;
	Node* h_graph_nodes;
	char *h_graph_mask, *h_updating_graph_mask, *h_graph_visited;
	char *input_fe, *input_fv;

	if (argc != 4) {
		printf("Usage is <starting_node> <edge-file> <vertex-file\n");
		exit(-1);
	}

	char *p;
	long conv = strtol(argv[1], &p, 10);

	// Check for errors: e.g., the string does not represent an integer
	// or the integer is larger than int
	if (errno != 0 || *p != '\0' || conv > INT_MAX) {
		printf("starting_node is invalid. exiting\n");
		exit(-1);
	} else {
		// No error
		bfs_starting_node = conv;    
		// printf("%d\n", num);
	}

	// bfs_starting_node = argv[1];
	input_fe = argv[2];
	input_fv = argv[3];

	try{
		/* For now, read the input files directly instead of reading from i/o*/
		// input_fe = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.e";
		// input_fv = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.v";

		// char *input_fe = "trisha-file.e";
		// char *input_fv = "trisha-file.v";
	
		// char *input_fe = "/home/tanand/rodinia_3.1/graph500-10-superconnected.e";
		
		no_of_nodes = read_and_return_no_of_nodes(input_fv);
		printf("Number of nodes read are : %d\n", no_of_nodes);
		edge_list_size = read_and_return_no_of_edges(input_fe);
		printf("Number of edges read are : %d\n", edge_list_size);
		printf("Starting node is : %d\n", bfs_starting_node);
		
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
		
		struct Edge* h_graph_edges = (struct Edge*) malloc(sizeof(struct Edge)*edge_list_size);
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

		//compute neighbours array for vertex push
		std::sort(h_graph_edges, h_graph_edges+edge_list_size, edge_compare);

		int* neighbours = (int *)malloc(edge_list_size * sizeof(int));

		int node_index = -1;
		for (int i=0; i < edge_list_size; i++) {
			// std::cout<<h_graph_edges[i].in_vertex<<", "<<h_graph_edges[i].out_vertex<<endl;
			if ((i==0) || (node_index != h_graph_edges[i].in_vertex)) {
				node_index = h_graph_edges[i].in_vertex;
				h_graph_nodes[node_index].starting = i;
			}
			neighbours[i] = h_graph_edges[i].out_vertex;
		}

		//compute reverse neighbours (parents) for vertex pull
		std::sort(h_graph_edges, h_graph_edges+edge_list_size, edge_compare_reverse);

		int* reverse_neighbours = (int *)malloc(edge_list_size * sizeof(int));

		node_index = -1;
		for (int i=0; i < edge_list_size; i++) {
			// std::cout<<h_graph_edges[i].out_vertex<<", "<<h_graph_edges[i].in_vertex<<endl;
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
		// run_bfs_gpu_rodinia(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);	
		int num_of_blocks = 1;
		int num_of_threads_per_block = edge_list_size;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(edge_list_size>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(edge_list_size/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		work_group_size = num_of_threads_per_block;
		
		std::cout<<"Edgelist Implementation"<<std::endl;
		// for (int i=0; i<6; i++)
			run_bfs_gpu_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, &time_taken);	
		std::cout<<std::endl<<"Reverse Edgelist Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
			run_bfs_gpu_reverse_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, &time_taken);

		num_of_blocks = 1;
		num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		work_group_size = num_of_threads_per_block;
		

		std::cout<<std::endl<<"Vertex Push Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
			run_bfs_gpu_vertex_push(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, neighbours, &time_taken, h_graph_visited);
		std::cout<<std::endl<<"Vertex Pull Implementation"<<std::endl;
		// for (int i=0; i<5; i++)	
			run_bfs_gpu_vertex_pull(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, reverse_neighbours, &time_taken, h_graph_visited);	
		//---------------------------------------------------------
		//--cpu entry
		// initalize the memory again
		// for(int i = 0; i < no_of_nodes; i++){
		// 	h_graph_mask[i]=false;
		// 	h_updating_graph_mask[i]=false;
		// 	h_graph_visited[i]=false;
		// }
		// //set the source node as true in the mask
		// source=0;
		// h_graph_mask[source]=true;
		// h_graph_visited[source]=true;
		// run_bfs_cpu(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost_ref);
		// //---------------------------------------------------------
		// //--result varification
		// compare_results<int>(h_cost_ref, h_cost, no_of_nodes);
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
