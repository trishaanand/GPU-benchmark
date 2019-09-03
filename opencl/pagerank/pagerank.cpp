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
#define NUM_ITERATIONS 50

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};

struct Edge {
	int in_vertex;
	int out_vertex;
};

bool edge_compare(Edge lhs, Edge rhs) {
	return (lhs.in_vertex < rhs.in_vertex);
}

bool edge_compare_reverse(Edge lhs, Edge rhs) {
	return (lhs.out_vertex < rhs.out_vertex);
}


//----------------------------------------------------------
//--Pagerank on GPUs - edgelist
//----------------------------------------------------------
void run_pagerank_gpu_edgelist(int no_of_nodes, Node *h_graph_nodes, int edge_list_size, \
		Edge *h_graph_edges, char *h_graph_mask, char *h_updating_graph_mask, \
		char *h_graph_visited, bool reverse, double* time_taken)
					throw(std::string){

	//int number_elements = height*width;
	// std::cout<<"Reached the function call"<<std::endl;
	float *h_pagerank = (float *) malloc (no_of_nodes*sizeof(float));
	float *h_pagerank_new = (float *) malloc (no_of_nodes*sizeof(float));
	float *temp;
	for (int i=0; i< no_of_nodes; i++) {
		h_pagerank[i] = 0.25;
		h_pagerank_new[i] = 0.25;
	}
	
	cl_mem d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, \
			d_graph_visited, d_pagerank, d_pagerank_new;
	try{
		// std::cout<<"Before transfer of data to the device."<<std::endl;
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		
		d_pagerank = _clMallocRW(no_of_nodes*sizeof(float), h_pagerank);
		d_pagerank_new = _clMallocRW(no_of_nodes*sizeof(float), h_pagerank_new);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_pagerank, no_of_nodes*sizeof(float), h_pagerank);
		_clMemcpyH2D(d_pagerank_new, no_of_nodes*sizeof(float), h_pagerank_new);
			
		// std::cout<<"After transfer of data to the device."<<std::endl;

		int i = 0;
		int j=0;

		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		// std::cout<<"Before starting the iterations of pagerank"<<std::endl;
		while(i < NUM_ITERATIONS){
			// std::cout<<"Inside the iteration. Before copy of pagerank to the device."<<std::endl;
			
			_clMemcpyH2D(d_pagerank, no_of_nodes*sizeof(float), h_pagerank);

			int kernel_id;
			if (reverse == true) {
				kernel_id = 3;
			} else {
				kernel_id = 2;
			}
			int kernel_idx = 0;
			_clSetArgs(kernel_id, kernel_idx++, d_graph_nodes);
			_clSetArgs(kernel_id, kernel_idx++, d_graph_edges);
			_clSetArgs(kernel_id, kernel_idx++, &edge_list_size, sizeof(int));
			_clSetArgs(kernel_id, kernel_idx++, d_pagerank);
			_clSetArgs(kernel_id, kernel_idx++, d_pagerank_new);
			
			// std::cout<<"Before invoking the kernel"<<std::endl;
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, edge_list_size, work_group_size);
			
			// std::cout<<"After invoking the kernel, before copying new pagerank back"<<std::endl;
			_clMemcpyD2H(d_pagerank_new, no_of_nodes*sizeof(float), h_pagerank_new);	
			// _clFinish();
			// std::cout<<"After the copy of new page rank back to the host. Printing new pageranks now"<<std::endl;
			// for (j=0; j<no_of_nodes;j++) {
			// 	std::cout<<j<<" : "<<h_pagerank_new[j]<<", ";
			// }
			// std::cout<<std::endl<<"Now copying new pagerank into the old one"<<std::endl;
			for (j=0; j<no_of_nodes; j++) {
				h_pagerank[j] = h_pagerank_new[j];
				// std::cout<<"Successful copy for iteration "<<j<<std::endl;
			} 
			i++;
			// temp = h_pagerank;
			// h_pagerank = h_pagerank_new;
			// h_pagerank_new = temp;
			// std::cout<<"End of iteration. new page rank has been copied into the old pagerank. "<<std::endl;
			}
			
		_clFinish();
		// std::cout<<"Num iterations : "<<h_depth<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInSeconds();
		*time_taken = kernel_time;
#endif
		//--3 transfer data from device to host
		_clMemcpyD2H(d_pagerank_new,no_of_nodes*sizeof(float), h_pagerank_new);
		std::cout<<"New page ranks are "<<std::endl;
		for (j=0; j<no_of_nodes;j++) {
			std::cout<<j<<" : "<<h_pagerank_new[j]<<", ";
		}
		std::cout<<std::endl;
		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		//--4 release cl resources.
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		// _clFree(d_graph_mask);
		// _clFree(d_updating_graph_mask);
		// _clFree(d_graph_visited);
		// _clFree(d_over);
		_clRelease();
	}
	catch(std::string msg){		
		_clFree(d_graph_nodes);
		_clFree(d_graph_edges);
		// _clFree(d_graph_mask);
		// _clFree(d_updating_graph_mask);
		// _clFree(d_graph_visited);
		// _clFree(d_over);
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
void run_bfs_gpu_vertex_push(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int * h_neighbours, double *time_taken)
								throw(std::string) {
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;
	}
	h_level[0] = 0; //Setting source as zero -> Assumption that zeroth index is the source. 
	//Change in case the source is read from the input

	cl_mem d_graph_nodes, d_graph_edges, d_over, d_depth, d_level, d_neighbours;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_neighbours = _clMalloc(edge_list_size*sizeof(int), h_neighbours);
		
		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		_clMemcpyH2D(d_neighbours, edge_list_size*sizeof(int), h_neighbours);
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		do{
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
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			}while(h_over);
			
		_clFinish();
		std::cout<<"Num iterations : "<<h_depth<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInSeconds();
		*time_taken = kernel_time;
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
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
void run_bfs_gpu_vertex_pull(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int * h_reverse_neighbours, double *time_taken)
								throw(std::string) {
	int h_depth = -1;
	char h_over;

	int *h_level = (int *) malloc (no_of_nodes*sizeof(int)); //store the current minimum depth seen by a node
	for (int i=0; i< no_of_nodes; i++) {
		h_level[i] = INT_MAX;
	}
	h_level[0] = 0; //Setting source as zero -> Assumption that zeroth index is the source. 
	//Change in case the source is read from the input

	/* Test code to delete */
	for (int i=0; i<edge_list_size; i++) {
		
	}

	cl_mem d_graph_nodes, d_graph_edges, d_over, d_depth, d_level, d_reverse_neighbours;
	try{
		//--1 transfer data from host to device
		_clInit();	
		d_graph_nodes = _clMalloc(no_of_nodes*sizeof(Node), h_graph_nodes);
		d_graph_edges = _clMalloc(edge_list_size*sizeof(Edge), h_graph_edges);
		d_reverse_neighbours = _clMalloc(edge_list_size*sizeof(int), h_reverse_neighbours);
		
		d_over = _clMallocRW(sizeof(char), &h_over);
		d_depth = _clMallocRW(sizeof(int), &h_depth);

		d_level = _clMallocRW(no_of_nodes*sizeof(int), h_level);
		
		_clMemcpyH2D(d_graph_nodes, no_of_nodes*sizeof(Node), h_graph_nodes);
		_clMemcpyH2D(d_graph_edges, edge_list_size*sizeof(Edge), h_graph_edges);	
		_clMemcpyH2D(d_level, no_of_nodes*sizeof(int), h_level);
		_clMemcpyH2D(d_reverse_neighbours, edge_list_size*sizeof(int), h_reverse_neighbours);
			
		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		do{
			h_over = false;
			h_depth = h_depth + 1;
			
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
			
			//int work_items = no_of_nodes;
			_clInvokeKernel(kernel_id, no_of_nodes, work_group_size);
			
			_clMemcpyD2H(d_over,sizeof(char), &h_over);
			}while(h_over);
			
		_clFinish();
		std::cout<<"Num iterations : "<<h_depth<<std::endl;
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInSeconds();
		*time_taken = kernel_time;
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
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
	try{
		/* For now, read the input files directly instead of reading from i/o*/
		char *input_fe = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.e";
		char *input_fv = "/var/scratch/alvarban/BSc_2k19/graphs/G500/graph500-10.v";
	
		no_of_nodes = read_and_return_no_of_nodes(input_fv);
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

		int num_of_blocks = 1;
		int num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK){
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}
		work_group_size = num_of_threads_per_block;
		// allocate host memory
		h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		for (int i=0; i < no_of_nodes; i++) {
			h_graph_nodes[i].no_of_edges = 0;
			h_graph_nodes[i].starting = -1;
		}
		h_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_updating_graph_mask = (char*) malloc(sizeof(char)*no_of_nodes);
		h_graph_visited = (char*) malloc(sizeof(char)*no_of_nodes);
	
		int start, edgeno;   
		
		struct Edge* h_graph_edges = (struct Edge*) malloc(sizeof(struct Edge)*edge_list_size);
		int neighbour_index = 0;
		for(int i=0; i < edge_list_size ; i++){
			int in_index, out_index;
			float cost; //for datagen
			fscanf(fp, "%d", &in_index);
			fscanf(fp, "%d", &out_index);
			fscanf(fp, "%f", &cost); //only for datagen - delete for others
			h_graph_edges[i].in_vertex = in_index;
			h_graph_edges[i].out_vertex = out_index;
			//Update the number of neighbours of the node with index in_index;
			h_graph_nodes[in_index].no_of_edges++;
		}

		//compute neighbours array for vertex push
		std::sort(h_graph_edges, h_graph_edges+edge_list_size, edge_compare);

		int* neighbours = (int *)malloc(edge_list_size * sizeof(int));

		int node_index = -1;
		for (int i=0; i < edge_list_size; i++) {
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
			if ((i==0) || (node_index != h_graph_edges[i].out_vertex)) {
				node_index = h_graph_edges[i].out_vertex;
				h_graph_nodes[node_index].starting = i;
			}
			reverse_neighbours[i] = h_graph_edges[i].in_vertex;
		}

		if(fp)
			fclose(fp);    
		double time_taken = 0;
		//---------------------------------------------------------
		//--gpu entry
		// run_bfs_gpu_rodinia(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost);	
		std::cout<<"Edgelist Implementation"<<std::endl;
		for (int i=0; i<1; i++)
		run_pagerank_gpu_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, false, &time_taken);	
		// std::cout<<"Reverse Edgelist Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
		// run_bfs_gpu_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, true, &time_taken);	
		// std::cout<<"Vertex Push Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
		// run_bfs_gpu_vertex_push(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, neighbours, &time_taken);
		// std::cout<<"Vertex Pull Implementation"<<std::endl;
		// for (int i=0; i<5; i++)
		// run_bfs_gpu_vertex_pull(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, reverse_neighbours, &time_taken);	
		
		
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
