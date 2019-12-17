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
#include <omp.h>
#include <math.h>
#include <fstream>
using namespace std;

#ifdef  PROFILING
#include "timer.h"
#endif

// #include "CLHelper.h"
// #include "util.h"

#define MAX_THREADS_PER_BLOCK 512
#define NUM_ITERATIONS 50

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
	for (int i=0; i< no_of_nodes; i++) {
		h_pagerank[i] = 0.25;
		h_pagerank_new[i] = 0.0;
	}
	
	try{
		//--1 transfer data from host to device
			
		int i = 0;
		int j=0;

		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		for (i = 0; i < NUM_ITERATIONS; i++) {
			int kernel_id;
			int kernel_idx;
			//Initialize the d_pagerank and d_pagerank_new arrays before invoking the kernel
			//Not required for the first iteration
			if (i!=0) {
				#pragma omp parallel for
				for (int itr = 0 ; itr < no_of_nodes; itr++) {
					h_pagerank[itr] = h_pagerank_new[itr];
				}
				
			}

			#pragma omp parallel for shared(h_pagerank_new)
			for (j=0; j<edge_list_size; j++) {
				Edge current_edge = h_graph_edges[j];

				float new_rank = 0.0f;
				int in_vertex = current_edge.in_vertex;
				int out_vertex = current_edge.out_vertex;
				int degree = h_graph_nodes[in_vertex].no_of_edges;
				
				if (degree != 0) new_rank = h_pagerank[in_vertex] / degree;

				#pragma omp atomic
				h_pagerank_new[out_vertex] += new_rank;
			}
		}
			

		
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
#endif
		//--3 transfer data from device to host
		// _clMemcpyD2H(d_pagerank_new,no_of_nodes*sizeof(float), h_pagerank_new);
		// std::cout<<"New page ranks are "<<std::endl;
		// // if (h_pagerank_new[j] != 0) {
		// for (j=0; j<9; j++) {
		// 	std::cout<<j<<" : "<<h_pagerank_new[j]<<", ";
		// }
		// std::cout<<std::endl;
		//--statistics
#ifdef	PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;		
#endif
		
	}
	catch(std::string msg){		
		
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--pagerank - vertex push
//----------------------------------------------------------
void run_pagerank_gpu_vertex_push(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int *h_neighbours, double *time_taken)
								throw(std::string) {
	float *h_pagerank = (float *) malloc (no_of_nodes*sizeof(float));
	float *h_pagerank_new = (float *) malloc (no_of_nodes*sizeof(float));
	float *temp;
	for (int i=0; i< no_of_nodes; i++) {
		h_pagerank[i] = 0.25;
		h_pagerank_new[i] = 0.0;
	}
	
	try{
		//--1 transfer data from host to device
		
		int i = 0;
		int j=0;

		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		for (i = 0; i < NUM_ITERATIONS; i++) {
			//Initialize the d_pagerank and d_pagerank_new arrays before invoking the kernel
			//Not required for the first iteration
			if (i!=0) {
				#pragma omp parallel for
				for (int itr = 0 ; itr < no_of_nodes; itr++) {
					h_pagerank[itr] = h_pagerank_new[itr];
				}
			}

			#pragma omp parallel for shared(h_pagerank_new)
			for (j=0; j<no_of_nodes; j++) {
				Node current_node = h_graph_nodes[j];
				float new_rank = 0.0f;
				int degree = current_node.no_of_edges;
				if (degree!=0) 
					new_rank = h_pagerank[j] / degree;

				int starting = current_node.starting;
				int max = starting + current_node.no_of_edges;
				for (int itr=starting; itr<max; itr++ ) {
					#pragma omp atomic
					h_pagerank_new[h_neighbours[itr]] += new_rank;
				}
			}
		}
		
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
			
#endif
		

#ifdef  PROFILING
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;	
#endif

		//--4 release cl resources.
		
	}
	catch(std::string msg){		
		
		std::string e_str = "in run_transpose_gpu -> ";
		e_str += msg;
		throw(e_str);
	}
	return ;
}

//----------------------------------------------------------
//--pagerank - vertex pull
//----------------------------------------------------------
void run_pagerank_gpu_vertex_pull(int no_of_nodes, Node* h_graph_nodes, int edge_list_size, Edge *h_graph_edges, int *h_reverse_neighbours, double *time_taken)
								throw(std::string) {
	float *h_pagerank = (float *) malloc (no_of_nodes*sizeof(float));
	float *h_pagerank_new = (float *) malloc (no_of_nodes*sizeof(float));
	float *temp;
	for (int i=0; i< no_of_nodes; i++) {
		h_pagerank[i] = 0.25;
		h_pagerank_new[i] = 0.0;
	}
	
	try{
		//--1 transfer data from host to device
		
		int i = 0;
		int j=0;

		//--2 invoke kernel
#ifdef	PROFILING
		timer kernel_timer;
		double kernel_time = 0.0;		
		kernel_timer.reset();
		kernel_timer.start();
#endif
		for (i = 0; i < NUM_ITERATIONS; i++) {
			//Initialize the d_pagerank and d_pagerank_new arrays before invoking the kernel
			//Not required for the first iteration
			if (i!=0) {
				#pragma omp parallel for
				for (int itr = 0 ; itr < no_of_nodes; itr++) {
					h_pagerank[itr] = h_pagerank_new[itr];
				}
			}

			#pragma omp parallel for
			for (j=0; j<no_of_nodes; j++) {

				//initialize to the last page rank seen by the vertex
				float new_rank = 0.0f; 
				Node current_node = h_graph_nodes[j];
				int starting = current_node.reverse_starting;
				int max = starting + current_node.no_of_reverse_edges;
				for (int i=starting; i<max; i++ ) {
					int neighbour_index = h_reverse_neighbours[i];
					Node neighbour_node = h_graph_nodes[neighbour_index];

					int degree = neighbour_node.no_of_edges;
					if (degree != 0) new_rank += (h_pagerank[neighbour_index])/degree;
				}

				h_pagerank_new[j] += new_rank;
			}
		}
		
#ifdef	PROFILING
		kernel_timer.stop();
		kernel_time = kernel_timer.getTimeInNanoSeconds();
		*time_taken = kernel_time;
	
		std::cout<<"kernel time(s):"<<kernel_time<<std::endl;	
#endif

		//--4 release cl resources.
		
	}
	catch(std::string msg){		
		
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

	if (argc != 3) {
		printf("Usage is <edge-file> <vertex-file\n");
		exit(-1);
	}

	input_fe = argv[1];
	input_fv = argv[2];

	try{
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
	
		int start, edgeno;   
		
		struct Edge* h_graph_edges = (struct Edge*) malloc(sizeof(struct Edge)*edge_list_size);
		struct Edge* h_graph_edges_copy = (struct Edge*) malloc(sizeof(struct Edge)*edge_list_size);
		int neighbour_index = 0;
		for(int i=0; i < edge_list_size ; i++){
			int in_index, out_index;
			float cost; //for datagen
			fscanf(fp, "%d", &in_index);
			fscanf(fp, "%d", &out_index);
			// fscanf(fp, "%f", &cost); //only for datagen - delete for others
			h_graph_edges[i].in_vertex = in_index;
			h_graph_edges[i].out_vertex = out_index;
			
			h_graph_edges_copy[i].in_vertex = in_index;
			h_graph_edges_copy[i].out_vertex = out_index;
			
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

		//reset the graph edge array
		for (int i=0; i < edge_list_size; i++) {
			h_graph_edges[i].in_vertex = h_graph_edges_copy[i].in_vertex;
			h_graph_edges[i].out_vertex = h_graph_edges_copy[i].out_vertex;
		}
		
		if(fp)
			fclose(fp);    
		double time_taken = 0;
		//---------------------------------------------------------
		//--gpu entry

		int num_of_blocks = 1;
		int num_of_threads_per_block = edge_list_size;

		//Make execution Parameters according to the number of edges
		//Distribute threads across multiple Blocks if necessary
		
		std::cout<<endl<<"Edgelist Implementation"<<std::endl;
		for (int i=0; i<5; i++)
			run_pagerank_gpu_edgelist(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, h_graph_mask, h_updating_graph_mask, h_graph_visited, false, &time_taken);	
		
		std::cout<<endl<<"Vertex Push Implementation"<<std::endl;
		for (int i=0; i<5; i++)
			run_pagerank_gpu_vertex_push(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, neighbours, &time_taken);
		
		std::cout<<endl<<"Vertex Pull Implementation"<<std::endl;
		for (int i=0; i<5; i++)
			run_pagerank_gpu_vertex_pull(no_of_nodes,h_graph_nodes,edge_list_size,h_graph_edges, reverse_neighbours, &time_taken);
		
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
