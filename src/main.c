 /**
 * @file main.c
 * @brief This file provides you with the original implementation of pagerank.
 * Your challenge is to optimise it using OpenMP and/or MPI.
 * @author Ludovic Capelli (Ludovic.Capelli@csiro.au)
 * @coordinator Jolanta Zjupa (j.zjupa@fz-juelich.de)
 **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <omp.h>
#include <mpi.h>

/// The number of vertices in the graph.
#define GRAPH_ORDER 1000
/// Parameters used in pagerank convergence, do not change.
#define DAMPING_FACTOR 0.85
/// The number of seconds to not exceed for the calculation loop.
#define MAX_TIME 10

/**
 * @brief Indicates which vertices are connected.
 * @details If an edge links vertex A to vertex B, then adjacency_matrix[A][B]
 * will be 1.0. The absence of edge is represented with value 0.0.
 * Redundant edges are still represented with value 1.0.
 */
double adjacency_matrix[GRAPH_ORDER][GRAPH_ORDER];
double max_diff = 0.0;
double min_diff = 1.0;
double total_diff = 0.0;

void initialize_graph(void)
{
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            adjacency_matrix[i][j] = 0.0;
        }
    }
}

/**
 * @brief Calculates the pagerank of all vertices in the graph.
 * @param pagerank The array in which store the final pageranks.
 */
void calculate_pagerank(double pagerank[], int start_i, int end_i, int rank_length, int rank)
{
    double initial_rank = 1.0 / GRAPH_ORDER;

    // Initialise all vertices to 1/n.
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        pagerank[i] = initial_rank;
    }

    double damping_value = (1.0 - DAMPING_FACTOR) / GRAPH_ORDER;
    double diff = 1.0;
    size_t iteration = 0;
    double start = MPI_Wtime();//omp_get_wtime();
    double elapsed = MPI_Wtime();//omp_get_wtime() - start;
    double time_per_iteration = 0;
    double new_pagerank[GRAPH_ORDER];
    for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)
    {
        new_pagerank[i] = 0.0;
    }

    // If we exceeded the MAX_TIME seconds, we stop. If we typically spend X seconds on an iteration, and we are less than X seconds away from MAX_TIME, we stop.
    while(elapsed < MAX_TIME && (elapsed + time_per_iteration) < MAX_TIME)
    {
        double iteration_start = MPI_Wtime();//omp_get_wtime();

        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = 0.0; //memset fill array with 0 at once without loop
        }

        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)
        {
            for(int j = 0; j < GRAPH_ORDER; j++)
            {
                if (adjacency_matrix[j][i] == 1.0)
                {
                    int outdegree = 0;

                    for(int k = 0; k < GRAPH_ORDER; k++)
                    {
                        if (adjacency_matrix[j][k] == 1.0)
                        {
                            outdegree++;
                        }
                    }
                    new_pagerank[i] += pagerank[j] / (double)outdegree;
                }
            }
        }

        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)
        {
            new_pagerank[i] = DAMPING_FACTOR * new_pagerank[i] + damping_value;
        }

        diff = 0.0;
        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)  //this can be done per node separately again
        {
            diff += fabs(new_pagerank[i] - pagerank[i]);
        }
        //---Add my own stuff here---
        double summed_diff;
        MPI_Reduce(&diff, &summed_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
          diff = summed_diff;
          //---//
          max_diff = (max_diff < diff) ? diff : max_diff;
          total_diff += diff;
          min_diff = (min_diff > diff) ? diff : min_diff;
        }
        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)  // this can be done per node separately again
        {
            pagerank[i] = new_pagerank[i];
        }
        //---Add my own stuff here---
        //double gathered_pagerank[GRAPH_ORDER];
        //MPI_Allgather(&pagerank[rank*rank_length], rank_length, MPI_DOUBLE, gathered_pagerank, rank_length, MPI_DOUBLE, MPI_COMM_WORLD);// point to start of changed pagerank instead or instead use MPI_IN_PLACE
        MPI_Allgather(MPI_IN_PLACE, rank_length, MPI_DOUBLE, pagerank, rank_length, MPI_DOUBLE, MPI_COMM_WORLD);
        //pagerank = gathered_pagerank;
        //---//

        double pagerank_total = 0.0;
        for(int i = start_i; i <= end_i; i++) //0; i < GRAPH_ORDER; i++)  // this can be done per node separately again
        {
            pagerank_total += pagerank[i];
        }
        //---Add my own stuff here---
        double summed_pagerank_total;
        // receive all pagerank_total at rank 0, sum up and check for sum==1 -> MPI_Reduce with MPI_SUM as operator
        MPI_Reduce(&pagerank_total, &summed_pagerank_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        pagerank_total = summed_pagerank_total;
        //---//
        if (rank == 0) {
          if(fabs(pagerank_total - 1.0) >= 1E-12)
          {
            printf("[ERROR] Iteration %zu: sum of all pageranks is not 1 but %.12f.\n", iteration, pagerank_total);
          }
        }
  		  double iteration_end = MPI_Wtime();//omp_get_wtime();
	  	  elapsed = MPI_Wtime() - start;//omp_get_wtime() - start;
		    iteration++;
		    time_per_iteration = elapsed / iteration;

    }

    //---Add my own stuff here---
    //---//
    printf("%zu iterations achieved in %.2f seconds\n", iteration, elapsed);
    
}

/**
 * @brief Populates the edges in the graph for testing.
 **/
void generate_nice_graph(void)
{
    printf("Generate a graph for testing purposes (i.e.: a nice and conveniently designed graph :) )\n");
    double start = MPI_Wtime();//omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", MPI_Wtime() - start);//omp_get_wtime() - start);
}

/**
 * @brief Populates the edges in the graph for the challenge.
 **/
void generate_sneaky_graph(void)
{
    printf("Generate a graph for the challenge (i.e.: a sneaky graph :P )\n");
    double start = MPI_Wtime();//omp_get_wtime();
    initialize_graph();
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        for(int j = 0; j < GRAPH_ORDER - i; j++)
        {
            int source = i;
            int destination = j;
            if(i != j)
            {
                adjacency_matrix[source][destination] = 1.0;
            }
        }
    }
    printf("%.2f seconds to generate the graph.\n", MPI_Wtime() - start);//omp_get_wtime() - start);
    
}

int main(int argc, char* argv[])
{
    // We do not need argc, this line silences potential compilation warnings.
    (void) argc;
    // We do not need argv, this line silences potential compilation warnings.
    (void) argv;

    //printf("This program has two graph generators: generate_nice_graph and generate_sneaky_graph. If you intend to submit, your code will be timed on the sneaky graph, remember to try both.\n");

    // Get the time at the very start.
    double start = MPI_Wtime();//omp_get_wtime();

    //---Add my own stuff here---
    int numtasks, rank, start_idx, end_idx;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    //---//

    //generate_nice_graph();
    generate_sneaky_graph();

    //---Add my own stuff here---
    int rank_length = GRAPH_ORDER / numtasks; 
    int remains = GRAPH_ORDER % numtasks;
    if (rank == numtasks - 1) {
      start_idx = rank_length * rank;
      end_idx = rank_length * rank + rank_length + remains - 1;
    } else {
      start_idx = rank_length * rank;
      end_idx = rank_length * rank + rank_length - 1;
    }
    //---//

    /// The array in which each vertex pagerank is stored.
    double pagerank[GRAPH_ORDER];
    calculate_pagerank(pagerank, start_idx, end_idx, rank_length, rank);

    // Calculates the sum of all pageranks. It should be 1.0, so it can be used as a quick verification.
    double sum_ranks = 0.0;
    for(int i = 0; i < GRAPH_ORDER; i++)
    {
        if(i % 100 == 0)
        {
            if (rank == 0) printf("PageRank of vertex %d: %.6f\n", i, pagerank[i]);
        }
        sum_ranks += pagerank[i];
    }
    printf("Sum of all pageranks = %.12f, total diff = %.12f, max diff = %.12f and min diff = %.12f.\n", sum_ranks, total_diff, max_diff, min_diff);
    double end = MPI_Wtime();//omp_get_wtime();

    printf("Total time taken: %.2f seconds.\n", end - start);

    //---Add my own stuff here---
    MPI_Finalize();
    //---//

    return 0;
}
