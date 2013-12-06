/* CSC 569 Final project
   Distributed k-means clustering
   Dec 10, 2013
   Ryan Staab
   Susan Marano
   Eric Buckthal
*/
#include <time.h>
#include <limits.h>
#include "kmeans.h"

/* Arrays for storing the read-in data */
   
float *Px;
float *Py;
float *Pz;
int length_data;


int main(int argc, char **argv) {

    /* Input error checking */

   if (argc != 3) {
      printf("Usage: kmeans <file1> <k>\n");
      return BAD_ARGS;
   }

   /* Initialization */
   int number_processes;
   int rank;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

   printf("I am (%d)\n", rank);

   if (rank == ROOT)
      printf("...Initialized...\n");
   
   length_data = 0;
   int send_element_count = 0;
   
   /* Read in data */
   
   if (rank == ROOT)
      length_data = vectorSize(argv[1]);
   
   /* Broadcast the length of the data to all nodes */
   
   MPI_Bcast(&length_data, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   /* Arrays for storing the read-in data */
   
   Px = (float *)calloc(sizeof(float), length_data);
   Py = (float *)calloc(sizeof(float), length_data);
   Pz = (float *)calloc(sizeof(float), length_data);
   
   /* Read in the data */
   
   if (rank == ROOT)
      fileRead(argv[1], Px, Py, Pz, length_data);
   
   /* Calculate number elements to send to each node */
   
   send_element_count = (length_data / number_processes);
   
   //TODO: handle what to do when num_elements has a remainder
   
   float *recvbufPx = (float *)calloc(sizeof(float), send_element_count);   
   float *recvbufPy = (float *)calloc(sizeof(float), send_element_count);   
   float *recvbufPz = (float *)calloc(sizeof(float), send_element_count);
   
   /* Scatter the data */
   
   MPI_Scatter(Px, send_element_count, MPI_FLOAT, recvbufPx, 
                    send_element_count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
                    
   MPI_Scatter(Py, send_element_count, MPI_FLOAT, recvbufPy, 
                    send_element_count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
                   
   MPI_Scatter(Pz, send_element_count, MPI_FLOAT, recvbufPz, 
                    send_element_count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
                    
   if (rank == ROOT)
      printf("...Data scattered...\n");
   
   /* k is a command line argument so we can run it with multiple vals ourselves */
   
   int k;
   if (rank == ROOT) {
      k = atoi(argv[2]); //TODO: error check
   }
   
   /* Broadcast the value of k */
   
   MPI_Bcast(&k, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   float *Cx = (float *)calloc(sizeof(float), k);
   float *Cy = (float *)calloc(sizeof(float), k);
   float *Cz = (float *)calloc(sizeof(float), k);
   float *Cxold;
   float *Cyold;
   float *Czold;
   int *final_assignments;
   int *assignments = (int *)calloc(sizeof(int), send_element_count);
   int i;   
   
   
   if (rank == ROOT) {
      final_assignments = (int *)calloc(sizeof(int), length_data);
      Cxold = (float *)calloc(sizeof(float), k);
      Cyold = (float *)calloc(sizeof(float), k);
      Czold = (float *)calloc(sizeof(float), k);
   
      //initialize the k cluster centers to random points from the data
      int r;
      
      srand(time(NULL));
      for (i = 0; i < k; i++) {
         r = rand() % length_data; //TODO: check if same number twice
         Cxold[i] = Cx[i] = Px[r];
         Cyold[i] = Cy[i] = Py[r];
         Czold[i] = Cz[i] = Pz[r];
      }
   }
   
   if (rank == ROOT)
      printf("...Clusters initialized...\n");
   
   int changed = 0;
   int *num_assigned = (int *)calloc(sizeof(int), k);
   
   
   
   do {
      //Bcast the cluster centers
      MPI_Bcast(Cx, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cy, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cz, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

      if (rank == ROOT)
         printf("...Broadcasted centers...\n");

      // do actual clustering assignments with cuda stuff
      cudaAssign(Cx, Cy, Cz, recvbufPx, recvbufPy, recvbufPz, assignments, send_element_count, k);
      
      if (rank == ROOT)
         printf("...Cuda finished...\n");
      
      //Each node has a three k-length array, each cell represents the sums of its assigned data vals, for x, y, z
      
      float *Cxtemp = (float *)calloc(sizeof(float), k);
      float *Cytemp = (float *)calloc(sizeof(float), k);
      float *Cztemp = (float *)calloc(sizeof(float), k);
      int *num_assigned_temp = (int *)calloc(sizeof(int), k);
      
      if (rank == ROOT)
         printf("...Calloced...\n");
      
      /* Calculate local new cluster center means */
      
      for (i = 0; i < send_element_count; i++) {
      
         Cxtemp[assignments[i]] += recvbufPx[i];
         Cytemp[assignments[i]] += recvbufPy[i];
         Cztemp[assignments[i]] += recvbufPz[i];
         num_assigned_temp[assignments[i]]++;
      }
      
      if (rank == ROOT)
         printf("...Assignments counted...\n");
     
      
      /* Reduce (sum) the local cluster means onto root */
      
      MPI_Reduce(Cxtemp, Cx, k, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      MPI_Reduce(Cytemp, Cy, k, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      MPI_Reduce(Cztemp, Cz, k, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      MPI_Reduce(num_assigned_temp, num_assigned, k, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      
      if (rank == ROOT)
         printf("...Reduced...\n");
      
      /* As root calculate final cluster centers */
      
      
      
      if (rank == ROOT) {
         
         for (i = 0; i < k; i++) {
            Cx[i] /= num_assigned[i];
            Cy[i] /= num_assigned[i];
            Cz[i] /= num_assigned[i];
         }
      }
      
      if (rank == ROOT)
         printf("...Freeing...\n");
      
      free(Cxtemp); free(Cytemp); free(Cztemp);
      free(num_assigned_temp);
      
      if (rank == ROOT)
         printf("...Freed...\n");
      
      /* Check if cluster means changed, and update old */
      
      if (rank == ROOT) {
         changed = centersChanged(Cxold, Cyold, Czold, Cx, Cy, Cz, k);
         for (i = 0; i < k; i++) {
            Cxold[i] = Cx[i];
            Cyold[i] = Cy[i];
            Czold[i] = Cz[i];
         }
      }
      
      MPI_Bcast(&changed, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   /* Continue the loop while cluster centers are not same as before */
   
      MPI_Barrier(MPI_COMM_WORLD);
   
   } while (changed);
   
   if (rank == ROOT)
      printf("...Finished clustering...\n");
      
   /* Send final assignments all to master for visualization */
   
   MPI_Gather(assignments, send_element_count, MPI_INT, final_assignments, 
                    send_element_count, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   /* Free everything */
   
   free(recvbufPx); free(recvbufPy); free(recvbufPz);
   free(num_assigned);
   free(assignments);
   
   if (rank == ROOT) {
      free(Cxold); free(Cyold); free(Czold);
   }
   
   //TODO: do the opengl stuff and output 3d visualization
   
   free(Cx); free(Cy); free(Cz);
   if (rank == ROOT) {
      free(Px); free(Py); free(Pz);
      free(final_assignments);
   }
   
   MPI_Finalize();
}
