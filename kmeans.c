/* CSC 569 Final project
   Distributed k-means clustering
   Dec 10, 2013
   Ryan Staab
   Susan Marano
   Eric Buckthal
*/
#include <mpi.h>
#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"

int main(int argc, char **argv) {

    /* Input error checking */

	//printf("argc: %d\n", argc);
   if (argc < 3) {
      printf("Usage: kmeans <file1> <k>\n");
      return BAD_ARGS;
   }

   /* Initialization */
   int number_processes;
   int rank;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

   //printf("I am (%d)\n", rank);

   if (rank == ROOT) {
      //printf("...Initialized...\n");
   }
   
   length_data = 0;
   int send_element_count = 0;
   
   /* Read in data */
   
   if (rank == ROOT) {
      //printf("reading file\n");
      length_data = vectorSize(argv[1]);
      //printf("file read %d\n", length_data);
   }
   
   /* Broadcast the length of the data to all nodes */
   
   MPI_Bcast(&length_data, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   /* Arrays for storing the read-in data */
   
   Px = (float *)calloc(sizeof(float), length_data);
   Py = (float *)calloc(sizeof(float), length_data);
   Pz = (float *)calloc(sizeof(float), length_data);
   
   /* Read in the data */
   
   if (rank == ROOT) {
      //printf("getting vector\n");
      fileRead(argv[1], Px, Py, Pz, length_data);


      float xmax = Px[0];
      float ymax = Py[0];
      float zmax = Pz[0]; 
      float xmin = Px[0];
      float ymin = Py[0];
      float zmin = Pz[0];

      int j;
      for(j=1; j<length_data; j++){
         if(Px[j] > xmax)
            xmax = Px[j];
         else if(Px[j] < xmin)
            xmin = Px[j];

         if(Py[j] > ymax)
            ymax = Py[j];
         else if(Py[j] < ymin)
            ymin = Py[j];
         
         if(Pz[j] > zmax)
            zmax = Pz[j];
         else if(Pz[j] < zmin)
            zmin = Pz[j];
      }

      //printf("%f %f\n", xmax, xmin);
         
      for(j=0; j<length_data; j++){
         Px[j] = (((Px[j]-xmin)* 10)/(xmax-xmin)) - 5;
         Py[j] = (((Py[j]-ymin)* 10)/(ymax-ymin)) - 5;
         Pz[j] = (((Pz[j]-zmin)* 10)/(zmax-zmin)) - 5;

         //printf("%f %f %f\n", Px[j], Py[j], Pz[j]);
      }
   }
   
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
      //printf("...Data scattered...\n");
   
   /* k is a command line argument so we can run it with multiple vals ourselves */
   
   if (rank == ROOT) {
      k_total = atoi(argv[2]); //TODO: error check
   }
   
   /* Broadcast the value of k */
   
   MPI_Bcast(&k_total, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   float *Cx = (float *)calloc(sizeof(float), k_total);
   float *Cy = (float *)calloc(sizeof(float), k_total);
   float *Cz = (float *)calloc(sizeof(float), k_total);
   float *Cxold;
   float *Cyold;
   float *Czold;
   int *assignments = (int *)calloc(sizeof(int), send_element_count);
   
   int i;

   int a_per_i_length = 110;
   
   if (rank == ROOT) {
      final_assignments = (int *)calloc(sizeof(int), length_data);
      Cxold = (float *)calloc(sizeof(float), k_total);
      Cyold = (float *)calloc(sizeof(float), k_total);
      Czold = (float *)calloc(sizeof(float), k_total);

      if((argc >= 4) && ((strcmp(argv[3], "-d" )==0)))
      	assignments_per_iter = (int *)calloc(sizeof(int), length_data * a_per_i_length);
   
      //initialize the k cluster centers to random points from the data
      int r;
      
      srand(time(NULL));
      for (i = 0; i < k_total; i++) {
         r = rand() % length_data; //TODO: check if same number twice
         Cxold[i] = Cx[i] = Px[r];
         Cyold[i] = Cy[i] = Py[r];
         Czold[i] = Cz[i] = Pz[r];
      }
   }
   
   //if (rank == ROOT)
      //printf("...Clusters initialized...\n");
   
   int changed = 0;
   int *num_assigned = (int *)calloc(sizeof(int), k_total);

   
   
   iter = 0;
   do {
      //Bcast the cluster centers
      MPI_Bcast(Cx, k_total, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cy, k_total, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cz, k_total, MPI_FLOAT, ROOT, MPI_COMM_WORLD);

      //if (rank == ROOT)
         //printf("...Broadcasted centers...\n");

      // do actual clustering assignments with cuda stuff
      cudaAssign(Cx, Cy, Cz, recvbufPx, recvbufPy, recvbufPz, assignments, send_element_count, k_total);
      
      //if (rank == ROOT)
         //printf("...Cuda finished...\n");
      
      //Each node has a three k-length array, each cell represents the sums of its assigned data vals, for x, y, z
      
      float *Cxtemp = (float *)calloc(sizeof(float), k_total);
      float *Cytemp = (float *)calloc(sizeof(float), k_total);
      float *Cztemp = (float *)calloc(sizeof(float), k_total);
      int *num_assigned_temp = (int *)calloc(sizeof(int), k_total);
      
      //if (rank == ROOT)
         //printf("...Calloced...\n");
      
      /* Calculate local new cluster center means */
      
      for (i = 0; i < send_element_count; i++) {
      
         Cxtemp[assignments[i]] += recvbufPx[i];
         Cytemp[assignments[i]] += recvbufPy[i];
         Cztemp[assignments[i]] += recvbufPz[i];
         num_assigned_temp[assignments[i]]++;
      }
      
      //if (rank == ROOT)
         //printf("...Assignments counted...\n");
     
      
      /* Reduce (sum) the local cluster means onto root */
      
      MPI_Reduce(Cxtemp, Cx, k_total, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      MPI_Reduce(Cztemp, Cz, k_total, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
      MPI_Reduce(num_assigned_temp, num_assigned, k_total, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

	  if((argc >= 4) && ((strcmp(argv[3], "-d" )==0))){
      	MPI_Gather(assignments, send_element_count, MPI_INT, &(assignments_per_iter[iter*length_data]), send_element_count, MPI_INT, ROOT, MPI_COMM_WORLD);
      }
      
      //if (rank == ROOT)
         //printf("...Reduced...\n");
      
      /* As root calculate final cluster centers */
      
      if (rank == ROOT) {
         
         for (i = 0; i < k_total; i++) {
            Cx[i] /= num_assigned[i];
            Cy[i] /= num_assigned[i];
            Cz[i] /= num_assigned[i];
         }
      }
      
      //if (rank == ROOT)
         //printf("...Freeing...\n");
      
      free(Cxtemp); free(Cytemp); free(Cztemp);
      free(num_assigned_temp);
      
      //if (rank == ROOT)
         //printf("...Freed...\n");
      
      /* Check if cluster means changed, and update old */
      
      if (rank == ROOT) {
         changed = centersChanged(Cxold, Cyold, Czold, Cx, Cy, Cz, k_total);
         for (i = 0; i < k_total; i++) {
            Cxold[i] = Cx[i];
            Cyold[i] = Cy[i];
            Czold[i] = Cz[i];
         }
      }
      
      MPI_Bcast(&changed, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
   
   /* Continue the loop while cluster centers are not same as before */

   
      MPI_Barrier(MPI_COMM_WORLD);

   
      iter++;
   } while (changed);
   
   //if (rank == ROOT) {
   //   printf("...Finished clustering...\n");
   //   printf("iter: %d\n", iter);
   //}
      
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
   
   MPI_Finalize();
   
   if(rank == ROOT) {
	  if((argc >= 4) && ((strcmp(argv[3], "-d" )==0)))
         drawEverything();
      //free(final_assignments);
   }

}

int addIteration(assignments) {

}

int vectorSize(char *filename) {

   struct stat stbuf;
   int flag = stat(filename, &stbuf);
   int filesize = stbuf.st_size;

   FILE *fp = fopen(filename, "r");

   char *text;
   if(NULL == (text = (char *)malloc((filesize+1) * sizeof(char)))) {
      return -1;
   }

   int nchar = fread(text, sizeof(char), filesize, fp);
   text[nchar] = '\0';

   char endline[5] = " \n\r\0";
   
   char *line;

   char *line_str;
   line = strtok_r(text, endline, &line_str);

   int vector_size = 0;
   while(line) {

      line = strtok_r(NULL, endline, &line_str);
      vector_size++;
   }


   free(text);
   fclose(fp);

   return vector_size;

}

int fileRead(char *filename, float *lon, float *lat, float *mag, int size) {

   struct stat stbuf;
   int flag = stat(filename, &stbuf);
   int filesize = stbuf.st_size;

   FILE *fp = fopen(filename, "r");

   char *text;
   if(NULL == (text = (char *)malloc((filesize+1) * sizeof(char)))) {
      return -1;
   }

   int nchar = fread(text, sizeof(char), filesize, fp);
   text[nchar] = '\0';
   
   char endline[5] = " \n\r\0";
   
   char *line;

   char *line_str;

   line = strtok_r(text, endline, &line_str);

   int i = 0;
   while(line) {
      char *end_num;

      char *num = strtok_r(line, "\t", &end_num);
      lon[i] = atof(num);

      num = strtok_r(NULL, "\t", &end_num);
      lat[i] = atof(num);

      num = strtok_r(NULL, "\t", &end_num);
      mag[i] = atof(num);


      line = strtok_r(NULL, endline, &line_str);
      i++;
   }

   free(text);
   fclose(fp);

   return size;

}

int centersChanged(float *Cxold, float *Cyold, float *Czold, float *Cx, float *Cy, float *Cz, int k) {
   int i, changed = 0;
   for (i = 0; i < k; i++) {
      if (Cxold[i] != Cx[i] || Cyold[i] != Cy[i] || Czold[i] != Cz[i] ) {
         changed = 1;
      }
   }
   return changed;
}
