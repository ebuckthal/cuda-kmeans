//remember the includes for random number
//#include <time.h>
//#include <stdlib.h>

int centersChanged(float *Cxold, float *Cyold, float *Czold, float *Cx, float *Cy, float *Cz, int k) {
   int i, changed = 0;
   for (i = 0; i < k; i++) {
      if (Cxold[i] != Cx[i] || Cyold[i] != Cy[i] || Czold[i] != Cz[i] ) {
         changed = 1;
      }
   }
   return changed;
}

int main(int argc, char **argv) {
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &number_processes);

   
   if (rank == ROOT)
      printf("...Initialized...\n");
   
   
   int length_data = 0;
   
   //TODO: as root, read in data here somehow
   
   //TODO: Scatter the data
   
   //TODO: Bcast the length of the data
   
   //TODO: as root choose k (hardcode?) Are we iterating over multiple values of k?
   // if yes, then we need to be able to choose the point of inflection for best k
   // and measure total error for each k (sum of within-cluster distance)
   
   float *Cx, *Cy, *Cz;
   Cx = (float *)calloc(sizeof(float), k);
   Cy = (float *)calloc(sizeof(float), k);
   Cz = (float *)calloc(sizeof(float), k);
   Cxold = (float *)calloc(sizeof(float), k);
   Cyold = (float *)calloc(sizeof(float), k);
   Czold = (float *)calloc(sizeof(float), k);
   
   //TODO: as root initialize the k cluster centers to random points from the data
   if (rank == ROOT) {
      int i;
      int r;
      srand(time(NULL));
      for (i = 0; i < k; i++) {
         r = rand() % length_data;
         Cxold[i] = Cx[i] = Px[r]; //Px etc need to be initialized
         Cyold[i] = Cy[i] = Py[r];
         Czold[i] = Cz[i] = Pz[r];
      }
   }
   
   // DO-------
   do {
      //TODO: Bcast the cluster centers
      MPI_Bcast(Cx, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cy, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      MPI_Bcast(Cz, k, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
      
      
      //TODO: do actual clustering computation and cuda stuff
      cudaAssign(Cx, Cy, Cz, Px, Py, Pz, assignments, length_data, k); //NEED to initialize assignments and Pwidth (length_data)
      
      //TODO: do something to organize the assignments or centers or something so that
      // they can be reduced ?
      
      //TODO: check if cluster centers are the same as the previous cluster centers
   
   //WHILE -------- cluster centers are not same as before
   } while (centersChanged(Cxold, Cyold, Czold, Cx, Cy, Cz));
   // i guess we done now
   
   /*reference code*/
   //float *recvbuf = (float *)calloc(sizeof(float), number_elements);   
   //float *recvbuf2 = (float *)calloc(sizeof(float), number_elements);   
   //float *recvbuf3 = (float *)calloc(sizeof(float), number_elements);
   
   
   //MPI_Bcast(&number_elements, 1, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
   
   //send_element_count = (number_elements / number_processes);
   
   //MPI_Scatter(first_vector, send_element_count, MPI_FLOAT, recvbuf, 
                     send_element_count, MPI_FLOAT, ROOT, MPI_COMM_WORLD);
}
