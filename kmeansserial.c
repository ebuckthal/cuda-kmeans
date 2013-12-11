/* Serial implementation of k-means clustering
   for speedup comparison
*/

#include <time.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "kmeans.h"

int main(int argc, char **argv) {
   /* Input error checking */

   if (argc < 3) {
      printf("Usage: kmeans <file1> <k>\n");
      return 1;
   }
   
   length_data = vectorSize(argv[1]);
   
   Px = (float *)calloc(sizeof(float), length_data);
   Py = (float *)calloc(sizeof(float), length_data);
   Pz = (float *)calloc(sizeof(float), length_data);
   
   fileRead(argv[1], Px, Py, Pz, length_data);
   
   //printf("...File read complete...\n");
   fflush(stdout);

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
      
   for(j=0; j<length_data; j++){
      Px[j] = (((Px[j]-xmin)* 10)/(xmax-xmin)) - 5;
      Py[j] = (((Py[j]-ymin)* 10)/(ymax-ymin)) - 5;
      Pz[j] = (((Pz[j]-zmin)* 10)/(zmax-zmin)) - 5;
   }
   
   k_total = atoi(argv[2]); //TODO: error check
   
   
   float *Cx = (float *)calloc(sizeof(float), k_total);
   float *Cy = (float *)calloc(sizeof(float), k_total);
   float *Cz = (float *)calloc(sizeof(float), k_total);
   float *Cxold = (float *)calloc(sizeof(float), k_total);
   float *Cyold = (float *)calloc(sizeof(float), k_total);
   float *Czold = (float *)calloc(sizeof(float), k_total);
   final_assignments = (int *)calloc(sizeof(int), length_data);
   
   int r, i;
   
   srand(time(NULL));
   for (i = 0; i < k_total; i++) {
      r = rand() % length_data; //TODO: check if same number twice
      Cxold[i] = Cx[i] = Px[r];
      Cyold[i] = Cy[i] = Py[r];
      Czold[i] = Cz[i] = Pz[r];
   }
   
   int changed = 0;
   int *num_assigned = (int *)calloc(sizeof(int), k_total);
   int centerIdx;
   float d,temp_d;
   
   //printf("...Starting loop...\n");
   fflush(stdout);
   
   int iter2 = 0;
   
   do {
      // do actual clustering assignments
      for (j = 0; j < length_data; j++) {
		 d = INT_MAX/1.0;
         for (i = 0; i < k_total; i++) {
            temp_d = (Px[j] - Cx[i])*(Px[j] - Cx[i]) 
                       + (Py[j] - Cy[i])*(Py[j] - Cy[i]) 
                       + (Pz[j] - Cz[i])*(Pz[j] - Cz[i]);
            if (temp_d < d) {
               d = temp_d;
               centerIdx = i;
            }
         }
         final_assignments[j] = centerIdx;
      }
      
      /* Calculate local new cluster center means */
	  for(i=0; i < k_total; i++){
	     Cx[i] = Cy[i] = Cz[i] = 0;
		 num_assigned[i] = 0;
      }
      
      for (i = 0; i < length_data; i++) {
         Cx[final_assignments[i]] += Px[i];
         Cy[final_assignments[i]] += Py[i];
         Cz[final_assignments[i]] += Pz[i];
         num_assigned[final_assignments[i]]++;
      }
     
      /* calculate final cluster centers */
 
      for (i = 0; i < k_total; i++) {
         
         if(num_assigned[i] > 0) {
         
               assert(num_assigned[i] > 0);
            
               Cx[i] /= num_assigned[i];
               Cy[i] /= num_assigned[i];
               Cz[i] /= num_assigned[i];
            
         }
      } 
      
      /* Check if cluster means changed, and update old */
      //printf("%d\n", iter2);
      
      changed = centersChanged(Cxold, Cyold, Czold, Cx, Cy, Cz, k_total);
      for (i = 0; i < k_total; i++) {
         Cxold[i] = Cx[i];
         Cyold[i] = Cy[i];
         Czold[i] = Cz[i];


      }
			  //int z = 0;
			  //for(; z < k_total; z++) {
				//  printf("%d ", num_assigned[z]);
			  //}
			// printf("\n");
   
   /* Continue the loop while cluster centers are not same as before */
   iter2++;
   } while (changed);
   
   //printf("...Finished clustering...\n");
   
   free(Cxold); free(Cyold); free(Czold);
   free(Cx); free(Cy); free(Cz);

	assignments_per_iter = final_assignments;
	iter = 1;

   if((argc >= 4) && ((strcmp(argv[3], "-d" )==0)))
      drawEverything();
   free(final_assignments);
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
