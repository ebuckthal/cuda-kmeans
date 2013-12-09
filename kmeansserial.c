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

float *Px;
float *Py;
float *Pz;

int *final_assignments;
int length_data;
int k_total;

int centersChanged(float*, float*, float*, float*, float*, float*, int);
int vectorSize(char *);
int fileRead(char *, float *, float *, float *, int);

int main(int argc, char **argv) {
   /* Input error checking */

   if (argc != 3) {
      printf("Usage: kmeans <file1> <k>\n");
      return 1;
   }
   
   length_data = vectorSize(argv[1]);
   
   Px = (float *)calloc(sizeof(float), length_data);
   Py = (float *)calloc(sizeof(float), length_data);
   Pz = (float *)calloc(sizeof(float), length_data);
   
   fileRead(argv[1], Px, Py, Pz, length_data);
   
   printf("...File read complete...\n");
   fflush(stdout);
   
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
   int j, temp_d, centerIdx;
   int d = INT_MAX;
   
   printf("...Starting loop...\n");
   fflush(stdout);
   
   do {
      // do actual clustering assignments
      for (j = 0; j < length_data; j++) {
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
      
      for (i = 0; i < length_data; i++) {
         Cx[final_assignments[i]] += Px[i];
         Cy[final_assignments[i]] += Py[i];
         Cz[final_assignments[i]] += Pz[i];
         num_assigned[final_assignments[i]]++;
      }
     
      /* calculate final cluster centers */
 
      for (i = 0; i < k_total; i++) {
         Cx[i] /= num_assigned[i];
         Cy[i] /= num_assigned[i];
         Cz[i] /= num_assigned[i];
      } 
      
      /* Check if cluster means changed, and update old */
      
      changed = centersChanged(Cxold, Cyold, Czold, Cx, Cy, Cz, k_total);
      for (i = 0; i < k_total; i++) {
         Cxold[i] = Cx[i];
         Cyold[i] = Cy[i];
         Czold[i] = Cz[i];
      }
   
   /* Continue the loop while cluster centers are not same as before */
   } while (changed);
   
   printf("...Finished clustering...\n");
   
   free(Cxold); free(Cyold); free(Czold);
   free(Cx); free(Cy); free(Cz);
   
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
