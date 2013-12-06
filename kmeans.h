#ifndef KMEANS_H
#define KMEANS_H

#include <mpi.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define BAD_ARGS 1
#define ROOT 0


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

#endif
