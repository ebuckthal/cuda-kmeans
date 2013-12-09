#ifndef KMEANS_H
#define KMEANS_H

#define BAD_ARGS 1
#define ROOT 0

/* Arrays for storing the read-in data */
   
float *Px;
float *Py;
float *Pz;

int *assignments_per_iter;

int *final_assignments;
int length_data;
int iter;

int k_total;

int mainDraw(void);

int centersChanged(float*, float*, float*, float*, float*, float*, int);

// some function prototypes
int mainDraw(void);

#endif
