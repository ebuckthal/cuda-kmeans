
__global__ void clusterAssign(float *Cx, float *Cy, float *Cz, float *Px, float *Py, float *Pz, int *assigns, int Pwidth, int k) 
{
   int tid, i, temp_d, centerIdx;
   int d = MAX_INT;

   tid = blockIdx.x * blockDim.x + threadIdx.x;

   while (tid < width) {
      for (i = 0; i < k; i++) {
         temp_d = (Px[tid] - Cx[i])*(Px[tid] - Cx[i]) 
                    + (Py[tid] - Cy[i])*(Py[tid] - Cy[i]) 
                    + (Pz[tid] - Cz[i])*(Pz[tid] - Cz[i]);
         if (temp_d < d) {
            d = temp_d;
            centerIdx = i;
         }
      }
      assigns[tid] = centerIdx;
      tid += blockDim.x * gridDim.x;      
   }
   
   return;
}

extern "C" void cudaAssign(float *Cxin, float *Cyin, float *Czin, float *Pxin, float *Pyin, float *Pzin, int *assignments, int Pwidth, int k) {
   float *Cx, *Cy, *Cz, *Px, *Py, *Pz;
   int *assigns;

   cudaMalloc(&Cx, (unsigned long)k*sizeof(float));
   cudaMalloc(&Cy, (unsigned long)k*sizeof(float));
   cudaMalloc(&Cz, (unsigned long)k*sizeof(float));
   cudaMalloc(&Px, (unsigned long)Pwidth*sizeof(float));
   cudaMalloc(&Py, (unsigned long)Pwidth*sizeof(float));
   cudaMalloc(&Pz, (unsigned long)Pwidth*sizeof(float));
   cudaMalloc(&assigns, (unsigned long)Pwidth*sizeof(int));

   cudaMemcpy(Cx, Cxin, k*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(Cy, Cyin, k*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(Cz, Czin, k*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(Px, Pxin, Pwidth*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(Py, Pyin, Pwidth*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(Pz, Pzin, Pwidth*sizeof(float), cudaMemcpyHostToDevice);
   
   clusterAssign<<<1024, 512>>>(Cx, Cy, Cz, Px, Py, Pz, assigns, Pwidth, k);
   cudaMemcpy(assignments, assigns, Pwidth*sizeof(int), cudaMemcpyDeviceToHost);
   
   cudaFree(Cx);
   cudaFree(Cy);
   cudaFree(Cz);   
   cudaFree(Px);   
   cudaFree(Py);   
   cudaFree(Pz);   
   cudaFree(assigns);   
}
