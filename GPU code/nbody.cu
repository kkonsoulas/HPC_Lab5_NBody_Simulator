#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "timer.h"
//307200 
#define TILING_SIZE 196608
#define BLOCK_SIZE 1024
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


#define SOFTENING 1e-9f  /* Will guard against denormals */

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}


__global__
void bodyPosition(Body *p, float dt, int n){
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i>= n)
    return;
  
  float pi_vy =  p[i].vy; 
  float pi_vz =  p[i].vz;

  p[i].x += p[i].vx*dt;
  p[i].y += pi_vy*dt;
  p[i].z += pi_vz*dt;

}




__global__ 
void bodyForce(Body *p, float dt, int n) {
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i>= n)
    return;

  float 
  pi_x = p[i].x,
  pi_y = p[i].y,
  pi_z = p[i].z,
  pj_x0 ,pj_x1,
  pj_y0 ,pj_y1,
  pj_z0 ,pj_z1,
  dx,dy,dz;

  pj_x0 =  p[0].x;
  pj_y0 = p[0].y;
  pj_z0 = p[0].z;

  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      if(j & 1){
        if(j+1 < n){
          pj_x0 = p[j+1].x;
          pj_y0 = p[j+1].y;
          pj_z0 = p[j+1].z;
        }
        
        dx = pj_x1 - pi_x;
        dy = pj_y1 - pi_y;
        dz = pj_z1 - pi_z;
      }
      else{
        if(j+1 < n){
          pj_x1 = p[j+1].x;
          pj_y1 = p[j+1].y;
          pj_z1 = p[j+1].z;
        }
        
        dx = pj_x0 - pi_x;
        dy = pj_y0 - pi_y;
        dz = pj_z0 - pi_z;
      }
      // dx = p[j].x - pi_x;
      // dy = p[j].y - pi_y;
      // dz = p[j].z - pi_z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }



  p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;

}



int main(const int argc, const char** argv) {

  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  float *d_buf;
  cudaMalloc(&d_buf,bytes);
  cudaCheckError();
  Body *d_p = (Body*)d_buf;
  
  //timer init
  double totalTime = 0.0;
  
  //StartTimer();
  //cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
  //totalTime += GetTimer() / 1000.0;
  
  int blocks = (nBodies / BLOCK_SIZE) + 1;
  


  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer(); 

    int i;

    cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
    bodyForce<<<blocks,BLOCK_SIZE>>>(d_p, dt, nBodies); // compute interbody forces
    cudaCheckError();

    cudaDeviceSynchronize();
    // blocks = (nBodies / BLOCK_SIZE) + 1;
    bodyPosition<<<blocks,BLOCK_SIZE>>>(d_p, dt, nBodies);

    cudaMemcpy(buf,d_buf,bytes,cudaMemcpyDeviceToHost);


    // cudaMemcpy(buf,d_buf,bytes,cudaMemcpyDeviceToHost);
    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    printf("Iteration %d: %.6f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1); 
  
  


  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  
  /*Saves the output in a txt file*/
  FILE* f;
  if ((f = fopen("measurements.txt","w")) == NULL){
    printf("Error in opening file!\n");
    fclose(f);
    return -1;
  }

  for(int i=0; i < nBodies; i++){
    fprintf(f,"%.3f %.3f %.3f\n",p[i].x,p[i].y,p[i].z);
  }
  cudaDeviceReset();
  free(buf);
  cudaFree(d_buf);
}
