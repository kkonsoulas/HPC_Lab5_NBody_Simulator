#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "timer.h"

#define BLOCK_SIZE 1024
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


#define SOFTENING 1e-9f  /* Will guard against denormals */

typedef struct { float *x, *y, *z, *vx, *vy, *vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyPosition(float *x ,float *y ,float *z ,float *vx ,float *vy ,float *vz ,float dt, int n){
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i>= n)
    return;
  
  float pi_vy =  vy[i]; 
  float pi_vz =  vz[i];

  x[i] += vx[i]*dt;
  y[i] += pi_vy*dt;
  z[i] += pi_vz*dt;

}


__global__ 
void bodyForce(float *px, float *py, float *pz, float *pvx, float *pvy, float *pvz, float dt, int n) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i>= n)
    return;

  float 
  pi_x = px[i],
  pi_y = py[i],
  pi_z = pz[i],
  pj_x0 ,pj_x1,
  pj_y0 ,pj_y1,
  pj_z0 ,pj_z1,
  dx,dy,dz;

  pj_x0 =  px[0];
  pj_y0 = py[0];
  pj_z0 = pz[0];

  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      if(j & 1){
        if(j+1 < n){
          pj_x0 = px[j+1];
          pj_y0 = py[j+1];
          pj_z0 = pz[j+1];
        }
        
        dx = pj_x1 - pi_x;
        dy = pj_y1 - pi_y;
        dz = pj_z1 - pi_z;
      }
      else{
        if(j+1 < n){
          pj_x1 = px[j+1];
          pj_y1 = py[j+1];
          pj_z1 = pz[j+1];
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



  pvx[i] += dt*Fx; pvy[i] += dt*Fy; pvz[i] += dt*Fz;

}



__global__ 
void bodyForce_unroll_8(float *px, float *py, float *pz, float *pvx, float *pvy, float *pvz, float dt, int n) {
int i = threadIdx.x + blockDim.x*blockIdx.x;
  if(i>= n)
    return;

  float
  pi_x = px[i],
  pi_y = py[i],
  pi_z = pz[i],
  pj_x0 ,pj_x1,
  pj_y0 ,pj_y1,
  pj_z0 ,pj_z1,
  dx,dy,dz;

  pj_x0 =  px[0];
  pj_y0 = py[0];
  pj_z0 = pz[0];

  pj_x1 = px[1];
  pj_y1 = py[1];
  pj_z1 = pz[1];

  float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 1; j < n; j++) {
      //1
        
        
      dx = pj_x0 - pi_x;
      dy = pj_y0 - pi_y;
      dz = pj_z0 - pi_z;


      pj_x0 = px[j+1];
      pj_y0 = py[j+1];
      pj_z0 = pz[j+1];

      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;

      //2


        
      dx = pj_x1 - pi_x;
      dy = pj_y1 - pi_y;
      dz = pj_z1 - pi_z;


      pj_x1 = px[j+1];
      pj_y1 = py[j+1];
      pj_z1 = pz[j+1];

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;


      //3
      pj_x1 = px[j];
      pj_y1 = py[j];
      pj_z1 = pz[j];
        
        
      dx = pj_x0 - pi_x;
      dy = pj_y0 - pi_y;
      dz = pj_z0 - pi_z;
      

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;



      //4
      pj_x0 = px[j];
      pj_y0 = py[j];
      pj_z0 = pz[j];

        
      dx = pj_x1 - pi_x;
      dy = pj_y1 - pi_y;
      dz = pj_z1 - pi_z;

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;



      //5
      pj_x1 = px[j];
      pj_y1 = py[j];
      pj_z1 = pz[j];
        
        
      dx = pj_x0 - pi_x;
      dy = pj_y0 - pi_y;
      dz = pj_z0 - pi_z;
      

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;


      //6
      pj_x0 = px[j];
      pj_y0 = py[j];
      pj_z0 = pz[j];

        
      dx = pj_x1 - pi_x;
      dy = pj_y1 - pi_y;
      dz = pj_z1 - pi_z;

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;

      //7
      pj_x1 = px[j];
      pj_y1 = py[j];
      pj_z1 = pz[j];
        
        
      dx = pj_x0 - pi_x;
      dy = pj_y0 - pi_y;
      dz = pj_z0 - pi_z;
      

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      j++;

      //8
      pj_x0 = px[j];
      pj_y0 = py[j];
      pj_z0 = pz[j];

        
      dx = pj_x1 - pi_x;
      dy = pj_y1 - pi_y;
      dz = pj_z1 - pi_z;

      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist = 1.0f / sqrtf(distSqr);
      invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      // j++;

    }



  pvx[i] += dt*Fx; pvy[i] += dt*Fy; pvz[i] += dt*Fz;

}


int main(const int argc, const char** argv) {

  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = 6*nBodies*sizeof(float);
  float *buf = (float*)malloc(bytes);

  Body p = {(float*)buf, ((float*)buf) + nBodies, ((float*)buf) + 2*nBodies,
            ((float*)buf) + 3*nBodies, ((float*)buf) + 4*nBodies, ((float*)buf) + 5*nBodies};
  
  float *tempBuf = (float*)malloc(bytes);
  randomizeBodies(tempBuf, 6*nBodies); // Init pos / vel data
  for(int i=0,j=0 ; i < nBodies ; i++,j=j+6){
    p.x[i] = tempBuf[j]; 
    p.y[i] = tempBuf[j+1];
    p.z[i] = tempBuf[j+2];
    p.vx[i] = tempBuf[j+3];
    p.vy[i] = tempBuf[j+4];
    p.vz[i] = tempBuf[j+5];

  }
  free(tempBuf);

  float *d_buf;
  cudaMalloc(&d_buf,bytes);
  cudaCheckError();
  Body d_p = {(float*)d_buf, ((float*)d_buf) + nBodies, ((float*)d_buf) + 2*nBodies,
            ((float*)d_buf) + 3*nBodies, ((float*)d_buf) + 4*nBodies, ((float*)d_buf) + 5*nBodies};

  
  //timer init
  double totalTime = 0.0;
  
  //StartTimer();
  //cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
  //totalTime += GetTimer() / 1000.0;
  
  int blocks = (nBodies / BLOCK_SIZE) + 1;
  

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();
    
    cudaMemcpy(d_buf,buf,bytes,cudaMemcpyHostToDevice);
    //unrolling code
    // if(nBodies % 8){
    //   bodyForce<<<blocks,BLOCK_SIZE>>>(d_p.x,d_p.y,d_p.z,d_p.vx,d_p.vy,d_p.vz, dt, nBodies); // compute interbody forces
    // }
    // else{
    //   bodyForce_unroll_8<<<blocks,BLOCK_SIZE>>>(d_p.x,d_p.y,d_p.z,d_p.vx,d_p.vy,d_p.vz, dt, nBodies); // compute interbody forces
    // }
    bodyForce<<<blocks,BLOCK_SIZE>>>(d_p.x,d_p.y,d_p.z,d_p.vx,d_p.vy,d_p.vz, dt, nBodies); // compute interbody forces


    cudaCheckError();
    cudaDeviceSynchronize();

    bodyPosition<<<blocks,BLOCK_SIZE>>>(d_p.x,d_p.y,d_p.z,d_p.vx,d_p.vy,d_p.vz, dt, nBodies);

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
  if ((f = fopen("measurements_soa.txt","w")) == NULL){
    printf("Error in opening file!\n");
    return -1;
  }

  for(int i=0; i < nBodies; i++){
    fprintf(f,"%.3f %.3f %.3f\n",p.x[i],p.y[i],p.z[i]);
  }
  cudaDeviceReset();
  free(buf);
  cudaFree(d_buf);
}
