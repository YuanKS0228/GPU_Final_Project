#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
void Set_GPU_Device(int device) {
    cudaSetDevice(device);
}
__global__ void init_curand_states(curandState *states, int total_particles, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < total_particles) {
        curand_init(seed, id, 0, &states[id]);
    }
}

__global__ void generate_particles_kernel(float *x_p, float *y_p, float *vx_p, float *vy_p,
                                          int NX, int NY, int p_num, float DX, float DY,
                                          float *p_cell, float *d, float *s_cell, float *vx, float *vy,
                                          float *T, float *m_p_array, float *e_int_p, curandState *states,float R_GAS) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_particles = (NX + 2) * (NY + 2) * p_num;

    if (id >= total_particles) return;

    // 將 1D 的 id 反推回 j, k, z
    int j = id / ((NY + 2) * p_num);
    int k = (id / p_num) % (NY + 2);
    // int z = id % p_num; (原本邏輯中 z 其實沒被拿來計算，所以可以省略)

    int index = j * (NY + 2) + k;
    curandState localState = states[id];

    if (s_cell[index] == 0 && k > 0 && k < NY + 1 && j > 0) {
        x_p[id]  = j * DX + curand_uniform(&localState) * DX;
        y_p[id]  = k * DY + curand_uniform(&localState) * DY;
        vx_p[id] = vx[index] + curand_normal(&localState) * sqrtf(R_GAS * T[index]);
        vy_p[id] = vy[index] + curand_normal(&localState) * sqrtf(R_GAS * T[index]);
    } else {
        if (j == 0) {
            x_p[id]  = j * DX + curand_uniform(&localState) * DX;
            y_p[id]  = k * DY + curand_uniform(&localState) * DY;
            vx_p[id] = 3.0f + curand_normal(&localState) * sqrtf(R_GAS * T[index]);
            vy_p[id] = 0.0f + curand_normal(&localState) * sqrtf(R_GAS * T[index]);
        } else {
            x_p[id]  = 0.0f;
            y_p[id]  = 0.0f;
            vx_p[id] = 0.0f;
            vy_p[id] = 0.0f;
        }
    }
    if (j == 0) {
        m_p_array[id] = (1.4f * DX * DY) / p_num;
    } else {
        m_p_array[id] = p_cell[index] / p_num;
    }
    e_int_p[id] = 1.5f * m_p_array[id] * R_GAS * T[index];

    states[id] = localState; // 存回狀態
}

__global__ void move_particles_kernel(float *x_p, float *y_p, float *vx_p, float *vy_p, 
                                      int total_particles, float DX, float DY, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_particles) return;

    if (0 < x_p[i] && x_p[i] < 0.6f + DX && DY < y_p[i] && y_p[i] < 0.2f + DY) {
        x_p[i] += vx_p[i] * dt;
        y_p[i] += vy_p[i] * dt;
        if (x_p[i] > 0.6f + DX) { x_p[i] = 2.0f * (0.6f + DX) - x_p[i]; vx_p[i] = -vx_p[i]; }
        if (y_p[i] < DY)        { y_p[i] = 2.0f * DY - y_p[i]; vy_p[i] = -vy_p[i]; }
    }

    if (0 < x_p[i] && x_p[i] < 0.6f + DX && 0.2f + DY < y_p[i] && y_p[i] < 1.0f + DY) {
        x_p[i] += vx_p[i] * dt;
        y_p[i] += vy_p[i] * dt;
        if (y_p[i] > 1.0f + DY) { y_p[i] = 2.0f * (1.0f + DY) - y_p[i]; vy_p[i] = -vy_p[i]; }
    }

    if (0.6f + DX < x_p[i] && x_p[i] < 3.0f + DX && 0.2f + DY < y_p[i] && y_p[i] < 1.0f + DY) {
        x_p[i] += vx_p[i] * dt;
        y_p[i] += vy_p[i] * dt;
        if (y_p[i] < 0.2f + DY) { y_p[i] = 2.0f * (0.2f + DY) - y_p[i]; vy_p[i] = -vy_p[i]; }
        if (y_p[i] > 1.0f + DY) { y_p[i] = 2.0f * (1.0f + DY) - y_p[i]; vy_p[i] = -vy_p[i]; }
    }
}

__global__ void reset_grid_kernel(float *p_cell, float *mom_x_cell, float *mom_y_cell, float *energy_cell, int total_cells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_cells) {
        p_cell[i] = 0.0f;
        mom_x_cell[i] = 0.0f;
        mom_y_cell[i] = 0.0f;
        energy_cell[i] = 0.0f;
    }
}

__global__ void project_to_grid_kernel(float *x_p, float *y_p, float *vx_p, float *vy_p, float *m_p_array, float *e_int_p,
                                       float *p_cell, float *mom_x_cell, float *mom_y_cell, float *energy_cell,
                                       int NX, int NY, float DX, float DY, int total_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_particles) return;

    if (x_p[i] == 0.0f && y_p[i] == 0.0f) return;

    int j = (int)(x_p[i] / DX);
    int k = (int)(y_p[i] / DY);

    if (j > 0 && j < NX + 1 && k > 0 && k < NY + 1) {
        int index = j * (NY + 2) + k;
        float mass = m_p_array[i];
        float mx = mass * vx_p[i];
        float my = mass * vy_p[i];
        float energy = 0.5f * mass * (vx_p[i] * vx_p[i] + vy_p[i] * vy_p[i]) + e_int_p[i];

        // 使用原子操作避免資料衝突
        atomicAdd(&p_cell[index], mass);
        atomicAdd(&mom_x_cell[index], mx);
        atomicAdd(&mom_y_cell[index], my);
        atomicAdd(&energy_cell[index], energy);
    }
}

__global__ void update_grid_kernel(float *p_cell, float *mom_x_cell, float *mom_y_cell, float *energy_cell,
                                   float *vx, float *vy, float *T, float *d, float *p, 
                                   int NX, int NY, float DX, float DY, int total_cells,float R_GAS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_cells) return;

    if (p_cell[i] > 1e-10f) {
        vx[i] = mom_x_cell[i] / p_cell[i];
        vy[i] = mom_y_cell[i] / p_cell[i];
        float E_TOTAL = 0.5f * p_cell[i] * (vx[i] * vx[i] + vy[i] * vy[i]);
        float E_THERMAL = energy_cell[i] - E_TOTAL;
        T[i] = E_THERMAL / (2.5f * R_GAS * p_cell[i]);
        d[i] = p_cell[i] / (DX * DY);
        p[i] = d[i] * R_GAS * T[i];
    }
}

void Allocate_Memory(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
		     float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
		     float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                     float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		     curandState **d_states,int num_cells,int total_particles) {
    size_t size1 =num_cells *sizeof(float);
    size_t size2 =total_particles *sizeof(float);

    cudaError_t Error;
    //curandState *d_states;
    // Host memory
    *h_cell = (float*)malloc(size1);
    *h_s_cell = (float*)malloc(size1);
    *h_vx = (float*)malloc(size1);
    *h_vy = (float*)malloc(size1);
    *h_p = (float*)malloc(size1);
    *h_d = (float*)malloc(size1);
    *h_x_p = (float*)malloc(size2);
    *h_y_p = (float*)malloc(size2);
    *h_vx_p = (float*)malloc(size2);
    *h_vy_p = (float*)malloc(size2);
    *h_p_cell = (float*)malloc(size1);
    *h_mom_x_cell = (float*)malloc(size1);
    *h_mom_y_cell = (float*)malloc(size1);
    *h_energy_cell = (float*)malloc(size1);
    *h_T = (float*)malloc(size1);
    *h_m_p_array = (float*)malloc(size2);
    *h_e_int_p = (float*)malloc(size2);

    // Device memory
    Error = cudaMalloc((void**)d_cell, size1);
    printf("CUDA error (malloc d_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_s_cell, size1);
    printf("CUDA error (malloc d_s_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_vx, size1);
    printf("CUDA error (malloc d_vx) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_vy, size1);
    printf("CUDA error (malloc d_vy) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_p, size1);
    printf("CUDA error (malloc d_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_d, size1);
    printf("CUDA error (malloc d_d) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_x_p, size2);
    printf("CUDA error (malloc d_x_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_y_p, size2);
    printf("CUDA error (malloc d_y_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_vx_p, size2);
    printf("CUDA error (malloc d_vx_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_vy_p, size2);
    printf("CUDA error (malloc d_vy_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_p_cell, size1);
    printf("CUDA error (malloc d_p_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_mom_x_cell, size1);
    printf("CUDA error (malloc d_mom_x_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_mom_y_cell, size1);
    printf("CUDA error (malloc d_mom_y_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_energy_cell, size1);
    printf("CUDA error (malloc d_energy_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_T, size1);
    printf("CUDA error (malloc d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_m_p_array, size2);
    printf("CUDA error (malloc d_m_p_array) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void**)d_e_int_p, size2);
    printf("CUDA error (malloc d_e_int_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMalloc((void **)d_states, total_particles * sizeof(curandState));
    printf("CUDA error (malloc d_state) = %s\n", cudaGetErrorString(Error));

}
void Free_Memory(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
                 float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
                 float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                 float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		 curandState **d_state) {
    if (*h_cell) free(*h_cell);
    if (*h_s_cell) free(*h_s_cell);
    if (*h_vx) free(*h_vx);
    if (*h_vy) free(*h_vy);
    if (*h_p) free(*h_p);
    if (*h_d) free(*h_d);
    if (*h_x_p) free(*h_x_p);
    if (*h_y_p) free(*h_y_p);
    if (*h_vx_p) free(*h_vx_p);
    if (*h_vy_p) free(*h_vy_p);
    if (*h_p_cell) free(*h_p_cell);
    if (*h_mom_x_cell) free(*h_mom_x_cell);
    if (*h_mom_y_cell) free(*h_mom_y_cell);
    if (*h_energy_cell) free(*h_energy_cell);
    if (*h_T) free(*h_T);
    if (*h_m_p_array) free(*h_m_p_array);
    if (*h_e_int_p) free(*h_e_int_p);

    if (*d_cell) cudaFree(*d_cell);
    if (*d_s_cell) cudaFree(*d_s_cell);
    if (*d_vx) cudaFree(*d_vx);
    if (*d_vy) cudaFree(*d_vy);
    if (*d_p) cudaFree(*d_p);
    if (*d_d) cudaFree(*d_d);
    if (*d_x_p) cudaFree(*d_x_p);
    if (*d_y_p) cudaFree(*d_y_p);
    if (*d_vx_p) cudaFree(*d_vx_p);
    if (*d_vy_p) cudaFree(*d_vy_p);
    if (*d_p_cell) cudaFree(*d_p_cell);
    if (*d_mom_x_cell) cudaFree(*d_mom_x_cell);
    if (*d_mom_y_cell) cudaFree(*d_mom_y_cell);
    if (*d_energy_cell) cudaFree(*d_energy_cell);
    if (*d_T) cudaFree(*d_T);
    if (*d_m_p_array) cudaFree(*d_m_p_array);
    if (*d_e_int_p) cudaFree(*d_e_int_p);
    if (*d_state) cudaFree(*d_state);
}

void Send_To_Device(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
                    float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
                    float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                    float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		    int num_cells ,int total_particles) {
    // Size of data to send
    size_t size1 =num_cells *sizeof(float);
    size_t size2 =total_particles *sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send A to the GPU
    Error = cudaMemcpy(*d_cell, *h_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_cell -> d_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_s_cell, *h_s_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_s_cell -> d_s_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_vx, *h_vx, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_vx -> d_vx) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_vy, *h_vy, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_vy -> d_vy) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_p, *h_p, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_p -> d_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_d, *h_d, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_d -> d_d) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_x_p, *h_x_p, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_x_p -> d_x_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_y_p, *h_y_p, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_y_p -> d_y_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_vx_p, *h_vx_p, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_vx_p -> d_vx_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_vy_p, *h_vy_p, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_vy_p -> d_vy_p) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_p_cell, *h_p_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_p_cell -> d_p_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_mom_x_cell, *h_mom_x_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_mom_x_cell -> d_mom_x_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_mom_y_cell, *h_mom_y_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_mom_y_cell -> d_mom_y_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_energy_cell, *h_energy_cell, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_energy_cell -> d_energy_cell) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_T, *h_T, size1, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_T -> d_T) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_m_p_array, *h_m_p_array, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_m_p_array -> d_m_p_array) = %s\n", cudaGetErrorString(Error));
    Error = cudaMemcpy(*d_e_int_p, *h_e_int_p, size2, cudaMemcpyHostToDevice);
    printf("CUDA error (memcpy h_e_int_p -> d_e_int_p) = %s\n", cudaGetErrorString(Error));
}

void Get_From_Device(float **h_d,float **d_d,int num_cells) {
    // Size of data to send
    size_t size1 =num_cells *sizeof(float);
    // Grab a error type
    cudaError_t Error;
    // Send d_a to the host variable h_a
    Error = cudaMemcpy(*h_d, *d_d, size1, cudaMemcpyDeviceToHost);
    printf("CUDA error (memcpy d_d -> h_d) = %s\n", cudaGetErrorString(Error));
}

void gpu_init_rand(curandState *d_states,int total_particles, unsigned long seed) {
    int blockSize = 256;
    int numBlocksParticles = (total_particles + blockSize - 1) / blockSize;
    init_curand_states<<<numBlocksParticles, blockSize>>>(d_states, total_particles, seed);
    cudaDeviceSynchronize();
}

void Compute_New_Density(float *h_cell,float *h_s_cell,float *h_vx,float *h_vy,float *h_p,float *h_d,float *h_x_p,float *h_y_p,float *h_vx_p,float *h_vy_p,
                         float *h_p_cell, float *h_mom_x_cell,float *h_mom_y_cell,float *h_energy_cell,float *h_T,float *h_m_p_array,float *h_e_int_p,
                         float *d_cell,float *d_s_cell,float *d_vx,float *d_vy,float *d_p,float *d_d,float *d_x_p,float *d_y_p,float *d_vx_p,float *d_vy_p,
                         float *d_p_cell, float *d_mom_x_cell,float *d_mom_y_cell,float *d_energy_cell,float *d_T,float *d_m_p_array,float *d_e_int_p,
			 int num_cells ,int total_particles,curandState *d_states,int NX,int NY,int P_NUMBER,float DX,float DY,float DT,float R_GAS) {
    /*
    dim3 Threads(4,1,1);
    dim3 Block(5,1);
    */
    int blockSize = 256;
    int numBlocksParticles = (total_particles + blockSize - 1) / blockSize;
    int numBlocksCells = (num_cells + blockSize - 1) / blockSize;

    generate_particles_kernel<<<numBlocksParticles, blockSize>>>(d_x_p, d_y_p, d_vx_p, d_vy_p, NX, NY, P_NUMBER, DX, DY,
                                                                 d_p_cell, d_d, d_s_cell, d_vx, d_vy, d_T, d_m_p_array, d_e_int_p, d_states, R_GAS);

    move_particles_kernel<<<numBlocksParticles, blockSize>>>(d_x_p, d_y_p, d_vx_p, d_vy_p, total_particles, DX, DY, DT);

    reset_grid_kernel<<<numBlocksCells, blockSize>>>(d_p_cell, d_mom_x_cell, d_mom_y_cell, d_energy_cell, num_cells);

    project_to_grid_kernel<<<numBlocksParticles, blockSize>>>(d_x_p, d_y_p, d_vx_p, d_vy_p, d_m_p_array, d_e_int_p,
                                                              d_p_cell, d_mom_x_cell, d_mom_y_cell, d_energy_cell,
                                                              NX, NY, DX, DY, total_particles);

    update_grid_kernel<<<numBlocksCells, blockSize>>>(d_p_cell, d_mom_x_cell, d_mom_y_cell, d_energy_cell,
                                                      d_vx, d_vy, d_T, d_d, d_p, NX, NY, DX, DY, num_cells, R_GAS);
    cudaDeviceSynchronize(); // 確保每個 STEP 的 GPU 計算完成再進入下一步

}
