#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <curand_kernel.h>
#include"gpu.h"

#define R_GAS 1
#define K 1.4
#define D 1.4
#define P 1.0
#define L 3.0
#define H 1.0
#define NX atoi(argv[1])
#define NY atoi(argv[2])
#define DX (L/NX)
#define DY (H/NY)
#define DT 0.001
#define T_TARGET atoi(argv[3])
#define STEP (T_TARGET/DT)
#define P_NUMBER 100

int main(int argc, char *argv[]) {
    if(argc != 4) {
        printf("用法: ./solver <NX> <NY> <T_TARGET>\n");
        return -1;
    }

    int num_cells = (NX + 2) * (NY + 2);
    int total_particles = num_cells * P_NUMBER;
    float *h_cell, *h_s_cell, *h_vx, *h_vy, *h_p, *h_d, *h_x_p, *h_y_p, *h_vx_p, *h_vy_p, *h_p_cell, *h_mom_x_cell, *h_mom_y_cell, *h_energy_cell, *h_T, *h_m_p_array, *h_e_int_p;
    float *d_cell, *d_s_cell, *d_vx, *d_vy, *d_p, *d_d, *d_x_p, *d_y_p, *d_vx_p, *d_vy_p, *d_p_cell, *d_mom_x_cell, *d_mom_y_cell, *d_energy_cell, *d_T, *d_m_p_array, *d_e_int_p;
    curandState *d_states;
    Set_GPU_Device(0);
    Allocate_Memory(&h_cell,&h_s_cell,&h_vx,&h_vy,&h_p,&h_d,&h_x_p,&h_y_p,&h_vx_p,&h_vy_p,&h_p_cell,&h_mom_x_cell,&h_mom_y_cell,&h_energy_cell,&h_T,&h_m_p_array,&h_e_int_p,
                    &d_cell,&d_s_cell,&d_vx,&d_vy,&d_p,&d_d,&d_x_p,&d_y_p,&d_vx_p,&d_vy_p,&d_p_cell,&d_mom_x_cell,&d_mom_y_cell,&d_energy_cell,&d_T,&d_m_p_array,&d_e_int_p,
		    &d_states,num_cells,total_particles);

    for (int i = 0; i < NX + 2; i++) {
        for (int j = 0; j < NY + 2; j++) {
            int index = i * (NY + 2) + j;
            float cx = (0.5 + i) * DX;
            float cy = (0.5 + j) * DY;
            h_cell[index] = 0;
            h_s_cell[index] = 0;
            h_p[index] = 1.0;
            h_d[index] = 1.4;
            h_T[index] = h_p[index] / (h_d[index] * R_GAS);
            h_vx[index] = 3.0;
            h_vy[index] = 0.0;
            h_p_cell[index] = h_d[index] * DX * DY;
            h_mom_x_cell[index] = 0;
            h_mom_y_cell[index] = 0;
            h_energy_cell[index] = 0;
            if (cx >= 0.6f + DX && cx <= 3.0f + DX && cy >= 0.0f + DY && cy <= 0.2f + DY) {
                h_s_cell[index] = 1;
            }
        }
    }
    Send_To_Device(&h_cell,&h_s_cell,&h_vx,&h_vy,&h_p,&h_d,&h_x_p,&h_y_p,&h_vx_p,&h_vy_p,
                   &h_p_cell,&h_mom_x_cell,&h_mom_y_cell,&h_energy_cell,&h_T,&h_m_p_array,&h_e_int_p,
                   &d_cell,&d_s_cell,&d_vx,&d_vy,&d_p,&d_d,&d_x_p,&d_y_p,&d_vx_p,&d_vy_p,
                   &d_p_cell,&d_mom_x_cell,&d_mom_y_cell,&d_energy_cell,&d_T,&d_m_p_array,&d_e_int_p,
		   num_cells ,total_particles);
    printf("初始化 CURAND...\n");
    gpu_init_rand(d_states,total_particles,1234ULL);
    cudaDeviceSynchronize();

    printf("開始模擬...\n");
    // 主迴圈
    for (int i = 0; i < STEP; i++) {

        Compute_New_Density(h_cell,h_s_cell,h_vx,h_vy,h_p,h_d,h_x_p,h_y_p,h_vx_p,h_vy_p,
                            h_p_cell,h_mom_x_cell,h_mom_y_cell,h_energy_cell,h_T,h_m_p_array,h_e_int_p,
                            d_cell,d_s_cell,d_vx,d_vy,d_p,d_d,d_x_p,d_y_p,d_vx_p,d_vy_p,
                            d_p_cell,d_mom_x_cell,d_mom_y_cell,d_energy_cell,d_T,d_m_p_array,d_e_int_p,
			    num_cells,total_particles,d_states,NX,NY,P_NUMBER,DX,DY,DT,R_GAS);

        cudaDeviceSynchronize(); // 確保每個 STEP 的 GPU 計算完成再進入下一步
    }
    Get_From_Device(&h_d,&d_d,num_cells);

    FILE *pFile;
        pFile=fopen("results.txt","w");
        for (int i = 1; i < NX+2; i++) {
                for (int j = 1; j < NY+2; j++) {
                        int index=i*(NY+2)+j;
			float current_d = h_d[index];
                        float cx=(i+0.5)*DX;
                        float cy=(j+0.5)*DY;
			if (cx >= 0.6f + DX && cy <= 0.2f + DY) {
                        	current_d = 0.0f;
                        }
                        fprintf(pFile, "%g\t,%g\t,%g\n", cx, cy, current_d);
                }
        }
        fclose(pFile);

    /*
    for (int i = 0; i < STEP; i++) {
    	if ((i + 1) % 1000 == 0) {
            	int sec = (i + 1) / 1000;
            	printf("正在輸出第 %d.0 秒的數據...\n", sec);
            	char filename[50];
            	sprintf(filename, "results_%d.0s.txt", sec);
            	FILE *pFile = fopen(filename, "w");
            	if (pFile != NULL) {
                	for (int j = 1; j < NX + 2; j++) {
                    		for (int k = 1; k < NY + 2; k++) {
                        		int index = j * (NY + 2) + k;
                        		float current_d = h_d[index];
                        		float cx = (j + 0.5f) * DX;
                        		float cy = (k + 0.5f) * DY;
                        		if (cx >= 0.6f + DX && cy <= 0.2f + DY) {
                            		current_d = 0.0f;
                        		}
                        	fprintf(pFile, "%g\t,%g\t,%g\n", cx, cy, current_d);
                    		}
                	}
                fclose(pFile);
    		}
    	}
    }

    */
    printf("Hello (模擬結束)\n");

    // 釋放記憶體 (改用 cudaFree)
    Free_Memory(&h_cell,&h_s_cell,&h_vx,&h_vy,&h_p,&h_d,&h_x_p,&h_y_p,&h_vx_p,&h_vy_p,
                &h_p_cell,&h_mom_x_cell,&h_mom_y_cell,&h_energy_cell,&h_T,&h_m_p_array,&h_e_int_p,
                &d_cell,&d_s_cell,&d_vx,&d_vy,&d_p,&d_d,&d_x_p,&d_y_p,&d_vx_p,&d_vy_p,
                &d_p_cell,&d_mom_x_cell,&d_mom_y_cell,&d_energy_cell,&d_T,&d_m_p_array,&d_e_int_p,&d_states);
    return 0;
}
