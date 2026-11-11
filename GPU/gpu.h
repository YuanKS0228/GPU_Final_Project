#include <curand_kernel.h>
void Set_GPU_Device(int device);
void Allocate_Memory(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
                     float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
		     float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                     float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		     curandState **d_states,int num_cells,int total_particles);
void Free_Memory(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
                 float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
                 float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                 float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		 curandState **d_states);

void Send_To_Device(float **h_cell,float **h_s_cell,float **h_vx,float **h_vy,float **h_p,float **h_d,float **h_x_p,float **h_y_p,float **h_vx_p,float **h_vy_p,
                    float **h_p_cell, float **h_mom_x_cell,float **h_mom_y_cell,float **h_energy_cell,float **h_T,float **h_m_p_array,float **h_e_int_p,
                    float **d_cell,float **d_s_cell,float **d_vx,float **d_vy,float **d_p,float **d_d,float **d_x_p,float **d_y_p,float **d_vx_p,float **d_vy_p,
                    float **d_p_cell, float **d_mom_x_cell,float **d_mom_y_cell,float **d_energy_cell,float **d_T,float **d_m_p_array,float **d_e_int_p,
		    int num_cells,int total_particles);

void Get_From_Device(float **h_d,float **d_d,int num_cells);

void gpu_init_rand(curandState *d_states,int total_particles, unsigned long seed);

void Compute_New_Density(float *h_cell,float *h_s_cell,float *h_vx,float *h_vy,float *h_p,float *h_d,float *h_x_p,float *h_y_p,float *h_vx_p,float *h_vy_p,
                         float *h_p_cell, float *h_mom_x_cell,float *h_mom_y_cell,float *h_energy_cell,float *h_T,float *h_m_p_array,float *h_e_int_p,
                         float *d_cell,float *d_s_cell,float *d_vx,float *d_vy,float *d_p,float *d_d,float *d_x_p,float *d_y_p,float *d_vx_p,float *d_vy_p,
                         float *d_p_cell,float *d_mom_x_cell,float *d_mom_y_cell,float *d_energy_cell,float *d_T,float *d_m_p_array,float *d_e_int_p,
			 int num_cells,int total_particles,curandState *d_states,int NX,int NY,int P_NUMBER,float DX,float DY,float DT,float R_GAS);
