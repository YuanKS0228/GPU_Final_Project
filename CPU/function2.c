#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void compute_particle(float *x_p,float *y_p,float *vx_p,float *vy_p,int NX,int NY,float *p_cell
		     ,float DX,float DY,float *mom_x_cell,float *mom_y_cell ,float *energy_cell
                     ,float *d,int R_GAS,int P_NUMBER,float *vx,float *vy ,float *T,float *p,float *m_p_array,float *e_int_p){
	//total particle
        int index_2 = (NX+2)*(NY+2)*P_NUMBER;
	for(int i = 0; i < (NX+2)*(NY+2); i++){
                p_cell[i] = 0.0;
                mom_x_cell[i] = 0.0;
                mom_y_cell[i] = 0.0;
                energy_cell[i] = 0.0;
        }

        for(int i=0;i<index_2;i++){
                if(x_p[i] == 0.0 && y_p[i] == 0.0) {
                        continue;
                }
                int j=(int)(x_p[i]/DX);
                int k=(int)(y_p[i]/DY);
                if(j>0 && j<NX+1 && k>0 && k<NY+1){
                        int index=j*(NY+2)+k;
                        p_cell[index] += m_p_array[i];
                        mom_x_cell[index] += m_p_array[i] * vx_p[i];
                        mom_y_cell[index] += m_p_array[i] * vy_p[i];
                        energy_cell[index] += 0.5 * m_p_array[i] * (vx_p[i]*vx_p[i] + vy_p[i]*vy_p[i])+e_int_p[i];
                }
        }

        for(int j=0;j<NX+2;j++){
                for(int k=0;k<NY+2;k++){
                        int index=j*(NY+2)+k;
                        if (p_cell[index] > 1e-10) {
                                vx[index]=mom_x_cell[index]/p_cell[index];
                                vy[index]=mom_y_cell[index]/p_cell[index];
                                float E_TOTAL=0.5*p_cell[index]*(vx[index]*vx[index]+vy[index]*vy[index]);
                                float E_THERMAL=energy_cell[index]-E_TOTAL;
                                T[index]=E_THERMAL/(2.5*R_GAS*p_cell[index]);
                                d[index]=p_cell[index]/(DX*DY);
                                p[index] = d[index] * R_GAS * T[index];
                        }
                }
        }
return;
}
