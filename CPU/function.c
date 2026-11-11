#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void produce_particle(float *x_p,float *y_p,float *vx_p,float *vy_p,int NX,int NY,int P_NUMBER,float DT
		     ,float *p_cell,float DX,float DY,float *mom_x_cell,float *mom_y_cell ,float *energy_cell
		     ,float *d,int R_GAS,float *s_cell,float *vx,float *vy ,float *T,float *p,float *m_p_array,float *e_int_p){
	float my_rand(){
        	return (float)rand()/RAND_MAX;
	}
	float randn(){
        	float TWO_PI=2.0*M_PI;
        	float u1=(float)rand()/RAND_MAX;
        	float u2=(float)rand()/RAND_MAX;
        	float z1=sqrt(-2.0*log(u1))*cos(TWO_PI*u2);
        	return z1;
	}
	for(int j=0;j<NX+2;j++){
                for(int k=0;k<NY+2;k++){
                        int index=j*(NY+2)+k;
                        for(int z=0;z<P_NUMBER ;z++){
				int index_1 = (j*(NY+2)+k)*P_NUMBER+z;
                                if(s_cell[index] == 0 && k>0 && k<NY+1 && j>0){
                                	x_p[index_1]=j*DX+my_rand()*DX;
                                        y_p[index_1]=k*DY+my_rand()*DY;
                                        vx_p[index_1]=vx[index]+randn()*sqrt(R_GAS*T[index]);
                                        vy_p[index_1]=vy[index]+randn()*sqrt(R_GAS*T[index]);


				}else{
					if(j == 0){
						x_p[index_1]=j*DX+my_rand()*DX;
                                        	y_p[index_1]=k*DY+my_rand()*DY;
                                        	vx_p[index_1]=3.0+randn()*sqrt(R_GAS*T[index]);
                                        	vy_p[index_1]=0.0+randn()*sqrt(R_GAS*T[index]);
					}else{
                                        	x_p[index_1]=0.0;
                                        	y_p[index_1]=0.0;
                                        	vx_p[index_1]=0.0;
                                        	vy_p[index_1]=0.0;
					}
                                }

				if (j == 0) {
        			// 入口粒子永遠維持標準密度 1.4 對應的質量
        				m_p_array[index_1] = (1.4 * DX * DY) / P_NUMBER;
    				} else {
        			// 內部網格的粒子，平分該網格上一回合收集到的總質量
        				m_p_array[index_1] = p_cell[index] / P_NUMBER;
    				}
				e_int_p[index_1] = 1.5 * m_p_array[index_1] * R_GAS * T[index]; // 裝入 3/2 的隱形熱能

                        }
                }
	}
	/*for(int i=0;i<index_2;i++){
		if (0<x_p[i] && x_p[i]<0.6+DX && DY<y_p[i] && y_p[i]<0.2+DY){
               		x_p[i]=x_p[i]+vx_p[i]*DT;
                        y_p[i]=y_p[i]+vy_p[i]*DT;

			if (x_p[i]>0.6+DX){
                        	x_p[i]=2*(0.6+DX)-x_p[i];
                        	vx_p[i]=-vx_p[i];
                	}
                        if (y_p[i]<DY){
                               	y_p[i]=2*DY-y_p[i];
                                vy_p[i]=-vy_p[i];
                        }
		}//end of if

		if (0<x_p[i] && x_p[i]<0.6+DX && 0.2+DY<y_p[i] && y_p[i]<1+DY){
			x_p[i]=x_p[i]+vx_p[i]*DT;
                        y_p[i]=y_p[i]+vy_p[i]*DT;

                        if (y_p[i]>1+DY){
                      		y_p[i]=2*(1+DY)-y_p[i];
                        	vy_p[i]=-vy_p[i];
                        }
		}//end of if

		if (0.6+DX<x_p[i] && x_p[i]<3+DX && 0.2+DY<y_p[i] && y_p[i]<1.0+DY){
			x_p[i]=x_p[i]+vx_p[i]*DT;
                        y_p[i]=y_p[i]+vy_p[i]*DT;

                        if (y_p[i]<0.2+DY){
				y_p[i]=2*(0.2+DY)-y_p[i];
                                vy_p[i]=-vy_p[i];
                        }
                        if (y_p[i]>1.0+DY){
                                y_p[i]=2*(1+DY)-y_p[i];
                                vy_p[i]=-vy_p[i];
                        }
		}//end of if
	}//end of for

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
	*/
return;
}//end of function

