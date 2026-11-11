#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"function.h"
#include"function1.h"
#include"function2.h"


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
	float *cell,*s_cell,*vx,*vy,*p,*d,*x_p,*y_p,*vx_p,*vy_p,*p_cell,*mom_x_cell,*mom_y_cell,*energy_cell,*T,*m_p_array,*e_int_p ;
	cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	vx=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	vy=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	x_p=(float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
        y_p=(float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
	vx_p=(float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
        vy_p=(float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
	p=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	d=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	T=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	s_cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	p_cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	mom_x_cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
        mom_y_cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
        energy_cell=(float*)malloc((NX+2)*(NY+2)*sizeof(float));
	m_p_array = (float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
	e_int_p = (float*)malloc((NX+2)*(NY+2)*P_NUMBER*sizeof(float));
	//define cell
	for(int i=0;i<NX+2;i++){
		for(int j=0;j<NY+2;j++){
			int index=i*(NY+2)+j;
			float cx=(0.5+i)*DX;
			float cy=(0.5+j)*DY;
			cell[index]=0;
			s_cell[index]=0;
			p[index]=1.0;
                        d[index]=1.4;
			T[index]=p[index]/(d[index]*R_GAS);
                        vx[index]=3.0;
                        vy[index]=0.0;
			p_cell[index]=d[index] * DX * DY;
        		mom_x_cell[index]=0;
       	 		mom_y_cell[index]=0;
        		energy_cell[index]=0;
			if( cx >= 0.6+DX && cx <= 3.0+DX && cy >= 0.0+DY && cy <= 0.2+DY){
                                s_cell[index] = 1;
			}
		}
	}
	for(int i=0;i<STEP;i++){
		produce_particle(x_p,y_p,vx_p,vy_p,NX,NY,P_NUMBER,DT,p_cell,DX,DY,mom_x_cell,mom_y_cell ,energy_cell,d,R_GAS,s_cell,vx,vy ,T,p,m_p_array,e_int_p);

		move_particle(x_p,y_p,vx_p,vy_p,DT,DX,DY,P_NUMBER,NX,NY);

		compute_particle(x_p,y_p,vx_p,vy_p,NX,NY,p_cell,DX,DY,mom_x_cell,mom_y_cell ,energy_cell,d,R_GAS,P_NUMBER,vx,vy ,T,p,m_p_array,e_int_p);
	}

	FILE *pFile;
    	pFile=fopen("results.txt","w");
    	for (int i = 1; i < NX+1; i++) {
                for (int j = 1; j < NY+1; j++) {
                        int index=i*(NY+2)+j;
			float x=(i+0.5)*DX;
			float y=(j+0.5)*DY;
                        fprintf(pFile,"%g\t,%g\t,%g\n",x,y,d[index]);
                }
    	}
	fclose(pFile);
	printf("simulation finish(模擬結束)\n");
/*
	for(int i=0;i<STEP;i++){
	if ((i + 1) % 1000 == 0) {

            int sec = (i + 1) / 1000;
            printf("正在輸出第 %d.0 秒的數據...\n", sec);

            // 自動生成不同的檔名 (例如 results_1.0s.txt)
            char filename[50];
            sprintf(filename, "results_%d.0s.txt", sec);

            pFile = fopen(filename, "w");
            if (pFile != NULL) {
                for (int j = 1; j < NX+2; j++) {
                    for (int k = 1; k < NY+2; k++) {
                        int index = j*(NY+2) + k;

                        // 計算當前這秒的密度
                        float current_d = p_cell[index] / (DX * DY);

                        // 網格中心座標
                        float cx = (j + 0.5) * DX;
                        float cy = (k + 0.5) * DY;

                        // 把階梯內部(實心牆壁)的密度強制設為 0，讓圖表乾淨
                        if (cx >= 0.6+DX && cy <= 0.2+DY) {
                            current_d = 0.0;
                        }

                        // 寫入檔案
                        fprintf(pFile, "%g\t,%g\t,%g\n", cx, cy, current_d);
                    }
                }
                fclose(pFile);
            } else {
                printf("Error: 無法建立檔案 %s\n", filename);
            }
        }
}
*/
free(cell);
free(vx);
free(vy);
free(p);
free(d);
free(T);
free(s_cell);
free(x_p);
free(y_p);
free(vx_p);
free(vy_p);
free(p_cell);
free(mom_x_cell);
free(mom_y_cell);
free(energy_cell);
free(m_p_array);
free(e_int_p);
return 0;
}
