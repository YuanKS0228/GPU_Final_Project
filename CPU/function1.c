#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void move_particle(float *x_p,float *y_p,float *vx_p,float *vy_p,float DT,float DX,float DY,int P_NUMBER,int NX,int NY){
	//total particle
        int index_2 = (NX+2)*(NY+2)*P_NUMBER;
	for(int i=0;i<index_2;i++){
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
return;
}

