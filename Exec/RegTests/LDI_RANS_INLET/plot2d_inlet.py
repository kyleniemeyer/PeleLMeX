import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import scipy.interpolate as spi



#Center data locations for all 7 cups
cen_y_swirler_0 =  0.000000000000 
cen_z_swirler_0 =  0.000000000000

cen_y_swirler_1 = -0.011904858869 
cen_z_swirler_1 =  0.020620105463

cen_y_swirler_2 = -0.023809960000 
cen_z_swirler_2 =  0.000000000000

cen_y_swirler_3 = -0.011904858869 
cen_z_swirler_3 = -0.020620105463

cen_y_swirler_4 =  0.011904858869 
cen_z_swirler_4 = -0.020620105463

cen_y_swirler_5 =  0.023809960000 
cen_z_swirler_5 =  0.000000000000

cen_y_swirler_6 =  0.011904858869 
cen_z_swirler_6 =  0.020620105463

inlet_annulus_r_max=0.01016500
inlet_annulus_r_min=0.00337055

num_of_x_cells=100j
num_of_y_cells=100j
num_of_z_cells=100j

nx=int(num_of_x_cells.imag)
ny=int(num_of_y_cells.imag)
nz=int(num_of_z_cells.imag)




def Write_Intermediate_DataFile(FluentFilename,OutputFileName,center_x,center_y):
    x_bound_lw = center_x-0.01016
    x_bound_up = center_x+0.01016
    dx=(x_bound_up-x_bound_lw)/nx
    y_bound_lw = center_y-0.01016
    y_bound_up = center_y+0.01016
    dy=(y_bound_up-y_bound_lw)/ny
    dz=dy
    z_bound_lw = -0.01016
    z_bound_up = z_bound_lw+dz*num_of_z_cells
    
    data = np.genfromtxt(FluentFilename, dtype=None,skip_header=1,delimiter=",")    
    points=np.zeros((len(data),2))
    u_values=np.zeros(len(data))
    v_values=np.zeros(len(data))
    w_values=np.zeros(len(data))
    ctr=0
    for i in data:
        points[ctr,0]=i[1]
        points[ctr,1]=i[2]
        u_values[ctr]=i[6]
        v_values[ctr]=i[7]
        w_values[ctr]=i[8]
        ctr=ctr+1

    grid_x, grid_y = np.mgrid[x_bound_lw:x_bound_up:num_of_x_cells, y_bound_lw:y_bound_up:num_of_y_cells]
    grid_x_3d, grid_y_3d, grid_z_3d = np.mgrid[x_bound_lw:x_bound_up:num_of_x_cells, y_bound_lw:y_bound_up:num_of_y_cells, z_bound_lw:z_bound_up:num_of_z_cells]

    grid_uval_interp = griddata(points, u_values, (grid_x, grid_y), method='cubic')
    grid_vval_interp = griddata(points, v_values, (grid_x, grid_y), method='cubic')
    grid_wval_interp = griddata(points, w_values, (grid_x, grid_y), method='cubic')

    Xr=grid_x_3d.reshape(-1,order="F")
    Yr=grid_y_3d.reshape(-1,order="F")
    Zr=grid_z_3d.reshape(-1,order="F")
    
    #Remove all nans create from interpolation
    np.nan_to_num(grid_uval_interp,copy=False, nan=0.0, posinf=None, neginf=None)
    np.nan_to_num(grid_vval_interp,copy=False, nan=0.0, posinf=None, neginf=None)
    np.nan_to_num(grid_wval_interp,copy=False, nan=0.0, posinf=None, neginf=None)

    for i in np.arange(nx):
        for j in np.arange(ny):
            rad=((grid_x[i,j]-center_x)**2.0+(grid_y[i,j]-center_y)**2.0)**0.5
            if(rad<inlet_annulus_r_min):
                grid_wval_interp[i,j]=0.0
                grid_uval_interp[i,j]=0.0
                grid_vval_interp[i,j]=0.0

    plt.subplot(111)
    plt.imshow(grid_uval_interp.T, extent=(x_bound_lw,x_bound_up,y_bound_lw,y_bound_up), origin='lower')
    grid_uval_interp_3d=np.zeros((nx,ny,nz))
    grid_vval_interp_3d=np.zeros((nx,ny,nz))
    grid_wval_interp_3d=np.zeros((nx,ny,nz))

    for k in np.arange(nz):
        grid_uval_interp_3d[:,:,k]=grid_uval_interp
        grid_vval_interp_3d[:,:,k]=grid_vval_interp
        grid_wval_interp_3d[:,:,k]=grid_wval_interp

    Ur=grid_uval_interp_3d.reshape(-1,order="F")
    Vr=grid_vval_interp_3d.reshape(-1,order="F")
    Wr=grid_wval_interp_3d.reshape(-1,order="F")
    
    data_to_write = np.vstack((Xr, Yr, Zr, Ur, Vr, Wr)).T
    np.savetxt(OutputFileName, data_to_write, fmt="%.18e", delimiter=",", header="x, y, z, u, v, w")


def ShowContours(IntermediateFileName,var_to_plot,index_to_plot,lim_min,lim_max):
    data_read = np.genfromtxt(IntermediateFileName, dtype=None,skip_header=1,delimiter=",")
    
    X_new = data_read[:,0]
    Y_new = data_read[:,1]
    U_new = data_read[:,var_to_plot]
    print(np.min(U_new))
    
    X_n = X_new.reshape(nz,ny,nx).T
    Y_n = Y_new.reshape(nz,ny,nx).T
    U_n = U_new.reshape(nz,ny,nx).T
    
    x=X_n[:,:,index_to_plot]
    y=Y_n[:,:,index_to_plot]
    u=U_n[:,:,index_to_plot]

    fig, ax = plt.subplots(figsize=(6,6))
    levels = np.linspace(lim_min,lim_max)
    cf=ax.contourf(x,y,u,levels=levels,cmap='rainbow')
    fig.colorbar(cf, ax=ax)
    plt.show()    
    
    
def main():
    Write_Intermediate_DataFile("inlet_0","LDI_RANS_INLET0.dat",cen_z_swirler_0,cen_y_swirler_0)
    Write_Intermediate_DataFile("inlet_1","LDI_RANS_INLET1.dat",cen_z_swirler_1,cen_y_swirler_1)
    Write_Intermediate_DataFile("inlet_2","LDI_RANS_INLET2.dat",cen_z_swirler_2,cen_y_swirler_2)
    Write_Intermediate_DataFile("inlet_3","LDI_RANS_INLET3.dat",cen_z_swirler_3,cen_y_swirler_3)
    Write_Intermediate_DataFile("inlet_4","LDI_RANS_INLET4.dat",cen_z_swirler_4,cen_y_swirler_4)
    Write_Intermediate_DataFile("inlet_5","LDI_RANS_INLET5.dat",cen_z_swirler_5,cen_y_swirler_5)
    Write_Intermediate_DataFile("inlet_6","LDI_RANS_INLET6.dat",cen_z_swirler_6,cen_y_swirler_6)
    #ShowContours("LDI_RANS_INLET0.dat",3,2,-94,94)
main()
    
    
            
