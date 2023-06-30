#ifndef PELELM_PROB_H_
#define PELELM_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLM_Index.H>
#include <pelelm_prob_parm.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>

// -----------------------------------------------------------
// Search for the closest index in an array to a given value
// using the bisection technique.
// INPUTS/OUTPUTS:
// xtable(0:n-1) => array to search in (ascending order)
// n             => array size
// x             => x location
// idxlo        <=> output st. xtable(idxlo) <= x < xtable(idxlo+1)
// -----------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
locate(const amrex::Real* xtable, const int n, const amrex::Real& x, int& idxlo)
{
  // If x is out of bounds, return boundary index
  if (x >= xtable[n - 1]) {
    idxlo = n - 1;
    return;
  }
  if (x <= xtable[0]) {
    idxlo = 0;
    return;
  }

  // Do the bisection
  idxlo = 0;
  int idxhi = n - 1;
  bool notdone = true;
  while (notdone) {
    if (idxhi - idxlo <= 1) {
      notdone = false;
    } else {
      const int idxmid = (idxhi + idxlo) / 2;
      if (x >= xtable[idxmid]) {
        idxlo = idxmid;
      } else {
        idxhi = idxmid;
      }
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
set_Y_from_Phi(const amrex::Real phi_in,
               amrex::Real Y[])
{
   auto eos = pele::physics::PhysicsType::eos();
   amrex::Real Xt[NUM_SPECIES] = {0.0};
   amrex::Real a = 18.5;  // For NC12H26
   Xt[O2_ID] = 1.0 / ( 1.0 + phi_in / a + 0.79 / 0.21 );
   Xt[NC12H26_ID] = phi_in * Xt[O2_ID] / a;
   Xt[N2_ID] = 1.0 - Xt[O2_ID] - Xt[NC12H26_ID];

   eos.X2Y(Xt,Y);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void pelelm_initdata(int i, int j, int k,
                     int is_incompressible,
                     amrex::Array4<amrex::Real> const& state,
                     amrex::Array4<amrex::Real> const& /*aux*/,
                     amrex::GeometryData const& geomdata,
                     ProbParm const& prob_parm,
                     pele::physics::PMF::PmfData::DataContainer const * pmf_data)
{
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx      = geomdata.CellSize();

    amrex::Real x[3] = {
      prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
      prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
      prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

    AMREX_D_TERM(const amrex::Real Lx = prob_hi[0] - prob_lo[0];,
                 const amrex::Real Ly = prob_hi[1] - prob_lo[1];,
                 const amrex::Real Lz = prob_hi[2] - prob_lo[2]);

    AMREX_D_TERM(const amrex::Real xc = prob_lo[0] + 0.5 * Lx;,
                 const amrex::Real yc = prob_lo[1] + 0.5 * Ly;,
                 const amrex::Real zc = prob_lo[2] + 0.5 * Lz;);

    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    // Species mass
    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;
    state(i,j,k,TEMP) = prob_parm.T_mean;

    if (prob_parm.do_ignit) {
        // Add hot air in front of the premixers:
        amrex::GpuArray<amrex::Real,2*4> prem_centers{-0.0163322, 0.0163322,
                                                       0.0163322, 0.0163322,
                                                      -0.0163322,-0.0163322,
                                                       0.0163322,-0.0163322};
        for (int pm = 0; pm < 4; ++pm) {
          amrex::Real z_center = prob_lo[2] + prob_parm.ignit_lowZ;
          amrex::Real pm_rad = sqrt(  (x[0]-prem_centers[2*pm])  *(x[0]-prem_centers[2*pm])
                                    + (x[1]-prem_centers[2*pm+1])*(x[1]-prem_centers[2*pm+1])
                                    + (x[2]-z_center)*(x[2]-z_center));
          amrex::Real tanh_dist = 0.5 * (1.0 - std::tanh((std::abs(pm_rad)-prob_parm.ignit_rad)/0.001));
          state(i,j,k,TEMP) += tanh_dist * (prob_parm.ignit_temp-prob_parm.T_mean);
        }
    }

    state(i,j,k,VELX) = 0.0;
    state(i,j,k,VELY) = 0.0;
    state(i,j,k,VELZ) = 0.0;

    amrex::Real P_cgs = prob_parm.P_mean*10;

    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i,j,k,TEMP), rho_cgs);
    state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i,j,k,TEMP), massfrac, h_cgs);
    state(i,j,k,RHOH) = h_cgs * 1.0e-4 * state(i,j,k,DENSITY);

    for (int n = 0; n < NUM_SPECIES; n++) {
           state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
    }

}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
time_eta(
  amrex::Real time, amrex::Real start, amrex::Real duration, amrex::Real tau)
{
  amrex::Real eta_time;
  eta_time = 0.5 * (std::tanh((time - start) / (duration * tau)) *
                    std::tanh((-time + start + duration) / (duration * tau)) +
                    1.0);
  eta_time = std::min(std::max(eta_time, 0.0), 1.0);
  return eta_time;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
Get_Axial_Velocity(amrex::Real rad)
{
   amrex::Real rad2,rad3,rad4;
   amrex::Real Vax;
   rad2=rad*rad;
   rad3=rad2*rad;
   rad4=rad2*rad2;
   if(rad<0.272546154)
   {
      Vax = -87.333*rad4 + 70.653*rad3 - 12.225*rad2+ 1.0613*rad - 0.2365;
   }
   else if(rad>=0.272546154 && rad<0.83056)
   {
      Vax = -4.1043*rad4 + 13.292*rad3 - 16.291*rad2 + 9.5823*rad - 1.5582;
   }
   else if(rad>=0.83056 && rad<=1.0)
   {
      Vax = -1017.7*rad4 + 3734.4*rad3 - 5124.4*rad2 + 3117.6*rad - 708.9;
   }
   return Vax;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
Get_Tang_Velocity(amrex::Real rad)
{
   amrex::Real rad2,rad3,rad4;
   amrex::Real Vt;
   rad2=rad*rad;
   rad3=rad2*rad;
   rad4=rad2*rad2;
   if(rad<0.3023)
   {
      Vt =  -161.32*rad4 + 127.4*rad3 - 30.622*rad2 + 4.0507*rad - 0.1307;
   }
   else if(rad>=0.3023 && rad<0.9489)
   {
      Vt = -4.1079*rad4+ 13.818*rad3 - 17.652*rad2 + 10.342*rad - 1.3959 ;
   }
   else if(rad>=0.9489 && rad<=1)
   {
      Vt = -1.7043*rad+2.6172 ;
   }
   else
   {
   // amrex::Print()<<"\nrad out of bounds "<<rad;
   }
   return Vt;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
Get_Rad_Velocity(amrex::Real rad)
{
   amrex::Real rad2,rad3,rad4;
   amrex::Real Vr;
   rad2=rad*rad;
   rad3=rad2*rad;
   rad4=rad2*rad2;
   if(rad<0.33912)
  {
   Vr = -194.09*rad4 + 176.94*rad3 - 47.218*rad2 + 6.2139*rad - 0.1896;
  }
  else if(rad>=0.33912 && rad<0.877387692)
  {
   Vr = -28.831*rad4 + 80.084*rad3 - 90.188*rad2 + 42.115*rad - 5.8326 ;
  }
  else if(rad>=0.877387692 && rad<=1.0)
  {
   Vr = -49175*rad4+ 179772*rad3 - 246359*rad2 + 149975*rad - 34219;
  }
  else
  {
    //amrex::Print()<<"\nError rad out of bounds vr "<<rad;
   }
   return Vr;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
CalculateRadiusOnPlane(amrex::Real xval,
                       amrex::Real yval,
                       amrex::Real x_center,
                       amrex::Real y_center)
{
   return sqrt((xval-x_center)*(xval-x_center)+((yval-y_center)*(yval-y_center)));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const int m_nAux,
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const *pmf_data)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;
  constexpr amrex::Real deg_to_radians = Pi/180.0;
  auto eos = pele::physics::PhysicsType::eos();

  // Radial coordinate of the point on boundary. The inlets lie on lowZ X-Y plane
  amrex::Real x_c = x[0];
  amrex::Real y_c = x[1];
  amrex::Real z_c = x[2];
  amrex::Real radius = sqrt(x[0]*x[0]+x[1]*x[1]);

  // Default chamber gas composition
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  massfrac[O2_ID] = 0.233;
  massfrac[N2_ID] = 0.767;

  // Set up premixer composition
  amrex::GpuArray<amrex::Real,2*4> prem_centers{-0.0163322, 0.0163322,
                                                 0.0163322, 0.0163322,
                                                -0.0163322,-0.0163322,
                                                 0.0163322,-0.0163322};
  for (int pm = 0; pm < 4; ++pm) {
    amrex::Real pm_rad = sqrt(  (x[0]-prem_centers[2*pm])  *(x[0]-prem_centers[2*pm])
                              + (x[1]-prem_centers[2*pm+1])*(x[1]-prem_centers[2*pm+1]));
    if (pm_rad < 0.0083312 && pm_rad >= 0.0057404) {
       set_Y_from_Phi(prob_parm.premix_phi,massfrac);
    }
  }

  // Thermo/fluid variables
  amrex::Real p_cgs = prob_parm.P_mean * 10.0;		//From user input and converting it into cgs unit
  amrex::Real rho_cgs = 0.0;
  amrex::Real RhoH_temp = 0.0;
  s_ext[TEMP] = prob_parm.T_mean;			//From user input
  eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
  s_ext[DENSITY] = rho_cgs * 1.0e3;			// CGS -> MKS conversion
  eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
  s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];    // CGS -> MKS conversion


  for (int n = 0; n < NUM_SPECIES; n++) 		// Specifying inflow gas composition
  {
     s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
  }


  bool use_strip_method_for_bafflecoolingholes=false;

  //Top baffle cooling holes
  amrex::Real ymax_bch_top = (66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real ymin_bch_top = (66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real xmin_bch_top = -62.865*0.001;
  amrex::Real xmax_bch_top =  62.865*0.001;

  //Bottom baffle cooling holes
  amrex::Real ymax_bch_bot = (-66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real ymin_bch_bot = (-66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real xmin_bch_bot = -62.865*0.001;
  amrex::Real xmax_bch_bot =  62.865*0.001;

  //front baffle cooling holes
  amrex::Real xmax_bch_frt = (66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real xmin_bch_frt = (66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real ymin_bch_frt = -62.865*0.001;
  amrex::Real ymax_bch_frt =  62.865*0.001;

  //back baffle cooling holes
  amrex::Real xmax_bch_bak = (-66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real xmin_bch_bak = (-66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  amrex::Real ymin_bch_bak = -62.865*0.001;
  amrex::Real ymax_bch_bak =  62.865*0.001;

  const int num_of_baffle_cooling_holes=34;
  amrex::Real radius_baffle_cooling_holes=0.79375*0.001;
  amrex::Real dist_between_bch=3.81*0.001;

  amrex::Real zvel_bch=100.0;

  //Outer dome cooling holes
  amrex::Real radius_outer_dome_cooling_hole=0.4445*0.001;
  constexpr amrex::Real radius_odchs=64.18579658*0.001;
  amrex::Real rmax_outer_dch=radius_odchs+radius_outer_dome_cooling_hole;
  amrex::Real rmin_outer_dch=radius_odchs-radius_outer_dome_cooling_hole;
  int num_of_outer_dch=52;
  amrex::Real delta_theta=360.0/52.0;
  amrex::Array<amrex::Real,3> odch_point0_normal     ={0.0,0.0,0.7244707034};

  //Inner dome cooling holes
  amrex::Real radius_inner_dome_cooling_hole=0.8255*0.001;
  constexpr amrex::Real radius_idchs=58.42*0.001;
  amrex::Real rmax_inner_dch=radius_idchs+radius_inner_dome_cooling_hole;
  amrex::Real rmin_inner_dch=radius_idchs-radius_inner_dome_cooling_hole;
  int num_of_inner_dch=52;
  amrex::Array<amrex::Real,3> idch_point0_normal     ={0.0,0.0,0.7095693667};

  const amrex::Real lbm_to_kg=0.453592;
  amrex::Real mfr_odch=0.02416*lbm_to_kg;
  amrex::Real mfr_idch=0.06464*lbm_to_kg;
  amrex::Real mfr_bch=0.09*2*lbm_to_kg;

  amrex::Real area_per_odch=Pi*(0.8255*0.001)*(0.8255*0.001);
  amrex::Real area_per_idch=Pi*(0.4445*0.001)*(0.4445*0.001);
  amrex::Real area_per_bch =Pi*(0.79375*0.001)*(0.79375*0.001);

  amrex::Real vel_mag_odch=mfr_odch/(area_per_odch*s_ext[DENSITY]*num_of_outer_dch);
  amrex::Real vel_mag_idch=mfr_idch/(area_per_idch*s_ext[DENSITY]*num_of_inner_dch);
  amrex::Real vel_mag_bch=mfr_bch/(area_per_bch*s_ext[DENSITY]*num_of_baffle_cooling_holes*4);


  //Use a strip for modeling baffle cooling holes
  if(use_strip_method_for_bafflecoolingholes)
    {
      //top
      if(y_c>=ymin_bch_top and y_c<=ymax_bch_top and x_c>=xmin_bch_top and x_c<=xmax_bch_top)
        {
          s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
          s_ext[VELY] = 0.0;
          s_ext[VELZ] = zvel_bch;

        } //bottom
      else if(y_c>=ymax_bch_bot and y_c<=ymin_bch_bot and x_c>=xmin_bch_bot and x_c<=xmax_bch_bot)
        {
          s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                    s_ext[VELY] = 0.0;
                    s_ext[VELZ] = zvel_bch;
        } //front
      else if(y_c>=ymin_bch_frt and y_c<=ymax_bch_frt and x_c>=xmin_bch_frt and x_c<=xmax_bch_frt)
        {
          s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                    s_ext[VELY] = 0.0;
                    s_ext[VELZ] = zvel_bch;
        }
      else if (y_c<=ymax_bch_bak and y_c>=ymin_bch_bak and x_c>=xmax_bch_bak and x_c<=xmin_bch_bak)
        {
          s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                    s_ext[VELY] = 0.0;
                    s_ext[VELZ] = zvel_bch;
        }
    }
  else  //Use circular patches as inlets for baffle cooling holes
    {
      if(y_c>=ymin_bch_top and y_c<=ymax_bch_top and x_c>=xmin_bch_top-radius_baffle_cooling_holes and x_c<=xmax_bch_top+radius_baffle_cooling_holes)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              amrex::Real y_center_hole=0.5*(ymin_bch_top+ymax_bch_top);
              amrex::Real x_center_hole=xmin_bch_top+i*dist_between_bch;
              amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes)
                {
                 	s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                        s_ext[VELY] = 0.0;
                        s_ext[VELZ] = vel_mag_bch;
                }
            }
        }
      else if(y_c>=ymax_bch_bot and y_c<=ymin_bch_bot and x_c>=xmin_bch_bot and x_c<=xmax_bch_bot)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              amrex::Real y_center_hole=0.5*(ymin_bch_bot+ymax_bch_bot);
              amrex::Real x_center_hole=xmin_bch_bot+i*dist_between_bch;
              amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes)
                {
                  s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                            s_ext[VELY] = 0.0;
                            s_ext[VELZ] = vel_mag_bch;
                }
            }
        } //front
      else if(y_c>=ymin_bch_frt and y_c<=ymax_bch_frt and x_c>=xmin_bch_frt and x_c<=xmax_bch_frt)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              amrex::Real y_center_hole=ymin_bch_frt+i*dist_between_bch;
              amrex::Real x_center_hole=0.5*(xmin_bch_frt+xmax_bch_frt);
              amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes)
                {
                  s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch;
                }
            }
        }
      else if (y_c<=ymax_bch_bak and y_c>=ymin_bch_bak and x_c>=xmax_bch_bak and x_c<=xmin_bch_bak)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              amrex::Real y_center_hole=ymin_bch_bak+i*dist_between_bch;
              amrex::Real x_center_hole=0.5*(xmin_bch_bak+xmax_bch_bak);
              amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes)
                {
                  s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch;
                }
            }
        }



    }

  //Model outer dome cooling holes
  if(radius<=rmax_outer_dch and radius>=rmin_outer_dch)
    {
      for(amrex::Real theta=0.0;theta<360;theta=theta+delta_theta)
        {
          amrex::Real theta_in_radians=theta*deg_to_radians;
          amrex::Real y_center_odch=radius_odchs*sin(theta_in_radians);
          amrex::Real x_center_odch=-radius_odchs*cos(theta_in_radians);
          //amrex::Print()<<"\n "<<y_center_odch<<" "<<x_center_odch;
          amrex::Real rad=sqrt(std::pow((x[1]-y_center_odch),2.0)+pow((x[0]-x_center_odch),2.0));
          amrex::Real nz = x[2]-odch_point0_normal[2];
          amrex::Real ny = y_center_odch-odch_point0_normal[1];
          amrex::Real nx = x_center_odch-odch_point0_normal[0];
          amrex::Real normal = sqrt(nx*nx+ny*ny+nz*nz);
          nx=nx/normal;
          ny=ny/normal;
          nz=nz/normal;
          if(rad<=radius_outer_dome_cooling_hole)
            {
              //amrex::Print()<<"\n Yes";

              s_ext[VELX] = vel_mag_odch*nx;           //Dummy value. change it before prod run
              s_ext[VELY] = vel_mag_odch*ny;
              s_ext[VELZ] = vel_mag_odch*nz;
            }
        }
    }

  //Model inner dome cooling holes
    if(radius<=rmax_inner_dch and radius>=rmin_inner_dch)
      {
        for(amrex::Real theta=delta_theta/2.0;theta<360;theta=theta+delta_theta)
          {
            amrex::Real theta_in_radians=theta*deg_to_radians;
            amrex::Real y_center_idch=radius_idchs*sin(theta_in_radians);
            amrex::Real x_center_idch=-radius_idchs*cos(theta_in_radians);
            amrex::Real rad=sqrt(std::pow((x[1]-y_center_idch),2.0)+pow((x[0]-x_center_idch),2.0));
            amrex::Real nz = x[2]-idch_point0_normal[2];
            amrex::Real ny = y_center_idch-idch_point0_normal[1];
            amrex::Real nx = x_center_idch-idch_point0_normal[0];
            amrex::Real normal = sqrt(nx*nx+ny*ny+nz*nz);
            nx=nx/normal;
            ny=ny/normal;
            nz=nz/normal;
            if(rad<=radius_inner_dome_cooling_hole)
              {
                s_ext[VELX] = vel_mag_idch*nx;           //Dummy value. change it before prod run
                s_ext[VELY] = vel_mag_idch*ny;
                s_ext[VELZ] = vel_mag_odch*nz;
              }
          }
      }



}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc (int i, int j, int k,
           amrex::Array4<amrex::Real> const& beta,
           amrex::GeometryData const& geomdata,
           amrex::Box const& domainBox,
           const int  dir,
           const int  beta_comp,
           const int  nComp)
{
    amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
    // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
    // otherwise this routine could be called for other face diffusivity (Temp, velocity, ...)

    // Zero only on the lowZ for species.
    if ( (beta_comp == 0) && (nComp == NUM_SPECIES) ) {
      if (dir == 2 && k <= domainBox.smallEnd(dir)) {
        const amrex::Real* prob_lo = geomdata.ProbLo();
        const amrex::Real* dx      = geomdata.CellSize();
        const amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
        const amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];

        // Walls everywhere be default
        int wall_tag = 1;

        // Not on the premixers:
        amrex::GpuArray<amrex::Real,2*4> prem_centers{-0.0163322, 0.0163322,
                                                       0.0163322, 0.0163322,
                                                      -0.0163322,-0.0163322,
                                                       0.0163322,-0.0163322};
        for (int pm = 0; pm < 4; ++pm) {
          amrex::Real pm_rad = sqrt(  (x-prem_centers[2*pm])  *(x-prem_centers[2*pm])
                                    + (y-prem_centers[2*pm+1])*(y-prem_centers[2*pm+1]));
          if (pm_rad < 0.0083312 && pm_rad >= 0.0057404) {
            wall_tag = 0;
          }
        }

        // Not on the pilot
        amrex::Real pilot_rad = sqrt(x*x+y*y);
        if (pilot_rad <= 0.00508) {
          wall_tag = 0;
        }

        if (wall_tag == 1) {
          for (int n = beta_comp; n < beta_comp + nComp; n++) {
             beta(i,j,k,n) = 0.0;
          }
        }
      }
    }
}
#endif
