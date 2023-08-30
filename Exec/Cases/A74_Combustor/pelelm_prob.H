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

// Some functions to compute the overlap area between a square and circle
// more or less copied from: https://stackoverflow.com/questions/622287/area-of-intersection-between-circle-and-rectangle

// helper function
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
section_area(const amrex::Real x0,
             const amrex::Real x1,
             const amrex::Real h,
             const amrex::Real r) {
  const amrex::Real sec = (h < r) ? std::sqrt(r*r - h*h) : 0;
  const amrex::Real arg1 = amrex::max(-sec, amrex::min(sec, x1) );
  const amrex::Real arg2 = amrex::max(-sec, amrex::min(sec, x0) );
  const amrex::Real term1 = 0.5* (std::sqrt(1.0 - arg1*arg1/(r*r)) * arg1*r + r*r*std::asin(arg1/r) - 2.0 * h * arg1);
  const amrex::Real term2 = 0.5* (std::sqrt(1.0 - arg2*arg2/(r*r)) * arg2*r + r*r*std::asin(arg2/r) - 2.0 * h * arg2);
  return term1 - term2;
}

// Area of a square and circle intersection
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
square_circle_intersection_area(
                                const amrex::Real x0,
                                const amrex::Real x1,
                                const amrex::Real y0,
                                const amrex::Real y1,
                                const amrex::Real cx,
                                const amrex::Real cy,
                                const amrex::Real r
)
{
  // We assume coordinates of the box are passed in sorted order x0 = xlo, x1 = xhi, etc
  AMREX_ASSERT(x1 > x0);
  AMREX_ASSERT(y1 > y0);
  AMREX_ASSERT(r >= 0.0);

  // Convert to coordinates centered on circle
  const amrex::Real x0new = x0 - cx;
  const amrex::Real x1new = x1 - cx;
  const amrex::Real y0new = y0 - cy;
  const amrex::Real y1new = y1 - cy;

  // Compute the area
  const amrex::Real sign = copysign(1.0,y0new) * copysign(1.0,y1new);
  const amrex::Real y0mod = amrex::min(std::abs(y0new), std::abs(y1new));
  const amrex::Real y1mod = amrex::max(std::abs(y0new), std::abs(y1new));
  const amrex::Real area0 = section_area(x0new, x1new, 0.0, r);
  const amrex::Real area1 = area0 - section_area(x0new, x1new, y0mod, r);
  const amrex::Real area2 = area0 - section_area(x0new, x1new, y1mod, r);

  return area2 - sign*area1;
}


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
amrex::Real
CalculateMixtureFraction(amrex::Real non_dim_r)
{
  amrex::Real mixture_fraction=0.0;
  amrex::Real fuel_mass_fraction=0.0;
  if(non_dim_r>=0.9895)
    {
      mixture_fraction = -0.3926*non_dim_r+0.3926;
    }
  else if(non_dim_r>=0.8289)
    {
      mixture_fraction = 169.605257*pow(non_dim_r,5.0)-744.5464031*pow(non_dim_r,4.0)+1306.488623*pow(non_dim_r,3.0)-1145.458256*pow(non_dim_r,2.0)+501.7677282*non_dim_r-87.8513965;
    }
  else if(non_dim_r>=0.4158)
    {
      mixture_fraction =-0.00035592*pow(non_dim_r,3.0)+0.00258535*pow(non_dim_r,2.0)-0.00431028*non_dim_r+0.0037782;
    }
  else if(non_dim_r>=0.0132)
    {
      mixture_fraction =-3.52950391*pow(non_dim_r,5.0)+4.55039337*pow(non_dim_r,4.0)-2.28201112*pow(non_dim_r,3.0)+0.56780962*pow(non_dim_r,2.0)-0.07716233*non_dim_r+0.00814645;
    }
  else
    {
      mixture_fraction = 0.5479*non_dim_r;
    }

  return(mixture_fraction);
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

  // Default chamber gas composition
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

    //Diffusion flame properties
  constexpr amrex::Real Y_Fuel_0=1.0;
  constexpr amrex::Real Y_O2_0=0.233;
  constexpr amrex::Real Y_N2_0=0.767;

  massfrac[O2_ID] = 0.233;
  massfrac[N2_ID] = 0.767;

  // Set up premixer composition
  /*This was assuming uniform mixture fraction at inlets. But we have fmix profiles from Fluent simulations.
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
  }*/

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

  //Radial coordinate of the point on boundary. The inlets lie on Y-Z plane
  amrex::Real x_c = x[0];
  amrex::Real y_c = x[1];
  amrex::Real z_c = geomdata.ProbLo(2); // assumes zlo face, we're actually on the face not cell center
  amrex::Real radius=sqrt(std::pow((x[0]),2.0)+pow((x[1]),2.0));

  bool use_strip_method_for_bafflecoolingholes=false;

  //Top baffle cooling holes
  constexpr amrex::Real ymax_bch_top = (66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real ymin_bch_top = (66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real xmin_bch_top = -62.865*0.001;
  constexpr amrex::Real xmax_bch_top =  62.865*0.001;

  //Bottom baffle cooling holes
  constexpr amrex::Real ymax_bch_bot = (-66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real ymin_bch_bot = (-66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real xmin_bch_bot = -62.865*0.001;
  constexpr amrex::Real xmax_bch_bot =  62.865*0.001;

  //front baffle cooling holes
  constexpr amrex::Real xmax_bch_frt = (66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real xmin_bch_frt = (66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real ymin_bch_frt = -62.865*0.001;
  constexpr amrex::Real ymax_bch_frt =  62.865*0.001;

  //back baffle cooling holes
  constexpr amrex::Real xmax_bch_bak = (-66.2178-0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real xmin_bch_bak = (-66.2178+0.79375)*0.001;   //bch->baffle cooling holes
  constexpr amrex::Real ymin_bch_bak = -62.865*0.001;
  constexpr amrex::Real ymax_bch_bak =  62.865*0.001;

  const int num_of_baffle_cooling_holes=34;
  constexpr amrex::Real radius_baffle_cooling_holes=0.79375*0.001;
  constexpr amrex::Real dist_between_bch=3.81*0.001;

  constexpr amrex::Real zvel_bch=100.0;

  //Outer dome cooling holes
  constexpr amrex::Real radius_outer_dome_cooling_hole=0.4445*0.001;
  constexpr amrex::Real radius_odchs=64.18579658*0.001;
  constexpr amrex::Real rmax_outer_dch=radius_odchs+radius_outer_dome_cooling_hole;
  constexpr amrex::Real rmin_outer_dch=radius_odchs-radius_outer_dome_cooling_hole;
  constexpr int num_of_outer_dch=52;
  constexpr amrex::Real delta_theta=360.0/52.0;
  constexpr amrex::Array<amrex::Real,3> odch_point0_normal     ={0.0,0.0,0.7244707034};

  //Inner dome cooling holes
  constexpr amrex::Real radius_inner_dome_cooling_hole=0.8255*0.001;
  constexpr amrex::Real radius_idchs=58.42*0.001;
  constexpr amrex::Real rmax_inner_dch=radius_idchs+radius_inner_dome_cooling_hole;
  constexpr amrex::Real rmin_inner_dch=radius_idchs-radius_inner_dome_cooling_hole;
  constexpr int num_of_inner_dch=52;
  constexpr amrex::Array<amrex::Real,3> idch_point0_normal     ={0.0,0.0,0.7095693667};

  constexpr amrex::Real lbm_to_kg=0.453592;
  constexpr amrex::Real mfr_idch=0.02416*lbm_to_kg;
  constexpr amrex::Real mfr_odch=0.06464*lbm_to_kg;
  constexpr amrex::Real mfr_bch=0.09*2*lbm_to_kg;

  constexpr amrex::Real area_per_odch=Pi*(radius_outer_dome_cooling_hole)*(radius_outer_dome_cooling_hole);
  constexpr amrex::Real area_per_idch=Pi*(radius_inner_dome_cooling_hole)*(radius_inner_dome_cooling_hole);
  constexpr amrex::Real area_per_bch =Pi*(radius_baffle_cooling_holes)*(radius_baffle_cooling_holes);

  const amrex::Real vel_mag_odch=mfr_odch/(area_per_odch*s_ext[DENSITY]*num_of_outer_dch);
  const amrex::Real vel_mag_idch=mfr_idch/(area_per_idch*s_ext[DENSITY]*num_of_inner_dch);
  const amrex::Real vel_mag_bch=mfr_bch/(area_per_bch*s_ext[DENSITY]*num_of_baffle_cooling_holes*4);

  //amrex::Real vel_mag_odch=0.0;
  //amrex::Real vel_mag_idch=0.0;
  //amrex::Real vel_mag_bch=0.0;
  constexpr amrex::Real rmax_pilot = 0.005080;
  constexpr amrex::Real rmin_pilot = 0.0;

  constexpr amrex::Real x_cp_00 = -0.0163322;
  constexpr amrex::Real y_cp_00 = 0.0163322;

  constexpr amrex::Real x_cp_01 = 0.0163322;
  constexpr amrex::Real y_cp_01 = 0.0163322;

  constexpr amrex::Real x_cp_02 = -0.0163322;
  constexpr amrex::Real y_cp_02 = -0.0163322;

  constexpr amrex::Real x_cp_03 = 0.0163322;
  constexpr amrex::Real y_cp_03 = -0.0163322;

  constexpr amrex::Real rmax_cp = 0.0083312;
  constexpr amrex::Real rmin_cp = 0.0057404;

  const amrex::Real radius_cp00 = sqrt(std::pow((x[0]-x_cp_00),2.0)+pow((x[1]-y_cp_00),2.0));
  const amrex::Real radius_cp01 = sqrt(std::pow((x[0]-x_cp_01),2.0)+pow((x[1]-y_cp_01),2.0));
  const amrex::Real radius_cp02 = sqrt(std::pow((x[0]-x_cp_02),2.0)+pow((x[1]-y_cp_02),2.0));
  const amrex::Real radius_cp03 = sqrt(std::pow((x[0]-x_cp_03),2.0)+pow((x[1]-y_cp_03),2.0));

  const amrex::Real* dx      = geomdata.CellSize();
  const amrex::Real diag_dist = 0.5*sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
  const amrex::Real face_xlo = x[0] - 0.5*dx[0];
  const amrex::Real face_xhi = x[0] + 0.5*dx[0];
  const amrex::Real face_ylo = x[1] - 0.5*dx[1];
  const amrex::Real face_yhi = x[1] + 0.5*dx[1];
  const amrex::Real inv_area = 1.0 / (dx[0] * dx[1]);
  int in_inlet = 0;

  //Use a strip for modeling baffle cooling holes
  if(use_strip_method_for_bafflecoolingholes)
    {
      //top
      if( (y_c>=ymin_bch_top and y_c<=ymax_bch_top and x_c>=xmin_bch_top and x_c<=xmax_bch_top)
          || (y_c>=ymax_bch_bot and y_c<=ymin_bch_bot and x_c>=xmin_bch_bot and x_c<=xmax_bch_bot)
          || (y_c>=ymin_bch_frt and y_c<=ymax_bch_frt and x_c>=xmin_bch_frt and x_c<=xmax_bch_frt)
          || (y_c<=ymax_bch_bak and y_c>=ymin_bch_bak and x_c>=xmax_bch_bak and x_c<=xmin_bch_bak) )
        {
          s_ext[VELX] = 0.0;           //Dummy value. change it before prod run
          s_ext[VELY] = 0.0;
          s_ext[VELZ] = zvel_bch;
          in_inlet += 1;
        }
    }
  else  //Use circular patches as inlets for baffle cooling holes. only air
    {
      if(y_c>=ymin_bch_top-diag_dist and y_c<=ymax_bch_top+diag_dist and x_c>=xmin_bch_top-diag_dist-radius_baffle_cooling_holes and x_c<=xmax_bch_top+diag_dist+radius_baffle_cooling_holes)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              const amrex::Real y_center_hole=0.5*(ymin_bch_top+ymax_bch_top);
              const amrex::Real x_center_hole=xmin_bch_top+i*dist_between_bch;
              const amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes-diag_dist) {
                // face is fully in inlet
                s_ext[VELX] = 0.0;
                s_ext[VELY] = 0.0;
                s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch;
                in_inlet += 1;
              } else if (rad<=radius_baffle_cooling_holes+diag_dist) {
                // face may be partially covered by inlet
                const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_hole, y_center_hole, radius_baffle_cooling_holes);
                if (area > 0.0) {
                  const amrex::Real area_frac = area * inv_area;
                  s_ext[VELX] = 0.0;
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch * area_frac;
                  in_inlet += 1;
                }
              }
            }
        }
      else if(y_c>=ymax_bch_bot-diag_dist and y_c<=ymin_bch_bot+diag_dist and x_c>=xmin_bch_bot-radius_baffle_cooling_holes-diag_dist and x_c<=xmax_bch_bot+diag_dist+radius_baffle_cooling_holes)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              const amrex::Real y_center_hole=0.5*(ymin_bch_bot+ymax_bch_bot);
              const amrex::Real x_center_hole=xmin_bch_bot+i*dist_between_bch;
              const amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes-diag_dist) {
                // face is fully in inlet
                s_ext[VELX] = 0.0;
                s_ext[VELY] = 0.0;
                s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch;
                in_inlet += 1;
              } else if (rad<=radius_baffle_cooling_holes+diag_dist) {
                // face may be partially covered by inlet
                const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_hole, y_center_hole, radius_baffle_cooling_holes);
                if (area > 0.0) {
                  const amrex::Real area_frac = area * inv_area;
                  s_ext[VELX] = 0.0;
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch * area_frac;
                  in_inlet += 1;
                }
              }
            }
        } //front
      else if(y_c>=ymin_bch_frt-radius_baffle_cooling_holes-diag_dist and y_c<=ymax_bch_frt+radius_baffle_cooling_holes+diag_dist and x_c>=xmin_bch_frt-diag_dist and x_c<=xmax_bch_frt+diag_dist)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              const amrex::Real y_center_hole=ymin_bch_frt+i*dist_between_bch;
              const amrex::Real x_center_hole=0.5*(xmin_bch_frt+xmax_bch_frt);
              const amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes-diag_dist) {
                // face is fully in inlet
                s_ext[VELX] = 0.0;
                s_ext[VELY] = 0.0;
                s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch;
                in_inlet += 1;
              } else if (rad<=radius_baffle_cooling_holes+diag_dist) {
                // face may be partially covered by inlet
                const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_hole, y_center_hole, radius_baffle_cooling_holes);
                if (area > 0.0) {
                  const amrex::Real area_frac = area * inv_area;
                  s_ext[VELX] = 0.0;
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch * area_frac;
                  in_inlet += 1;
                }
              }
            }
        }
      else if (y_c<=ymax_bch_bak+radius_baffle_cooling_holes+diag_dist and y_c>=ymin_bch_bak-radius_baffle_cooling_holes-diag_dist and x_c>=xmax_bch_bak-diag_dist and x_c<=xmin_bch_bak+diag_dist)
        {
          for(int i=0;i<num_of_baffle_cooling_holes;i++)
            {
              const amrex::Real y_center_hole=ymin_bch_bak+i*dist_between_bch;
              const amrex::Real x_center_hole=0.5*(xmin_bch_bak+xmax_bch_bak);
              const amrex::Real rad=sqrt(std::pow((x[1]-y_center_hole),2.0)+pow((x[0]-x_center_hole),2.0));
              if(rad<=radius_baffle_cooling_holes-diag_dist) {
                // face is fully in inlet
                s_ext[VELX] = 0.0;
                s_ext[VELY] = 0.0;
                s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch;
                in_inlet += 1;
              } else if (rad<=radius_baffle_cooling_holes+diag_dist) {
                // face may be partially covered by inlet
                const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_hole, y_center_hole, radius_baffle_cooling_holes);
                if (area > 0.0) {
                  const amrex::Real area_frac = area * inv_area;
                  s_ext[VELX] = 0.0;
                  s_ext[VELY] = 0.0;
                  s_ext[VELZ] = vel_mag_bch*prob_parm.mfr_factor_bch * area_frac;
                  in_inlet += 1;
                }
              }
            }
        }

    }

  //Model outer dome cooling holes
  if(radius<=rmax_outer_dch+diag_dist and radius>=rmin_outer_dch-diag_dist)
    {
      for(amrex::Real theta=0.0;theta<360-0.5*delta_theta;theta=theta+delta_theta)
        {
          const amrex::Real theta_in_radians=theta*deg_to_radians;
          const amrex::Real y_center_odch=radius_odchs*sin(theta_in_radians);
          const amrex::Real x_center_odch=-radius_odchs*cos(theta_in_radians);

          const amrex::Real rad=sqrt(std::pow((x[1]-y_center_odch),2.0)+pow((x[0]-x_center_odch),2.0));
          amrex::Real nz = z_c-odch_point0_normal[2];
          amrex::Real ny = y_center_odch-odch_point0_normal[1];
          amrex::Real nx = x_center_odch-odch_point0_normal[0];
          const amrex::Real abs_inv_nz = 1.0/std::abs(nz);
          nx=nx*abs_inv_nz;
          ny=ny*abs_inv_nz;
          nz=nz*abs_inv_nz;
          //amrex::Print()<<"\n"<<theta<<" "<<x_center_odch<<" "<<y_center_odch<<" "<<x[2]<<" "<<nx<<" "<<ny<<" "<<nz;
          if(rad<=radius_outer_dome_cooling_hole-diag_dist) {
              s_ext[VELX] = vel_mag_odch*nx*prob_parm.mfr_factor_odch;
              s_ext[VELY] = vel_mag_odch*ny*prob_parm.mfr_factor_odch;
              s_ext[VELZ] = vel_mag_odch*nz*prob_parm.mfr_factor_odch;
              in_inlet += 1;
          } else if (rad<=radius_outer_dome_cooling_hole+diag_dist) {
            const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_odch, y_center_odch, radius_outer_dome_cooling_hole);
            if (area > 0.0) {
              const amrex::Real area_frac = area * inv_area;
              s_ext[VELX] = vel_mag_odch*nx*prob_parm.mfr_factor_odch*area_frac;
              s_ext[VELY] = vel_mag_odch*ny*prob_parm.mfr_factor_odch*area_frac;
              s_ext[VELZ] = vel_mag_odch*nz*prob_parm.mfr_factor_odch*area_frac;
              in_inlet += 1;
            }
          }
        }
    }

  //Model inner dome cooling holes
  if(radius<=rmax_inner_dch+diag_dist and radius>=rmin_inner_dch-diag_dist)
    {
      for(amrex::Real theta=delta_theta/2.0;theta<360;theta=theta+delta_theta)
        {
          const amrex::Real theta_in_radians=theta*deg_to_radians;
          const amrex::Real y_center_idch=radius_idchs*sin(theta_in_radians);
          const amrex::Real x_center_idch=-radius_idchs*cos(theta_in_radians);
          const amrex::Real rad=sqrt(std::pow((x[1]-y_center_idch),2.0)+pow((x[0]-x_center_idch),2.0));
          amrex::Real nz = z_c-idch_point0_normal[2];
          amrex::Real ny = y_center_idch-idch_point0_normal[1];
          amrex::Real nx = x_center_idch-idch_point0_normal[0];
          const amrex::Real abs_inv_nz = 1.0/std::abs(nz);
          nx=nx*abs_inv_nz;
          ny=ny*abs_inv_nz;
          nz=nz*abs_inv_nz;
          if(rad<=radius_inner_dome_cooling_hole - diag_dist) {
              s_ext[VELX] = vel_mag_idch*nx*prob_parm.mfr_factor_idch;
              s_ext[VELY] = vel_mag_idch*ny*prob_parm.mfr_factor_idch;
              s_ext[VELZ] = vel_mag_idch*nz*prob_parm.mfr_factor_idch;
              in_inlet += 1;
          } else if (rad<=radius_inner_dome_cooling_hole + diag_dist) {
            const amrex::Real area = square_circle_intersection_area(face_xlo, face_xhi, face_ylo, face_yhi, x_center_idch, y_center_idch, radius_inner_dome_cooling_hole);
            if (area > 0.0) {
              const amrex::Real area_frac = area * inv_area;
              s_ext[VELX] = vel_mag_idch*nx*prob_parm.mfr_factor_idch*area_frac;
              s_ext[VELY] = vel_mag_idch*ny*prob_parm.mfr_factor_idch*area_frac;
              s_ext[VELZ] = vel_mag_idch*nz*prob_parm.mfr_factor_idch*area_frac;
              in_inlet += 1;
            }
          }
        }
    }

  // With the turb inflow stuff we can't apply correction
  if(radius<=rmax_pilot and radius>=rmin_pilot) //For the moment, only air
    {
      s_ext[VELX] = s_ext[VELX]*prob_parm.mfr_factor_pilot;
      s_ext[VELY] = s_ext[VELY]*prob_parm.mfr_factor_pilot;
      s_ext[VELZ] = s_ext[VELZ]*prob_parm.mfr_factor_pilot;
      in_inlet += 1;
    }
  if(radius_cp00<=rmax_cp and radius_cp00>=rmin_cp)
    {
      s_ext[VELX] = s_ext[VELX]*prob_parm.mfr_factor_cp00;
      s_ext[VELY] = s_ext[VELY]*prob_parm.mfr_factor_cp00;
      s_ext[VELZ] = s_ext[VELZ]*prob_parm.mfr_factor_cp00;
      in_inlet += 1;
      const amrex::Real non_dim_r=(radius_cp00-rmin_cp)/(rmax_cp-rmin_cp);
      const amrex::Real mixture_fraction =  CalculateMixtureFraction(non_dim_r);

      massfrac[NC12H26_ID] = mixture_fraction*Y_Fuel_0;
      massfrac[O2_ID] = Y_O2_0*(1-mixture_fraction);
      massfrac[N2_ID] = 1.0-massfrac[NC12H26_ID]-massfrac[O2_ID];

      s_ext[TEMP] = prob_parm.T_mean;                           //From user input
      eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;                         // CGS -> MKS conversion
      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];        // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++)
        {
          s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
        }
    }
  if(radius_cp01<=rmax_cp and radius_cp01>=rmin_cp)
    {
      s_ext[VELX] = s_ext[VELX]*prob_parm.mfr_factor_cp01;
      s_ext[VELY] = s_ext[VELY]*prob_parm.mfr_factor_cp01;
      s_ext[VELZ] = s_ext[VELZ]*prob_parm.mfr_factor_cp01;
      in_inlet += 1;
      const amrex::Real non_dim_r=(radius_cp01-rmin_cp)/(rmax_cp-rmin_cp);
      const amrex::Real mixture_fraction =  CalculateMixtureFraction(non_dim_r);

      massfrac[NC12H26_ID] = mixture_fraction*Y_Fuel_0;
      massfrac[O2_ID] = Y_O2_0*(1-mixture_fraction);
      massfrac[N2_ID] = 1.0-massfrac[NC12H26_ID]-massfrac[O2_ID];

      s_ext[TEMP] = prob_parm.T_mean;                           //From user input
      eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;                         // CGS -> MKS conversion
      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];        // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++)
        {
          s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
        }
    }
  if(radius_cp02<=rmax_cp and radius_cp02>=rmin_cp)
    {
      s_ext[VELX] = s_ext[VELX]*prob_parm.mfr_factor_cp02;
      s_ext[VELY] = s_ext[VELY]*prob_parm.mfr_factor_cp02;
      s_ext[VELZ] = s_ext[VELZ]*prob_parm.mfr_factor_cp02;
      in_inlet += 1;
      const amrex::Real non_dim_r=(radius_cp02-rmin_cp)/(rmax_cp-rmin_cp);
      const amrex::Real mixture_fraction = CalculateMixtureFraction(non_dim_r);

      massfrac[NC12H26_ID] = mixture_fraction*Y_Fuel_0;
      massfrac[O2_ID] = Y_O2_0*(1-mixture_fraction);
      massfrac[N2_ID] = 1.0-massfrac[NC12H26_ID]-massfrac[O2_ID];

      s_ext[TEMP] = prob_parm.T_mean;                           //From user input
      eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;                         // CGS -> MKS conversion
      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];        // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++)
        {
          s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
        }
    }
  if(radius_cp03<=rmax_cp and radius_cp03>=rmin_cp)
    {
      s_ext[VELX] = s_ext[VELX]*prob_parm.mfr_factor_cp03;
      s_ext[VELY] = s_ext[VELY]*prob_parm.mfr_factor_cp03;
      s_ext[VELZ] = s_ext[VELZ]*prob_parm.mfr_factor_cp03;
      in_inlet += 1;
      const amrex::Real non_dim_r=(radius_cp03-rmin_cp)/(rmax_cp-rmin_cp);
      const amrex::Real mixture_fraction = CalculateMixtureFraction(non_dim_r);

      massfrac[NC12H26_ID] = mixture_fraction*Y_Fuel_0;
      massfrac[O2_ID] = Y_O2_0*(1-mixture_fraction);
      massfrac[N2_ID] = 1.0-massfrac[NC12H26_ID]-massfrac[O2_ID];

      s_ext[TEMP] = prob_parm.T_mean;                           //From user input
      eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;                         // CGS -> MKS conversion
      eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
      s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];        // CGS -> MKS conversion

      for (int n = 0; n < NUM_SPECIES; n++)
        {
          s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
        }
    }

  // Wall if we didn't catch anything above
  if (in_inlet == 0) {
    s_ext[VELX] = 0.0;
    s_ext[VELY] = 0.0;
    s_ext[VELZ] = 0.0;
  } else if (in_inlet > 1) {
    // We messed up if we caught more than one inlet
    std::string abort_string = "bcnormal: point in more than one inlet.";
    amrex::Abort(abort_string);
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
