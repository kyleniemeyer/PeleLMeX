#ifndef PELELM_PROB_PARM_H_
#define PELELM_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
   bool activate_fuel_injection = false;
   amrex::Real fuel_injection_max_radius=0.5;
   amrex::Real Y_O2_FUEL=0.065963;
   amrex::Real Y_N2_FUEL=0.434037;
   amrex::Real Y_C12H26_FUEL=0.5;

   // Chamber conditions
   amrex::Real P_mean        = 101325.0;
   amrex::Real T_mean        = 300.0;

   // Premixers
   amrex::Real premix_phi = 1.136;

   // Ignition
   int do_ignit = 0;
   amrex::Real ignit_rad = 0.008;
   amrex::Real ignit_lowZ = 0.02;
   amrex::Real ignit_temp = 1400.0;
   
};
#endif
