#ifndef _EBUSERDEFINED_H_
#define _EBUSERDEFINED_H_

using namespace amrex;

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

void
EBUserDefined(const Geometry& geom,
              const int required_coarsening_level,
              const int max_coarsening_level)
{
    // ParmParse your geometry parameters
    // --> fixed geometry of the challenge problem
    //Geometric parameters. Default values

    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo = geom.ProbLoArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> phi = geom.ProbHiArray();

    amrex::ParmParse pp("eb2");
    std::string geomversion{"Simple"};
    pp.query("geom_version",geomversion);

    const amrex::Real mm_to_m=0.001;
    const amrex::Real deg_to_rad=atan(1.0)*4.0/180.0;
    const amrex::Real L_combustor=470*mm_to_m;

    if (geomversion == "Simple") {

        amrex::Real d_dump                      = 134.493*mm_to_m;
        amrex::Real d_corner_premixer           = 16.6624*mm_to_m;
        amrex::Real r_corner_premixer_inner_0   = 4.5323*mm_to_m;
        amrex::Real r_corner_premixer_inner_1   = 5.7404*mm_to_m;

        amrex::Real d_central_pilot             = 5.08*2*mm_to_m;
        amrex::Real a_dump                      = 134.493*mm_to_m;

        amrex::Real d_bafflecoolinghole         = 2.5875*mm_to_m;

        amrex::Real BCH_x     = 787.1325*mm_to_m;
        amrex::Real BCH_y     = 0.0*mm_to_m;
        amrex::Real BCH_z     = 0.0*mm_to_m;

        amrex::Real d_dome1   = 121.92*mm_to_m;
        amrex::Real d_dome2   = 111.22025*mm_to_m;

        int axis_direction=0;

        amrex::Real center_dump_x = 1154.5065*mm_to_m;
        amrex::Real center_cp_x   = 784.079821*mm_to_m;
        amrex::Real center_cp_y   = 16.3322*mm_to_m;
        amrex::Real center_cp_z   = 16.3322*mm_to_m;

        amrex::Real y_bch = 0.0662178;
        amrex::Real z_bch = 0.0662178;
        amrex::Real a_bch = 0.062875;
        amrex::Real bch_thickness = 0.001;

        amrex::Real nx=0.0;
        amrex::Real ny=1.0;

        amrex::Real tol = 0.0;

        Array<Real,3> transitionbox_top_point     ={0.9363575, d_dump/2.0,0.0};
        Array<Real,3> transitionbox_bot_point     ={0.9363575,-d_dump/2.0,0.0};
        Array<Real,3> transitionbox_frnt_point    ={0.9363575,0.0, d_dump/2.0};
        Array<Real,3> transitionbox_back_point    ={0.9363575,0.0,-d_dump/2.0};

        Array<Real,3> transitionbox_top_normal      ={cos(75.0*deg_to_rad), sin(75.0*deg_to_rad),0.0};
        Array<Real,3> transitionbox_bot_normal      ={cos(75.0*deg_to_rad), -sin(75.0*deg_to_rad),0.0};
        Array<Real,3> transitionbox_frnt_normal      ={cos(75.0*deg_to_rad), 0.0,sin(75.0*deg_to_rad)};
        Array<Real,3> transitionbox_back_normal      ={cos(75.0*deg_to_rad), 0.0,-sin(75.0*deg_to_rad)};

        //Dump region
        amrex::EB2::CylinderIF Dump(0.5*d_dump,L_combustor, axis_direction, {AMREX_D_DECL(center_dump_x,0,0)}, true);
        amrex::EB2::BoxIF Dump_Box({AMREX_D_DECL(0.78713250,-0.0672465,-0.0672465)},{AMREX_D_DECL(0.9363575, 0.0672465, 0.0672465)},true);

        //Creating the transition section from block to cylinder
        amrex::EB2::PlaneIF transitionbox_top_plane(transitionbox_top_point,transitionbox_top_normal,true);
        amrex::EB2::PlaneIF transitionbox_bot_plane(transitionbox_bot_point,transitionbox_bot_normal,true);
        amrex::EB2::PlaneIF transitionbox_frnt_plane(transitionbox_frnt_point,transitionbox_frnt_normal,true);
        amrex::EB2::PlaneIF transitionbox_back_plane(transitionbox_back_point,transitionbox_back_normal,true);
        amrex::EB2::PlaneIF truncate_plane_xbeg(transitionbox_back_point,{AMREX_D_DECL(-1.0, 0.0,0.0)},true);
        amrex::EB2::PlaneIF truncate_plane_xend({AMREX_D_DECL(1.0163575, 0.0,0.0)},{AMREX_D_DECL(1.0, 0.0,0.0)},true);
        auto transition_box= amrex::EB2::makeUnion(transitionbox_top_plane,
                                                   transitionbox_bot_plane,
                                                   transitionbox_frnt_plane,
                                                   transitionbox_back_plane,
                                                   truncate_plane_xbeg,
                                                   truncate_plane_xend);

        auto polys_final=amrex::EB2::makeIntersection(Dump,Dump_Box,transition_box);
        auto gshop = amrex::EB2::makeShop(polys_final);
        amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);

    } else if (geomversion == "Complex") {
        amrex::Real d_dump                    = 134.493*mm_to_m;
        amrex::Real d_corner_premixer         = 16.6624*mm_to_m;
        amrex::Real L_corner_premixer         = 25*mm_to_m;
        amrex::Real r_corner_premixer_inner_0 = 4.5323*mm_to_m;
        amrex::Real r_corner_premixer_inner_1 = 5.7404*mm_to_m;
        amrex::Real l_corner_premixer_inner   = 11.5687*mm_to_m;

        amrex::Real d_central_pilot           = 5.08*2*mm_to_m;
        amrex::Real a_dump                    = 134.493*mm_to_m;

        amrex::Real d_bafflecoolinghole       = 2.5875*mm_to_m;
        amrex::Real L_bafflecoolinghole       = 100*mm_to_m;
        amrex::Real BCH_x   = 787.1325*mm_to_m;
        amrex::Real BCH_y   = 0.0*mm_to_m;
        amrex::Real BCH_z   = 0.0*mm_to_m;

        amrex::Real d_dome1 = 121.92*mm_to_m;
        amrex::Real d_dome2 = 111.22025*mm_to_m;
        amrex::Real L_dome1 = 0.0302*2*mm_to_m;
        amrex::Real L_dome2 = 1.016*2*mm_to_m;

        int axis_direction=0;

        Array<Real,3> center_domain={0.5*(plo[0]+phi[0]),0.5*(plo[1]+phi[1]),0.5*(plo[2]+phi[2])};

        Array<amrex::Real,3> cp_inner_point ={plo[0]+r_corner_premixer_inner_0,0.0,0.0};
        Array<amrex::Real,3> cp_inner_point_cyl ={plo[0]+d_corner_premixer/2.0,0.0,0.0};
        Array<amrex::Real,3> cp_inner_normal={-l_corner_premixer_inner,-(r_corner_premixer_inner_0-r_corner_premixer_inner_1),0.0};

        amrex::Real center_dump_x = 1154.5065*mm_to_m;
        amrex::Real center_cp_x   = 784.079821*mm_to_m;
        amrex::Real center_cp_y   = 16.3322*mm_to_m;
        amrex::Real center_cp_z   = 16.3322*mm_to_m;

        amrex::Real y_bch = 0.0662178;
        amrex::Real z_bch = 0.0662178;
        amrex::Real a_bch = 0.062875;
        amrex::Real bch_thickness = 0.001;

        amrex::Real nx=0.0;
        amrex::Real ny=1.0;

        amrex::Real tol = 0.0;

        //Dump region
        amrex::EB2::CylinderIF Dump(0.5*d_dump,L_combustor, axis_direction, {AMREX_D_DECL(center_dump_x,0,0)}, true);
        amrex::EB2::BoxIF Dump_Box({AMREX_D_DECL(0.78713250,-0.0672465,-0.0672465)},{AMREX_D_DECL(0.93525715, 0.0672465, 0.0672465)},true);

        //Corner premixers
        amrex::EB2::CylinderIF CentralPilot(0.5*d_central_pilot,L_corner_premixer, axis_direction, {AMREX_D_DECL(center_cp_x,0.0,0.0)}, true);

        amrex::EB2::PlaneIF cp_inner_plane_inclined(cp_inner_point,cp_inner_normal,true);
        amrex::EB2::PlaneIF cp_inner_cyl(cp_inner_point_cyl,{AMREX_D_DECL(1.0,0,0)},true);
        amrex::EB2::PlaneIF cp_inner_plane_top({AMREX_D_DECL(r_corner_premixer_inner_0,l_corner_premixer_inner,0)},{AMREX_D_DECL(0.0,-1.0,0.0)},true);
        amrex::EB2::PlaneIF cp_inner_plane_top1({AMREX_D_DECL(r_corner_premixer_inner_0,L_corner_premixer+tol,0)},{AMREX_D_DECL(0.0,1.0,0.0)},true);
        amrex::EB2::PlaneIF cp_inner_plane_bot({AMREX_D_DECL(r_corner_premixer_inner_0,-0.001,0)},{AMREX_D_DECL(0.0,-1.0,0.0)},true);

        auto cp_plane1=amrex::EB2::makeUnion(cp_inner_cyl,cp_inner_plane_top1);//,cp_inner_plane_top,cp_inner_plane_top1,cp_inner_plane_bot);
        auto cp_plane2=amrex::EB2::makeIntersection(cp_inner_plane_inclined,cp_inner_plane_top);//,cp_inner_plane_top,cp_inner_plane_top1,cp_inner_plane_bot);
        auto cp_plane3=amrex::EB2::makeUnion(cp_plane1,cp_plane2);
        auto cp_plane4=amrex::EB2::makeUnion(cp_plane3,cp_inner_plane_bot);
        auto cp_plane_translated=amrex::EB2::translate(cp_plane3,{AMREX_D_DECL(-plo[0],0,0)});
        auto cp_inner = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(cp_plane_translated), 90 * M_PI / 180, 1),{AMREX_D_DECL(plo[0],0,0)});
        auto CornerPremixer_0 = amrex::EB2::translate(cp_inner,{AMREX_D_DECL(0.0,center_cp_y,center_cp_z)});
        auto CornerPremixer_1 = amrex::EB2::translate(cp_inner,{AMREX_D_DECL(0,-center_cp_y,center_cp_z)});
        auto CornerPremixer_2 = amrex::EB2::translate(cp_inner,{AMREX_D_DECL(0,center_cp_y,-center_cp_z)});
        auto CornerPremixer_3 = amrex::EB2::translate(cp_inner,{AMREX_D_DECL(0,-center_cp_y,-center_cp_z)});

        amrex::EB2::CylinderIF dome1(0.5*d_dome1,L_dome1, axis_direction, {AMREX_D_DECL(0.7871325,0,0)}, true);
        amrex::EB2::CylinderIF dome2(0.5*d_dome2,L_dome2, axis_direction, {AMREX_D_DECL(0.7871325,0,0)}, true);

        //Baffle cooling holes
        amrex::EB2::CylinderIF BaffleCoolingHole(0.5*d_bafflecoolinghole,L_dome1, axis_direction, {AMREX_D_DECL(BCH_x,BCH_y,BCH_z)}, true);
        amrex::EB2::BoxIF Bafflecooling_Box1({AMREX_D_DECL(0.75,y_bch-bch_thickness,-a_bch)},{AMREX_D_DECL(0.79,y_bch+bch_thickness,a_bch)},true);
        amrex::EB2::BoxIF Bafflecooling_Box2({AMREX_D_DECL(0.75,-y_bch-bch_thickness,-a_bch)},{AMREX_D_DECL(0.79,-y_bch+bch_thickness,a_bch)},true);
        amrex::EB2::BoxIF Bafflecooling_Box3({AMREX_D_DECL(0.75,-a_bch,z_bch-bch_thickness)},{AMREX_D_DECL(0.79,a_bch,z_bch+bch_thickness)},true);
        amrex::EB2::BoxIF Bafflecooling_Box4({AMREX_D_DECL(0.75,-a_bch,-z_bch-bch_thickness)},{AMREX_D_DECL(0.79,a_bch,-z_bch+bch_thickness)},true);

        // Oblique cooling holes
        amrex::EB2::PlaneIF nozzle_plane_top({AMREX_D_DECL(0.787462, 0.065, 0)},{AMREX_D_DECL(nx,ny,0.0)},true);
        amrex::EB2::PlaneIF nozzle_plane_bot({AMREX_D_DECL(0.787462, 0.055, 0)},{AMREX_D_DECL(nx,-ny,0.0)},true);
        auto pol=amrex::EB2::makeUnion(nozzle_plane_top,nozzle_plane_bot);
        amrex::EB2::PlaneIF nozzle_plane_bot1({AMREX_D_DECL(0.79, 0.0, 0)},{AMREX_D_DECL(1,0.0,0.0)},true);
        auto pol1=amrex::EB2::makeUnion(pol,nozzle_plane_bot1);
        auto nozzle_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(pol), 90 * M_PI / 180, 1),{AMREX_D_DECL(0.78,0, 0)});

        auto polys = amrex::EB2::makeIntersection(Dump,Dump_Box);//,CornerPremixer_2,CornerPremixer_3,CentralPilot);
        auto dome = amrex::EB2::makeIntersection(dome1,dome2);
        auto polys_comb_complement=amrex::EB2::makeUnion(polys,amrex::EB2::makeComplement(dome));

        auto polys_final=amrex::EB2::makeIntersection(Bafflecooling_Box1,
                                                      Bafflecooling_Box2,
                                                      Bafflecooling_Box3,
                                                      Bafflecooling_Box4,
                                                      polys_comb_complement,
                                                      CornerPremixer_0,
                                                      CornerPremixer_1,
                                                      CornerPremixer_2,
                                                      CornerPremixer_3,
                                                      CentralPilot);
        auto gshop = amrex::EB2::makeShop(polys_final);
        amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
    } else {
        amrex::Abort("Unknown geom_version ! Can only be 'Simple' or 'Complex'");
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
setEBState(const amrex::Real * /*xEBface[AMREX_SPACEDIM]*/,
           amrex::Real * /*s_ext[NVAR]*/,
           const amrex::Real /*time*/,
           amrex::GeometryData const& /*geomdata*/,
           ProbParm const& /*prob_parm*/)
{
  // Provide the state on the EB centroid provided
  // by xEBface
  // Not the entire has to be specified. For instance, if using isothermal
  // boundary, only s_ext[TEMP] will be used.
  // Aborting here (message will not show on GPUs)
  Abort("Default setEBState called. Copy and edit this file for your needs");
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
setEBType(const amrex::Real * /*xEBface[AMREX_SPACEDIM]*/,
          amrex::Real & /*EBflagType*/,
          amrex::GeometryData const& /*geomdata*/,
          ProbParm const& /*prob_parm*/)
{
  // Provide a factor applied on the EB face diffusion
  // coefficient: 0.0 will turn off the diffusion (homogeneous Neumann BC)
  // while 1.0 will let it unchanged. Intermediate values can be specified
  // for partially Dirichlet BCs
  Abort("Default setEBType called. Copy and edit this file for your needs");
}
#endif
#endif
