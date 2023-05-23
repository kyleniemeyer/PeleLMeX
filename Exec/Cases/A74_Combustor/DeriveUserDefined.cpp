#include <PeleLMDeriveFunc.H>
#include <PeleLM_Index.H>
#include <PelePhysics.H>
#include <mechanism.H>
#include <PeleLM.H>
#include <PeleLM_K.H>

using namespace amrex;

//
// User-defined derived variables list
//
Vector<std::string> pelelm_setuserderives()
{
  //Vector<std::string> var_names({"derUserDefine_null"});
  return {"tagCoolingJets"}; //var_names;
}

//
// User-defined derived definition
//
void pelelm_deruserdef (PeleLM* /*a_pelelm*/, const Box& bx, FArrayBox& derfab, int dcomp, int /*ncomp*/,
                        const FArrayBox& statefab, const FArrayBox& /*reactfab*/, const FArrayBox& /*pressfab*/,
                        const Geometry& geom, Real /*time*/, const Vector<BCRec>& /*bcrec*/, int /*level*/)
{
    AMREX_D_TERM(const amrex::Real dx = geom.CellSize(0);,
                 const amrex::Real dy = geom.CellSize(1);,
                 const amrex::Real dz = geom.CellSize(2););

    auto const& uder_arr = derfab.array(dcomp);
    const auto geomdata = geom.data();

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        const amrex::Real* prob_lo = geomdata.ProbLo();
        const amrex::Real* prob_hi = geomdata.ProbHi();
        AMREX_D_TERM(amrex::Real x = prob_lo[0] + (i+0.5)*dx;,
                     amrex::Real y = prob_lo[1] + (j+0.5)*dy;,
                     amrex::Real z = prob_lo[2] + (k+0.5)*dz;);
        amrex::Real yc = prob_lo[1] + 0.5 * (prob_hi[1] - prob_lo[1]);
        amrex::Real zc = prob_lo[2] + 0.5 * (prob_hi[2] - prob_lo[2]);
        amrex::Real rad = std::sqrt((y-yc)*(y-yc)+(z-zc)*(z-zc));
        if (rad > 0.0625 && rad < 0.0655 && x < (prob_lo[0]+0.002) ) {
          uder_arr(i,j,k,0) = 1.0;
        } else if (rad > 0.057 && rad < 0.060 && x < (prob_lo[0]+0.002) ) {
          uder_arr(i,j,k,0) = 1.0;
        } else if ( (std::abs(y-yc) > 0.065 || std::abs(z-zc) > 0.065) && x < (prob_lo[0]+0.002) ) {
          uder_arr(i,j,k,0) = 1.0;
        } else {
          uder_arr(i,j,k,0) = 0.0;
        }
    });
}
