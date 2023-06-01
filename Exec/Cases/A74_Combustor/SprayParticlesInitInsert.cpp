
#include "SprayParticles.H"
#include "SprayInjection.H"
#include "pelelm_prob.H"

using namespace amrex;

class PressureSwirlJet : public SprayJet
{
public:
  PressureSwirlJet(const std::string jet_name, const Geometry& geom)
  : SprayJet(jet_name, geom)
  {
     // Lets add anything else we need
     std::string ppspray = "spray." + m_jetName;
     amrex::ParmParse ps(ppspray);
     ps.get("l0", m_inj_l0);
     ps.get("vel_rms", m_inj_velrms);
  }

  bool get_new_particle(
    const Real time,
    const Real& phi_radial,
    const Real& cur_radius,
    Real& umag,
    Real& theta_spread,
    Real& phi_swirl,
    Real& dia_part,
    Real& T_part,
    Real* Y_part) override;

  inline const amrex::Real& jet_l0() const { return m_inj_l0; }
  inline const amrex::Real& jet_velrms() const { return m_inj_velrms; }

  amrex::Real m_storedDiam = -1.;

protected:
  amrex::Real m_inj_l0 = 0.0;
  amrex::Real m_inj_velrms = 0.0;
};

bool
PressureSwirlJet::get_new_particle(
    const Real time,
    const Real& phi_radial,
    const Real& cur_radius,
    Real& umag,
    Real& theta_spread,
    Real& phi_swirl,
    Real& dia_part,
    Real& T_part,
    Real* Y_part)
{
  dia_part = m_dropDist->get_dia();
  return true;
}

bool
SprayParticleContainer::injectParticles(Real time,
                                        Real dt,
                                        int nstep,
                                        int lev,
                                        int finest_level,
                                        ProbParm const& prob_parm)
{
  if (lev != 0) {
      return false;
  }

  auto js = static_cast<PressureSwirlJet*>(m_sprayJets[0].get());
  if (!js->jet_active(time) ||
      amrex::ParallelDescriptor::MyProc() != js->Proc() ||
      dt <= 0.0 ) {
    return false;
  }

  constexpr Real Pi_six = M_PI / 6.;

  // Get fuel species physical data
  SprayUnits SPU;
  const SprayData* fdat = m_sprayData;
  amrex::Real rhoL_avg = 0.;
  for (int spf = 0; spf < SPRAY_FUEL_NUM; ++spf) {
    rhoL_avg +=
      js->get_avg_Y(spf) / fdat->rhoL(js->get_avg_T(), spf);
  }
  rhoL_avg = 1. / rhoL_avg;
  const Real num_ppp = fdat->num_ppp;

  // Check if mass must be injected across multiple timesteps
  const amrex::Real avg_dia = js->get_avg_dia();
  const amrex::Real avg_mass = Pi_six * rhoL_avg * std::pow(avg_dia, 3);
  const amrex::Real min_dia = std::cbrt(SPU.min_mass / (Pi_six * rhoL_avg));
  if (avg_dia < min_dia || avg_mass < SPU.min_mass) {
    amrex::Abort(
      "Average droplet size too small, floating point issues expected");
  }

  //#######################################################
  // Based a Sanjose et al. 2011
  //#######################################################

  // Pressure-swirl has a core of air and a swirling fuel around it.
  // Compute the ratio of air area to total injector area
  // Using Rizk & Levebre 1985
  Real theta = js->spread_angle() * M_PI / 180.0;
  Real ratio = ( 1.0 - std::cos(theta)*std::cos(theta) ) / ( 1.0 + std::cos(theta)*std::cos(theta) );
  Real r0 = js->jet_dia()*0.5;
  Real ra = std::sqrt(ratio*r0*r0);
  Real r_mean = 0.5 * ( r0 + ra );

  // Also get extremas angles of the cone
  Real theta_min = std::abs(std::atan(std::tan(theta) * 2.0 * ra / r_mean));
  Real theta_max = std::abs(std::atan(std::tan(theta) * 2.0 * r0 / r_mean));

  // Get axial speed from mass conservation
  Real mass_flow_rate = js->mass_flow_rate();
  Real jet_vel = mass_flow_rate / ( rhoL_avg * M_PI * r0 * r0 * (1.0 - ratio));

  // Host container
  amrex::ParticleLocData pld;
  std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>> host_particles;

  int npart_inj = 0;

  // Check how much mass we need to inject, adding leftover from previous injections if any
  Real injection_mass = mass_flow_rate * dt;
  if (js->m_sumInjMass > 0.0 ) {
      injection_mass += js->m_sumInjMass;
  }

  // Check if we can inject the last particle we draw if there is one in stock
  if ( js->m_storedDiam > 0.0 ) {
      Real part_mass = num_ppp * Pi_six * rhoL_avg * std::pow(js->m_storedDiam, 3);
      if ( injection_mass < part_mass ) { // We can't do it, just add to mass counter and get out
          js->m_sumInjMass += mass_flow_rate * dt;
          return false;
      }
  }

  // Inject mass until we have the desired amount
  amrex::Real total_mass = 0.;
  amrex::Real remaining_mass = injection_mass;
  while (total_mass < injection_mass) {

      // Get a particle size
      Real cur_dia = 0.0;
      if ( js->m_storedDiam > 0.0 ) { // We have a stored particles, use it (already checked that we can use it)
         cur_dia = js->m_storedDiam;
         js->m_storedDiam = -1.0;
      } else {    // Draw a particle
         amrex::Real phi_radial, cur_rad, umag, theta_spread, phi_swirl, dia_part, T_part;
         amrex::Real Y_part[SPRAY_FUEL_NUM];
         js->get_new_particle(
           time, phi_radial, cur_rad, umag, theta_spread, phi_swirl, dia_part,
           T_part, Y_part);
         cur_dia = dia_part;
         if ( cur_dia < min_dia ) {
             continue;
         }
         Real pmass = num_ppp * Pi_six * rhoL_avg * std::pow(cur_dia, 3);
         if ( remaining_mass < pmass ) {  // Don't have enough mass left, store it for later
             js->m_storedDiam = cur_dia;
             js->m_sumInjMass = remaining_mass;
             break;
         }
      }

      // Pick a random radial location and get corresponding half angle
      // See Sanjose et al. for geometrical relation between real injector plane and injection
      // location plane -> provided by l0
      Real inj_p_rad_random = amrex::Random();
      Real inj_p_theta = theta_min + inj_p_rad_random * (theta_max - theta_min);
      Real inj_p_rad = js->jet_l0() * inj_p_theta + ra + inj_p_rad_random * (r0 - ra);
      Real inj_p_alpha = std::asin( (ra + inj_p_rad_random * (r0 - ra)) / inj_p_rad );

      // Get the random position on the azimuth [0:2*pi]
      Real inj_p_angle_random = amrex::Random() * 2.0 * M_PI;
      RealVect part_loc(AMREX_D_DECL(js->jet_cent()[0] + std::cos(inj_p_angle_random) * inj_p_rad,
                                     js->jet_cent()[1] + std::sin(inj_p_angle_random) * inj_p_rad,
                                     js->jet_cent()[2]));

      // Particle velocity in local cylindrical coordinates
      AMREX_D_TERM(Real ux_vel = jet_vel;,
                   Real ur_vel = std::cos(inj_p_alpha) * std::tan(inj_p_theta) * jet_vel;,
                   Real ut_vel = std::sin(inj_p_alpha) * std::tan(inj_p_theta) * jet_vel);

      // Particle velocity in cartesian coordinates
      AMREX_D_TERM(Real x_vel = -std::sin(inj_p_angle_random) * ut_vel + std::cos(inj_p_angle_random) * ur_vel;,
                   Real y_vel =  std::cos(inj_p_angle_random) * ut_vel + std::sin(inj_p_angle_random) * ur_vel;,
                   Real z_vel = ux_vel);

      // Particle velocity rms
      AMREX_D_TERM(Real x_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * js->jet_velrms() ;,
                   Real y_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * js->jet_velrms() ;,
                   Real z_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * js->jet_velrms() );

      // Move into RealVect
      RealVect part_vel(AMREX_D_DECL(x_vel+x_vel_rms,
                                     y_vel+y_vel_rms,
                                     z_vel+z_vel_rms));

      // Set up a new particle
      ParticleType p;
      p.id() = ParticleType::NextID();
      p.cpu() = ParallelDescriptor::MyProc();

      AMREX_D_TERM(p.rdata(SprayComps::pstateVel) = part_vel[0];,
                   p.rdata(SprayComps::pstateVel + 1) = part_vel[1];,
                   p.rdata(SprayComps::pstateVel + 2) = part_vel[2];);

      // Add particles as if they have advanced some random portion of dt
      Real pmov = amrex::Random();
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        p.pos(dir) = part_loc[dir] + pmov * dt * part_vel[dir];
      }

      p.rdata(SprayComps::pstateT) = js->get_avg_T();
      p.rdata(SprayComps::pstateDia) = cur_dia;
      for (int sp = 0; sp < SPRAY_FUEL_NUM; ++sp) {
          p.rdata(SprayComps::pstateY + sp) = 1.0;
      }

      // Put particle in place
      bool where = Where(p, pld);
      if (!where) {
          amrex::Abort("Bad injection particle");
      }
      std::pair<int, int> ind(pld.m_grid, pld.m_tile);

      host_particles[ind].push_back(p);

      Real pmass = Pi_six * rhoL_avg * std::pow(cur_dia, 3);
      total_mass += num_ppp * pmass;
      remaining_mass = injection_mass - total_mass;
      npart_inj +=1;
  }

  for (auto& kv : host_particles) {
    auto grid = kv.first.first;
    auto tile = kv.first.second;
    const auto& src_tile = kv.second;
    auto& dst_tile = GetParticles(lev)[std::make_pair(grid, tile)];
    auto old_size = dst_tile.GetArrayOfStructs().size();
    auto new_size = old_size + src_tile.size();
    dst_tile.resize(new_size);
    // Copy the AoS part of the host particles to the GPU
    amrex::Gpu::copy(
      amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
      dst_tile.GetArrayOfStructs().begin() + old_size);
  }

  // Redistribute is done outside of this function
  return true;
}

void
SprayParticleContainer::InitSprayParticles(const bool init_parts,
                                           ProbParm const& prob_parm)
{
  m_sprayJets.resize(1);
  std::string jet_name = "jet1";
  m_sprayJets[0] = std::make_unique<PressureSwirlJet>(jet_name, Geom(0));
  m_sprayJets[0]->set_inj_proc(0);
  // This ensures the initial time step size stays reasonable
  m_injectVel = m_sprayJets[0]->jet_vel();
  // Start without any particles
  return;
}
