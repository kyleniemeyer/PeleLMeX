#include <PeleLM.H>
#include <AMReX_ParmParse.H>

// -----------------------------------------------------------
// Read a binary file
// INPUTS/OUTPUTS:
// iname => filename
// nx    => input resolution
// ny    => input resolution
// nz    => input resolution
// data  <= output data
// -----------------------------------------------------------
void
read_binary(
  const std::string& iname,
  const size_t nx, 
  const size_t ny, 
  const size_t nz, 
  const size_t ncol,
  amrex::Vector<double>& data /*needs to be double*/)
{
  std::ifstream infile(iname, std::ios::in | std::ios::binary);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }

  for (size_t i = 0; i < nx * ny * nz * ncol; i++) {
    infile.read(reinterpret_cast<char*>(&data[i]), sizeof(data[i]));
  }
  infile.close();
}

AMREX_FORCE_INLINE
std::string
read_file(std::ifstream& in) 
{
  return static_cast<std::stringstream const&>(
           std::stringstream() << in.rdbuf())
    .str();
}

// -----------------------------------------------------------
// Read a csv file
// INPUTS/OUTPUTS:
// iname => filename
// nx    => input resolution
// ny    => input resolution
// nz    => input resolution
// data  <= output data
// -----------------------------------------------------------
void
read_csv(
  const std::string& iname,
  const size_t nx,
  const size_t ny,
  const size_t nz,
  amrex::Vector<amrex::Real>& data)
{
  std::ifstream infile(iname, std::ios::in);
  const std::string memfile = read_file(infile);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }
  infile.close();
  std::istringstream iss(memfile);

  // Read the file
  size_t nlines = 0;
  std::string firstline;
  std::string line;
  std::getline(iss, firstline); // skip header
  while (getline(iss, line)) {
    ++nlines;
  }

  // Quick sanity check
  if (nlines != nx * ny * nz) {
    amrex::Abort(
      "Number of lines in the input file (= " + std::to_string(nlines) +
      ") does not match the input resolution (=" + std::to_string(nx) + ")");
  }

  // Read the data from the file
  iss.clear();
  iss.seekg(0, std::ios::beg);
  std::getline(iss, firstline); // skip header
  int cnt = 0;
  while (std::getline(iss, line)) {
    std::istringstream linestream(line);
    std::string value;
    while (getline(linestream, value, ',')) {
      std::istringstream sinput(value);
      sinput >> data[cnt];
      cnt++;
    }
  }
}


void PeleLM::readProbParm()
{
   amrex::ParmParse pp("prob");

   //Geometric parameters
   pp.query("d_venturi", PeleLM::prob_parm->d_venturi);
   pp.query("d_swirlerexit", PeleLM::prob_parm->d_swirlerexit);
   pp.query("d_dump", PeleLM::prob_parm->d_dump);

   pp.query("l_venturi", PeleLM::prob_parm->l_venturi);
   pp.query("l_swirlercup", PeleLM::prob_parm->l_swirlercup);
   pp.query("l_dump", PeleLM::prob_parm->l_dump);

   // Chamber conditions
   pp.query("P_mean", PeleLM::prob_parm->P_mean);
   pp.query("T_mean", PeleLM::prob_parm->T_mean);

   
}
