#include <xfdtd/boundary/pml.h>
#include <xfdtd/common/constant.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/material/dispersive_material.h>
#include <xfdtd/material/material.h>
#include <xfdtd/monitor/field_monitor.h>
#include <xfdtd/monitor/movie_monitor.h>
#include <xfdtd/nffft/nffft_frequency_domain.h>
#include <xfdtd/object/object.h>
#include <xfdtd/shape/cube.h>
#include <xfdtd/shape/sphere.h>
#include <xfdtd/simulation/simulation.h>
#include <xfdtd/waveform_source/tfsf_3d.h>

#include <complex>
#include <filesystem>
#include <memory>
#include <string>
#include <xfdtd_cuda/simulation/simulation_hd.cuh>
#include <xtensor/xnpy.hpp>

inline void outputRelativePermittivity(
    const xfdtd::Array1D<xfdtd::Real>& freq,
    const std::shared_ptr<xfdtd::LinearDispersiveMaterial>& material,
    const std::string& file_name) {
  xt::dump_npy(file_name, xt::stack(xt::xtuple(
                              freq, material->relativePermittivity(freq))));
}

inline void runSimulation(std::shared_ptr<xfdtd::Material> sphere_material,
                          std::string_view dir, dim3 grid_dim, dim3 block_dim,
                          const xt::xarray<double>& freq) {
  const std::filesystem::path sphere_scatter_dir{dir};

  if (!std::filesystem::exists(sphere_scatter_dir) ||
      !std::filesystem::is_directory(sphere_scatter_dir)) {
    std::filesystem::create_directories(sphere_scatter_dir);
  }

  std::cout << "Save dir: " << std::filesystem::absolute(sphere_scatter_dir)
            << "\n";

  constexpr double dl{7.5e-3};

  auto domain{std::make_shared<xfdtd::Object>(
      "domain",
      std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, -0.175, -0.175},
                                    xfdtd::Vector{0.35, 0.35, 0.35}),
      xfdtd::Material::createAir())};

  constexpr double radius = 1e-1;
  auto sphere = std::make_shared<xfdtd::Object>(
      "scatter",
      std::make_unique<xfdtd::Sphere>(xfdtd::Vector{0, 0, 0}, radius),
      sphere_material);

  constexpr auto l_min{dl * 20};
  constexpr auto tau{l_min / 6e8};
  constexpr auto t_0{4.5 * tau};
  constexpr std::size_t tfsf_start{static_cast<size_t>(15)};
  auto tfsf{
      std::make_shared<xfdtd::TFSF3D>(tfsf_start, tfsf_start, tfsf_start, 0, 0,
                                      0, xfdtd::Waveform::gaussian(tau, t_0))};

  constexpr std::size_t nffft_start{static_cast<size_t>(11)};
  auto nffft_fd{std::make_shared<xfdtd::NFFFTFrequencyDomain>(
      nffft_start, nffft_start, nffft_start, freq,
      (sphere_scatter_dir / "fd").string())};

  auto movie_ex_xz{std::make_shared<xfdtd::MovieMonitor>(
      std::make_unique<xfdtd::FieldMonitor>(
          std::make_unique<xfdtd::Cube>(xfdtd::Vector{-0.175, 0, -0.175},
                                        xfdtd::Vector{0.35, dl, 0.35}),
          xfdtd::EMF::Field::EX, "", ""),
      10, "movie_ex_xz", (sphere_scatter_dir / "movie_ex_xz").string())};
  auto movie_ex_yz = std::make_shared<xfdtd::MovieMonitor>(
      std::make_unique<xfdtd::FieldMonitor>(
          std::make_unique<xfdtd::Cube>(xfdtd::Vector{0, -0.175, -0.175},
                                        xfdtd::Vector{dl, 0.35, 0.35}),
          xfdtd::EMF::Field::EX, "", ""),
      10, "movie_ex_yz", (sphere_scatter_dir / "movie_ex_yz").string());

  auto simulation =
      xfdtd::Simulation{dl, dl, dl, 0.9, xfdtd::ThreadConfig{2, 1, 1}};
  simulation.addObject(domain);
  simulation.addObject(sphere);
  simulation.addWaveformSource(tfsf);
  simulation.addNF2FF(nffft_fd);
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XN));
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::XP));
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YN));
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::YP));
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZN));
  simulation.addBoundary(
      std::make_shared<xfdtd::PML>(8, xfdtd::Axis::Direction::ZP));
  //   simulation.addMonitor(movie_ex_xz);
  //   simulation.addMonitor(movie_ex_yz);
  //   simulation.run(1000);

  auto simulation_hd = xfdtd::cuda::SimulationHD{&simulation};
  simulation_hd.setGridDim(grid_dim);
  simulation_hd.setBlockDim(block_dim);
  std::chrono::high_resolution_clock::time_point start_time =
      std::chrono::high_resolution_clock::now();
  simulation_hd.run(1400);

  nffft_fd->processFarField(
      xfdtd::constant::PI * 0.5,
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360),
      "xy");

  nffft_fd->processFarField(
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360), 0,
      "xz");

  nffft_fd->processFarField(
      xt::linspace<double>(-xfdtd::constant::PI, xfdtd::constant::PI, 360),
      xfdtd::constant::PI * 0.5, "yz");

  if (!xfdtd::MpiSupport::instance().isRoot()) {
    return;
  }

  auto time = tfsf->waveform()->time();
  auto incident_wave_data = tfsf->waveform()->value();
  if (!xfdtd::MpiSupport::instance().isRoot()) {
    return;
  }
  xt::dump_npy((sphere_scatter_dir / "time.npy").string(), time);
  xt::dump_npy((sphere_scatter_dir / "incident_wave.npy").string(),
               incident_wave_data);

  std::chrono::high_resolution_clock::time_point end_time =
      std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> time_span = end_time - start_time;
  std::cout << "Elapsed time: " << time_span.count() << " ms\n";
  std::cout << "Elapsed time: " << time_span.count() / 1000 << " s\n";
}

inline void testCase(
    const std::shared_ptr<xfdtd::LinearDispersiveMaterial>& material, int id,
    dim3 grid_dim, dim3 block_dim, double concerned_freq = 1e9) {
  std::filesystem::path data_dir =
      std::filesystem::path("./tmp/data/dispersive_material_scatter");

  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << concerned_freq / 1e9 << "_GHz";
  const auto concerned_freq_str = ss.str();

  auto eps = material->relativePermittivity({concerned_freq}).front();
  auto eps_r = std::real(eps);
  auto sigma = -std::imag(eps) * xfdtd::constant::EPSILON_0 * 2 *
               xfdtd::constant::PI * concerned_freq;

  auto matched_material_em = xfdtd::ElectroMagneticProperty{eps_r, 1, sigma, 0};

  std::cout << "Dispersive material: " << material->name() << " is equal to "
            << matched_material_em.toString() << " in " << concerned_freq_str
            << "\n";

  if (id != 0) {
    std::cout << "Simulation with dispersive material: " << material->name()
              << "\n";

    data_dir /= material->name();

    runSimulation(material, data_dir.string(), grid_dim, block_dim,
                  {concerned_freq});
  } else {
    auto non_dispersive_material = std::make_shared<xfdtd::Material>(
        material->name() + "_matched_" + concerned_freq_str,
        matched_material_em);

    std::cout << "Simulation with non-dispersive material: "
              << non_dispersive_material->name() << "\n";
    std::cout << "Non-dispersive material: "
              << non_dispersive_material->toString() << "\n";

    data_dir /= non_dispersive_material->name();
    runSimulation(non_dispersive_material, data_dir.string(), grid_dim,
                  block_dim, {concerned_freq});
  }

  outputRelativePermittivity(
      xt::linspace<double>(concerned_freq / 100, 5 * concerned_freq, 100),
      material, (data_dir / "relative_permittivity.npy").string());
}
