#include <xfdtd/common/constant.h>
#include <xfdtd/common/type_define.h>
#include <xfdtd/material/dispersive_material.h>

#include "dispersive_sphere_scatter.cuh"

auto mLorentzSphereScatter(dim3 grid_dim, dim3 block_dim) -> void {
  auto omega_p = xfdtd::Array1D<xfdtd::Real>{2 * xfdtd::constant::PI * 2e9};
  auto gamma = xfdtd::Array1D<xfdtd::Real>{xfdtd::constant::PI * 2e9};
  auto epsilon_inf = 2;
  auto epsilon_static = xfdtd::Array1D<xfdtd::Real>{5};
  auto nv = xfdtd::Array1D<xfdtd::Real>{xfdtd::constant::PI * 2e9};

  {
    testCase(xfdtd::MLorentzMaterial::makeMLorentz(
                 "lorentz_medium", epsilon_inf,
                 (epsilon_static - epsilon_inf) * omega_p * omega_p, {0},
                 omega_p * omega_p, 2 * nv, {1}),
             1, grid_dim, block_dim);
  }

  {
    testCase(xfdtd::MLorentzMaterial::makeMLorentz(
                 "lorentz_medium", epsilon_inf,
                 (epsilon_static - epsilon_inf) * omega_p * omega_p, {0},
                 omega_p * omega_p, 2 * nv, {1}),
             0, grid_dim, block_dim);
  }
}

int main(int argc, char *argv[]) {
  dim3 grid_dim{4, 4, 4};
  dim3 block_dim{8, 8, 8};

  if (7 <= argc) {
    grid_dim.x = std::atoi(argv[1]);
    grid_dim.y = std::atoi(argv[2]);
    grid_dim.z = std::atoi(argv[3]);
    block_dim.x = std::atoi(argv[4]);
    block_dim.y = std::atoi(argv[5]);
    block_dim.z = std::atoi(argv[6]);
  }

  std::cout << "Grid size: (" << grid_dim.x << ", " << grid_dim.y << ", "
            << grid_dim.z << "), Block size: (" << block_dim.x << ", "
            << block_dim.y << ", " << block_dim.z << ")\n";

  mLorentzSphereScatter(grid_dim, block_dim);
}
