#include <xfdtd_cuda/grid_space/grid_space_hd.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd {

namespace cuda {

GridSpaceHD::GridSpaceHD(Host *host)
    : HostDeviceCarrier<Host, Device>{host},
      _e_node_x_hd{host->eNodeX()},
      _e_node_y_hd{host->eNodeY()},
      _e_node_z_hd{host->eNodeZ()},
      _h_node_x_hd{host->hNodeX()},
      _h_node_y_hd{host->hNodeY()},
      _h_node_z_hd{host->hNodeZ()},
      _e_size_x_hd{host->eSizeX()},
      _e_size_y_hd{host->eSizeY()},
      _e_size_z_hd{host->eSizeZ()},
      _h_size_x_hd{host->hSizeX()},
      _h_size_y_hd{host->hSizeY()},
      _h_size_z_hd{host->hSizeZ()} {}

GridSpaceHD::~GridSpaceHD() { releaseDevice(); }

auto GridSpaceHD::copyHostToDevice() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("Host data is not initialized");
  }

  _e_node_x_hd.copyHostToDevice();
  _e_node_y_hd.copyHostToDevice();
  _e_node_z_hd.copyHostToDevice();
  _h_node_x_hd.copyHostToDevice();
  _h_node_y_hd.copyHostToDevice();
  _h_node_z_hd.copyHostToDevice();
  _e_size_x_hd.copyHostToDevice();
  _e_size_y_hd.copyHostToDevice();
  _e_size_z_hd.copyHostToDevice();
  _h_size_x_hd.copyHostToDevice();
  _h_size_y_hd.copyHostToDevice();
  _h_size_z_hd.copyHostToDevice();

  auto d = Device{};
  d._based_dx = host()->basedDx();
  d._based_dy = host()->basedDy();
  d._based_dz = host()->basedDz();
  d._min_dx = host()->minDx();
  d._min_dy = host()->minDy();
  d._min_dz = host()->minDz();
  d._e_node_x = _e_node_x_hd.device();
  d._e_node_y = _e_node_y_hd.device();
  d._e_node_z = _e_node_z_hd.device();
  d._h_node_x = _h_node_x_hd.device();
  d._h_node_y = _h_node_y_hd.device();
  d._h_node_z = _h_node_z_hd.device();
  d._e_size_x = _e_size_x_hd.device();
  d._e_size_y = _e_size_y_hd.device();
  d._e_size_z = _e_size_z_hd.device();
  d._h_size_x = _h_size_x_hd.device();
  d._h_size_y = _h_size_y_hd.device();
  d._h_size_z = _h_size_z_hd.device();

  copyToDevice(&d);
}

auto GridSpaceHD::copyDeviceToHost() -> void {
  if (host() == nullptr) {
    throw std::runtime_error("Host data is not initialized");
  }
}

auto GridSpaceHD::releaseDevice() -> void {
  _e_node_x_hd.releaseDevice();
  _e_node_y_hd.releaseDevice();
  _e_node_z_hd.releaseDevice();
  _h_node_x_hd.releaseDevice();
  _h_node_y_hd.releaseDevice();
  _h_node_z_hd.releaseDevice();
  _e_size_x_hd.releaseDevice();
  _e_size_y_hd.releaseDevice();
  _e_size_z_hd.releaseDevice();
  _h_size_x_hd.releaseDevice();
  _h_size_y_hd.releaseDevice();
  _h_size_z_hd.releaseDevice();

  releaseBaseDevice();
}

XFDTD_CUDA_GLOBAL auto __kenerlCheckGridSpace(const GridSpaceData *grid_space)
    -> void {
  if (grid_space == nullptr) {
    printf("GridSpaceData: nullptr\n");
    return;
  }
  // print based dx, dy, dz
  printf("GridSpaceData: Based dx = %f, dy = %f, dz = %f\n",
         grid_space->basedDx(), grid_space->basedDy(), grid_space->basedDz());

  printf("GridSpaceData: Size = (%lu, %lu, %lu)\n", grid_space->sizeX(),
         grid_space->sizeY(), grid_space->sizeZ());

  // print e node x
  printf("GridSpaceData: e node x = ");
  for (size_t i = 0; i < grid_space->eNodeX().size(); i++) {
    printf("%.3e ", grid_space->eNodeX()[i]);
  }
}

}  // namespace cuda

}  // namespace xfdtd