#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd {

namespace cuda {

GridSpaceHD::GridSpaceHD(const Host *host)
    : _host{host}, _device{nullptr}, _e_node_x_hd{host->eNodeX()},
      _e_node_y_hd{host->eNodeY()}, _e_node_z_hd{host->eNodeZ()},
      _h_node_x_hd{host->hNodeX()}, _h_node_y_hd{host->hNodeY()},
      _h_node_z_hd{host->hNodeZ()}, _e_size_x_hd{host->eSizeX()},
      _e_size_y_hd{host->eSizeY()}, _e_size_z_hd{host->eSizeZ()},
      _h_size_x_hd{host->hSizeX()}, _h_size_y_hd{host->hSizeY()},
      _h_size_z_hd{host->hSizeZ()} {}

GridSpaceHD::~GridSpaceHD() { releaseDevice(); }

auto GridSpaceHD::copyHostToDevice() -> void {
  if (_host == nullptr) {
    throw std::runtime_error("Host data is not initialized");
  }

  if (_device != nullptr) {
    releaseDevice();
  }

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaMalloc(&_device, sizeof(Device)));

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
  d._based_dx = _host->basedDx();
  d._based_dy = _host->basedDy();
  d._based_dz = _host->basedDz();
  d._min_dx = _host->minDx();
  d._min_dy = _host->minDy();
  d._min_dz = _host->minDz();
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

  try {
    XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(
        cudaMemcpy(_device, &d, sizeof(Device), cudaMemcpyHostToDevice));
  } catch (const std::exception &e) {
    d._e_node_x = nullptr;
    d._e_node_y = nullptr;
    d._e_node_z = nullptr;
    d._h_node_x = nullptr;
    d._h_node_y = nullptr;
    d._h_node_z = nullptr;
    d._e_size_x = nullptr;
    d._e_size_y = nullptr;
    d._e_size_z = nullptr;
    d._h_size_x = nullptr;
    d._h_size_y = nullptr;
    d._h_size_z = nullptr;
    throw e;
  }
}

auto GridSpaceHD::copyDeviceToHost() -> void {
  // Do nothing
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

  if (_device == nullptr) {
    return;
  }

  XFDTD_CORE_CUDA_CHECK_CUDA_ERROR(cudaFree(_device));
  _device = nullptr;
}

XFDTD_CUDA_GLOBAL auto
__kenerlCheckGridSpace(const GridSpaceData *grid_space) -> void {
  if (grid_space == nullptr) {
    printf("GridSpaceData: nullptr\n");
    return;
  }
  // print based dx, dy, dz
  printf("GridSpaceData: Based dx = %f, dy = %f, dz = %f\n",
         grid_space->basedDx(), grid_space->basedDy(), grid_space->basedDz());

  printf("GridSpaceData: Size = (%lu, %lu, %lu)\n", grid_space->sizeX(),
         grid_space->sizeY(), grid_space->sizeZ());
}

} // namespace cuda

} // namespace xfdtd