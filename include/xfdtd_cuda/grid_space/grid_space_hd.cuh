#ifndef __XFDTD_CUDA_GRID_SPACE_HD_CUH__
#define __XFDTD_CUDA_GRID_SPACE_HD_CUH__

#include <xfdtd/grid_space/grid_space.h>

#include <xfdtd_cuda/grid_space/grid_space.cuh>
#include <xfdtd_cuda/host_device_carrier.cuh>

namespace xfdtd::cuda {

class GridSpaceHD
    : public HostDeviceCarrier<xfdtd::GridSpace, xfdtd::cuda::GridSpace> {
  using Host = xfdtd::GridSpace;
  using Device = xfdtd::cuda::GridSpace;

 public:
  GridSpaceHD(Host *host);

  ~GridSpaceHD();

  auto operator=(const GridSpaceHD &) -> GridSpaceHD & = delete;

  auto operator=(GridSpaceHD &&) -> GridSpaceHD & = delete;

  auto copyHostToDevice() -> void;

  auto copyDeviceToHost() -> void;

  auto releaseDevice() -> void;

 private:
  TensorHD<Real, 1> _e_node_x_hd, _e_node_y_hd, _e_node_z_hd;
  TensorHD<Real, 1> _h_node_x_hd, _h_node_y_hd, _h_node_z_hd;
  TensorHD<Real, 1> _e_size_x_hd, _e_size_y_hd, _e_size_z_hd;
  TensorHD<Real, 1> _h_size_x_hd, _h_size_y_hd, _h_size_z_hd;
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_GRID_SPACE_HD_CUH__