#include <xfdtd_cuda/grid_space/grid_space.cuh>

namespace xfdtd::cuda {

XFDTD_CUDA_GLOBAL auto __kenerlCheckGridSpace(const GridSpace *grid_space)
    -> void {
  if (grid_space == nullptr) {
    std::printf("GridSpaceData: nullptr\n");
    return;
  }
  // print based dx, dy, dz
  std::printf("GridSpaceData: Based dx = %f, dy = %f, dz = %f\n",
              grid_space->basedDx(), grid_space->basedDy(),
              grid_space->basedDz());

  std::printf("GridSpaceData: Size = (%lu, %lu, %lu)\n", grid_space->sizeX(),
              grid_space->sizeY(), grid_space->sizeZ());

  // print e node x
  std::printf("GridSpaceData: e node x = ");
  for (size_t i = 0; i < grid_space->eNodeX().size(); i++) {
    std::printf("%.3e ", grid_space->eNodeX()[i]);
  }
}

}  // namespace xfdtd::cuda
