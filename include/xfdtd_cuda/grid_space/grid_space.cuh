#ifndef __XFDTD_CUDA_GRID_SPACE_GRID_SPACE_CUH__
#define __XFDTD_CUDA_GRID_SPACE_GRID_SPACE_CUH__

#include <xfdtd/common/type_define.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/tensor.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

namespace xfdtd {

namespace cuda {

/**
 * @brief Holds the grid space data
 *
 */
class GridSpace {
  friend class GridSpaceHD;

 public:
  XFDTD_CUDA_DUAL auto basedDx() const -> Real { return _based_dx; }

  XFDTD_CUDA_DUAL auto basedDy() const -> Real { return _based_dy; }

  XFDTD_CUDA_DUAL auto basedDz() const -> Real { return _based_dz; }

  XFDTD_CUDA_DUAL auto minDx() const -> Real { return _min_dx; }

  XFDTD_CUDA_DUAL auto minDy() const -> Real { return _min_dy; }

  XFDTD_CUDA_DUAL auto minDz() const -> Real { return _min_dz; }

  /**
   * @brief e node means that the corner of the cell
   *
   * @return const Array1D<Real>&
   */
  XFDTD_CUDA_DUAL auto eNodeX() const -> const Array1D<Real> & {
    return *_e_node_x;
  }

  /**
   * @brief h node means that the center of the cell
   *
   * @return const Array1D<Real>&
   */
  XFDTD_CUDA_DUAL auto hNodeX() const -> const Array1D<Real> & {
    return *_h_node_x;
  }

  XFDTD_CUDA_DUAL auto eNodeY() const -> const Array1D<Real> & {
    return *_e_node_y;
  }

  XFDTD_CUDA_DUAL auto eNodeZ() const -> const Array1D<Real> & {
    return *_e_node_z;
  }

  XFDTD_CUDA_DUAL auto hNodeY() const -> const Array1D<Real> & {
    return *_h_node_y;
  }

  XFDTD_CUDA_DUAL auto hNodeZ() const -> const Array1D<Real> & {
    return *_h_node_z;
  }

  XFDTD_CUDA_DUAL auto eSizeX() const -> const Array1D<Real> & {
    return *_e_size_x;
  }

  XFDTD_CUDA_DUAL auto eSizeY() const -> const Array1D<Real> & {
    return *_e_size_y;
  }

  XFDTD_CUDA_DUAL auto eSizeZ() const -> const Array1D<Real> & {
    return *_e_size_z;
  }

  XFDTD_CUDA_DUAL auto hSizeX() const -> const Array1D<Real> & {
    return *_h_size_x;
  }

  XFDTD_CUDA_DUAL auto hSizeY() const -> const Array1D<Real> & {
    return *_h_size_y;
  }

  XFDTD_CUDA_DUAL auto hSizeZ() const -> const Array1D<Real> & {
    return *_h_size_z;
  }

  /**
   * @brief The number of cells in x direction
   *
   * @return Index
   */
  XFDTD_CUDA_DUAL auto sizeX() const -> Index { return hNodeX().size(); }

  /**
   * @brief The number of cells in y direction
   *
   * @return Index
   */
  XFDTD_CUDA_DUAL auto sizeY() const -> Index { return hNodeY().size(); }

  /**
   * @brief The number of cells in z direction
   *
   * @return Index
   */
  XFDTD_CUDA_DUAL auto sizeZ() const -> Index { return hNodeZ().size(); }

 private:
  Real _based_dx{}, _based_dy{}, _based_dz{};
  Real _min_dx{}, _min_dy{}, _min_dz{};

  Array1D<Real> *_e_node_x{}, *_e_node_y{}, *_e_node_z{};
  Array1D<Real> *_h_node_x{}, *_h_node_y{}, *_h_node_z{};
  Array1D<Real> *_e_size_x{}, *_e_size_y{}, *_e_size_z{};
  Array1D<Real> *_h_size_x{}, *_h_size_y{}, *_h_size_z{};
};

XFDTD_CUDA_GLOBAL auto __kenerlCheckGridSpace(const GridSpace *grid_space)
    -> void;

}  // namespace cuda

}  // namespace xfdtd

#endif  // __XFDTD_CUDA_GRID_SPACE_GRID_SPACE_CUH__
