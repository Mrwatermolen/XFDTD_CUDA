#ifndef __XFDTD_CUDA_PML_CORRECTOR_CUH__
#define __XFDTD_CUDA_PML_CORRECTOR_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/util/transform/abc_xyz.h>

#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/index_task.cuh>

#include "xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh"

namespace xfdtd::cuda {

class EMF;

template <xfdtd::Axis::XYZ xyz>
class PMLCorrector {
 public:
  XFDTD_CUDA_DEVICE auto correctE() -> void;

  XFDTD_CUDA_DEVICE auto correctH() -> void;

  XFDTD_CUDA_DEVICE auto task() const -> IndexTask;

  EMF* _emf{};
  IndexTask _task{};

  Index _pml_global_e_start{}, _pml_global_h_start{};
  Index _pml_node_e_start{}, _pml_node_h_start{};
  Index _offset_c{};

  Array1D<Real>*_coeff_a_e{}, *_coeff_b_e{}, *_coeff_a_h{}, *_coeff_b_h{};
  Array3D<Real>*_c_ea_psi_hb{}, *_c_eb_psi_ha{}, *_c_ha_psi_eb{},
      *_c_hb_psi_ea{};
  Array3D<Real>*_ea_psi_hb{}, *_eb_psi_ha{}, *_ha_psi_eb{}, *_hb_psi_ea{};

 private:
  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto coeffA() const -> Array1D<Real>& {
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      return *_coeff_a_e;
    } else {
      return *_coeff_a_h;
    }
  }

  template <xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto coeffB() const -> Array1D<Real>& {
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      return *_coeff_b_e;
    } else {
      return *_coeff_b_h;
    }
  }

  template <xfdtd::EMF::Attribute attribute, Axis::XYZ xyz_0>
  XFDTD_CUDA_DEVICE auto cPsi() const -> Array3D<Real>& {
    constexpr auto xyz_a = Axis::tangentialAAxis<xyz>();
    constexpr auto xyz_b = Axis::tangentialBAxis<xyz>();

    static_assert(xyz_0 != xyz, "xyz_0 == xyz!");

    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz_0 == xyz_a) {
        return *_c_ea_psi_hb;
      } else {
        return *_c_eb_psi_ha;
      }
    } else {
      if constexpr (xyz_0 == xyz_a) {
        return *_c_ha_psi_eb;
      } else {
        return *_c_hb_psi_ea;
      }
    }
  }

  template <xfdtd::EMF::Attribute attribute, Axis::XYZ xyz_0>
  XFDTD_CUDA_DEVICE auto psi() -> Array3D<Real>& {
    constexpr auto xyz_a = Axis::tangentialAAxis<xyz>();
    constexpr auto xyz_b = Axis::tangentialBAxis<xyz>();

    static_assert(xyz_0 != xyz, "xyz_0 == xyz!");

    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz_0 == xyz_a) {
        return *_ea_psi_hb;
      } else {
        return *_eb_psi_ha;
      }
    } else {
      if constexpr (xyz_0 == xyz_a) {
        return *_ha_psi_eb;
      } else {
        return *_hb_psi_ea;
      }
    }
  }

  XFDTD_CUDA_DEVICE static auto decomposeTask(IndexTask task, Index id,
                                              Index size_x, Index size_y,
                                              Index size_z) -> IndexTask {
    auto [id_x, id_y, id_z] = columnMajorToRowMajor(id, size_x, size_y, size_z);
    auto x_range = decomposeRange(task.xRange(), id_x, size_x);
    auto y_range = decomposeRange(task.yRange(), id_y, size_y);
    auto z_range = decomposeRange(task.zRange(), id_z, size_z);
    return {x_range, y_range, z_range};
  }

  XFDTD_CUDA_DEVICE static auto decomposeRange(IndexRange range, Index id,
                                               Index size) -> IndexRange {
    auto problem_size = range.size();
    auto quotient = problem_size / size;
    auto remainder = problem_size % size;
    auto start = Index{range.start()};
    auto end = Index{range.end()};

    if (id < remainder) {
      start += id * (quotient + 1);
      end = start + quotient + 1;
      return {start, end};
    }

    start += id * quotient + remainder;
    end = start + quotient;
    return {start, end};
  }

  template <typename T>
  XFDTD_CUDA_DEVICE static constexpr auto columnMajorToRowMajor(
      T index, T size_x, T size_y, T size_z) -> std::tuple<T, T, T> {
    return std::make_tuple(index / (size_y * size_z),
                           (index % (size_y * size_z)) / size_z,
                           index % size_z);
  }
};

}  // namespace xfdtd::cuda

#endif  // __XFDTD_CUDA_PML_CORRECTOR_CUH__
