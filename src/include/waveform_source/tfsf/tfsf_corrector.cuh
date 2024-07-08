#ifndef __XFDTD_CUDA_TFSF_CORRECTOR_CUH__
#define __XFDTD_CUDA_TFSF_CORRECTOR_CUH__

#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>
#include <xfdtd/util/transform/abc_xyz.h>

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/common.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>
#include <xfdtd_cuda/index_task.cuh>

namespace xfdtd::cuda {

class TFSFCorrector {
  friend class TFSFCorrectorHD;

 public:
  template <Axis::Direction direction, xfdtd::EMF::Attribute attribute,
            Axis::XYZ field_xyz>
  XFDTD_CUDA_DEVICE auto correctTFSF(IndexTask task, Index offset_i,
                                     Index offset_j, Index offset_k) {
    if (!task.valid()) {
      return;
    }

    constexpr auto xyz = Axis::fromDirectionToXYZ<direction>();
    constexpr auto dual_xyz_a = Axis::tangentialAAxis<xyz>();
    constexpr auto dual_xyz_b = Axis::tangentialBAxis<xyz>();

    static_assert(field_xyz == dual_xyz_a || field_xyz == dual_xyz_b,
                  "Field XYZ must be dual");

    // xyz \times field_xyz
    // constant auto dual_field_xyz = Axis::cross<xyz, field_xyz>();
    constexpr auto dual_field_xyz =
        (field_xyz == dual_xyz_a) ? dual_xyz_b : dual_xyz_a;

    constexpr auto dual_attribute = xfdtd::EMF::dualAttribute(attribute);

    auto [as, bs, cs] = transform::xYZToABC<Index, xyz>(
        task.xRange().start(), task.yRange().start(), task.zRange().start());
    auto [ae, be, ce] = transform::xYZToABC<Index, xyz>(
        task.xRange().end(), task.yRange().end(), task.zRange().end());

    // EA: [as, ae), [bs, be+1), [c]
    // EB: [as, ae+1), [bs, be), [c]
    // HA: [as, ae+1), [bs, be), [c]
    // HB: [as, ae), [bs, be+1), [c]
    constexpr auto offset_a =
        ((attribute == xfdtd::EMF::Attribute::E && field_xyz == dual_xyz_b) ||
         (attribute == xfdtd::EMF::Attribute::H && field_xyz == dual_xyz_a))
            ? 1
            : 0;
    constexpr auto offset_b = offset_a == 1 ? 0 : 1;
    if constexpr (offset_a == 1) {
      if (ae == getEnd<dual_xyz_a>()) {
        ae += offset_a;
      }
    }

    if constexpr (offset_b == 1) {
      if (be == getEnd<dual_xyz_b>()) {
        be += offset_b;
      }
    }

    cs = (Axis::directionNegative<direction>()) ? cs : ce;
    ce = cs + 1;

    if constexpr (Axis::directionNegative<direction>()) {
      if (cs != getStart<xyz>()) {
        return;
      }
    } else {
      if (cs != getEnd<xyz>()) {
        return;
      }
    }

    // E in total field need add incident for forward H.
    // H in scattered field need deduct incident for backward E.
    constexpr auto compensate_flag = 1;
    // E = E + c_h * \times H
    // H = H - c_e * \times E
    constexpr auto equation_flag =
        (attribute == xfdtd::EMF::Attribute::E) ? 1 : -1;
    // EA: \times H = ( \partial H_c / \partial b - \partial H_b / \partial c )
    // EB: \times H = ( \partial H_a / \partial c - \partial H_c / \partial a )
    constexpr auto different_flag = (field_xyz == dual_xyz_a) ? -1 : 1;
    // different is (index) - (index - 1)
    // Direction in negative: -
    // Direction in positive: +
    constexpr auto direction_flag =
        Axis::directionNegative<direction>() ? -1 : 1;
    // OK. we get coefficient flag
    constexpr auto coefficient_flag =
        compensate_flag * equation_flag * different_flag * direction_flag;

    if constexpr (direction == Axis::Direction::XN &&
                  attribute == xfdtd::EMF::Attribute::E &&
                  field_xyz == Axis::XYZ::Z) {
      static_assert(coefficient_flag == -1, "Coefficient flag error");
    }

    // Global index
    auto [is, js, ks] = transform::aBCToXYZ<Index, xyz>(as, bs, cs);
    auto [ie, je, ke] = transform::aBCToXYZ<Index, xyz>(ae, be, ce);

    const auto t = calculationParam()->timeParam()->currentTimeStep();

    auto emf = this->emf();
    for (Index i{is}; i < ie; ++i) {
      for (Index j{js}; j < je; ++j) {
        for (Index k{ks}; k < ke; ++k) {
          const auto i_node = i - offset_i;
          const auto j_node = j - offset_j;
          const auto k_node = k - offset_k;

          auto [a, b, c] = transform::xYZToABC<Index, xyz>(i, j, k);

          if constexpr (Axis::directionNegative<direction>()) {
            if constexpr (attribute == xfdtd::EMF::Attribute::E) {
              auto [i_dual, j_dual, k_dual] =
                  transform::aBCToXYZ<Index, xyz>(a, b, c - 1);
              const auto dual_incident_field_v =
                  getInc<dual_attribute, dual_field_xyz>(t, i_dual, j_dual,
                                                         k_dual);
              const auto coefficient =
                  getCoefficient<xyz, attribute>() * coefficient_flag;

              auto&& field_v =
                  emf->field<attribute, field_xyz>()(i_node, j_node, k_node);
              field_v += dual_incident_field_v * coefficient;
            } else {
              const auto dual_incident_field_v =
                  getInc<dual_attribute, dual_field_xyz>(t, i, j, k);

              const auto coefficient =
                  getCoefficient<xyz, attribute>() * coefficient_flag;

              auto [ii, jj, kk] = transform::aBCToXYZ<Index, xyz>(a, b, c - 1);
              const auto ii_node = ii - offset_i;
              const auto jj_node = jj - offset_j;
              const auto kk_node = kk - offset_k;
              auto&& field_v =
                  emf->field<attribute, field_xyz>()(ii_node, jj_node, kk_node);
              field_v += dual_incident_field_v * coefficient;
            }

            continue;
          }

          auto&& field_v =
              emf->field<attribute, field_xyz>()(i_node, j_node, k_node);

          const auto dual_incident_field_v =
              getInc<dual_attribute, dual_field_xyz>(t, i, j, k);

          const auto coefficient =
              getCoefficient<xyz, attribute>() * coefficient_flag;

          field_v += dual_incident_field_v * coefficient;
        }
      }
    }
  };

  template <xfdtd::EMF::Attribute attribute, Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto getInc(Index t, Index i, Index j, Index k) const {
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        return exInc(t, i, j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return eyInc(t, i, j, k);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return ezInc(t, i, j, k);
      }
    } else {
      if constexpr (xyz == Axis::XYZ::X) {
        return hxInc(t, i, j, k);
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return hyInc(t, i, j, k);
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return hzInc(t, i, j, k);
      }
    }
  }

  template <Axis::XYZ xyz, xfdtd::EMF::Attribute attribute>
  XFDTD_CUDA_DEVICE auto getCoefficient() const {
    if constexpr (attribute == xfdtd::EMF::Attribute::E) {
      if constexpr (xyz == Axis::XYZ::X) {
        return cax();
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return cay();
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return caz();
      }
    } else {
      if constexpr (xyz == Axis::XYZ::X) {
        return cbx();
      } else if constexpr (xyz == Axis::XYZ::Y) {
        return cby();
      } else if constexpr (xyz == Axis::XYZ::Z) {
        return cbz();
      }
    }
  }

  template <Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto getStart() const -> Index {
    if constexpr (xyz == Axis::XYZ::X) {
      return _total_task.xRange().start();
    } else if constexpr (xyz == Axis::XYZ::Y) {
      return _total_task.yRange().start();
    } else if constexpr (xyz == Axis::XYZ::Z) {
      return _total_task.zRange().start();
    }
  }

  template <Axis::XYZ xyz>
  XFDTD_CUDA_DEVICE auto getEnd() const -> Index {
    if constexpr (xyz == Axis::XYZ::X) {
      return _total_task.xRange().end();
    } else if constexpr (xyz == Axis::XYZ::Y) {
      return _total_task.yRange().end();
    } else if constexpr (xyz == Axis::XYZ::Z) {
      return _total_task.zRange().end();
    }
  }

  XFDTD_CUDA_DEVICE auto task() const;

  XFDTD_CUDA_DEVICE auto globalStartI() const -> Index;

  XFDTD_CUDA_DEVICE auto globalStartJ() const -> Index;

  XFDTD_CUDA_DEVICE auto globalStartK() const -> Index;

  XFDTD_CUDA_DEVICE auto exInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto eyInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto ezInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto hxInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto hyInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE auto hzInc(Index t, Index i, Index j,
                               Index k) const -> Real;

  XFDTD_CUDA_DEVICE Real cax() const { return _cax; }

  XFDTD_CUDA_DEVICE Real cay() const { return _cay; }

  XFDTD_CUDA_DEVICE Real caz() const { return _caz; }

  XFDTD_CUDA_DEVICE Real cbx() const { return _cbx; }

  XFDTD_CUDA_DEVICE Real cby() const { return _cby; }

  XFDTD_CUDA_DEVICE Real cbz() const { return _cbz; }

  XFDTD_CUDA_DEVICE auto calculationParam() const -> const CalculationParam*;

  XFDTD_CUDA_DEVICE auto emf() const -> const EMF*;

  XFDTD_CUDA_DEVICE auto calculationParam() -> const CalculationParam*;

  XFDTD_CUDA_DEVICE auto emf() -> EMF*;

 private:
  const xfdtd::cuda::CalculationParam* _calculation_param{};
  xfdtd::cuda::EMF* _emf{};
  Index _node_offset_i{}, _node_offset_j{}, _node_offset_k{};
  IndexTask _total_task{};

  const Array1D<Real>* _projection_x_int{};
  const Array1D<Real>* _projection_y_int{};
  const Array1D<Real>* _projection_z_int{};
  const Array1D<Real>* _projection_x_half{};
  const Array1D<Real>* _projection_y_half{};
  const Array1D<Real>* _projection_z_half{};

  const Array2D<Real>*_e_inc{}, *_h_inc{};
  Real _cax, _cbx, _cay, _cby, _caz, _cbz;
  Real _transform_e_x, _transform_e_y, _transform_e_z, _transform_h_x,
      _transform_h_y, _transform_h_z;

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

#endif  // __XFDTD_CUDA_TFSF_CORRECTOR_CUH__
