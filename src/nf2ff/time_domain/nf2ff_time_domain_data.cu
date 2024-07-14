#include <xfdtd/common/constant.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <xfdtd_cuda/calculation_param/calculation_param.cuh>
#include <xfdtd_cuda/electromagnetic_field/electromagnetic_field.cuh>

#include "nf2ff/interpolate_scheme.cuh"
#include "nf2ff/time_domain/nf2ff_time_domain_agency.cuh"
#include "nf2ff/time_domain/nf2ff_time_domain_data.cuh"
#include "xfdtd_cuda/common.cuh"

namespace xfdtd::cuda {

// define a global variable for cuda
__constant__ Real C_0 = 299792458.0;

template <typename T>
XFDTD_CUDA_DEVICE static constexpr auto columnMajorToRowMajor(
    T index, T size_x, T size_y, T size_z) -> std::tuple<T, T, T> {
  return std::make_tuple(index / (size_y * size_z),
                         (index % (size_y * size_z)) / size_z, index % size_z);
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

XFDTD_CUDA_DEVICE static auto decomposeTask(IndexTask task, Index id,
                                            Index size_x, Index size_y,
                                            Index size_z) -> IndexTask {
  auto [id_x, id_y, id_z] = columnMajorToRowMajor(id, size_x, size_y, size_z);
  auto x_range = decomposeRange(task.xRange(), id_x, size_x);
  auto y_range = decomposeRange(task.yRange(), id_y, size_y);
  auto z_range = decomposeRange(task.zRange(), id_z, size_z);
  return {x_range, y_range, z_range};
}

template <xfdtd::Axis::XYZ xyz, xfdtd::EMF::Attribute attribute>
XFDTD_CUDA_DEVICE static auto rVector(Index i, Index j, Index k,
                                      const GridSpace* grid_space) {
  if constexpr (xyz == Axis::XYZ::X) {
    return TempVector{grid_space->eNodeX().at(i), grid_space->hNodeY().at(j),
                      grid_space->hNodeZ().at(k)};

  } else if constexpr (xyz == Axis::XYZ::Y) {
    return TempVector{grid_space->hNodeX().at(i), grid_space->eNodeY().at(j),
                      grid_space->hNodeZ().at(k)};
  } else if constexpr (xyz == Axis::XYZ::Z) {
    return TempVector{grid_space->hNodeX().at(i), grid_space->hNodeY().at(j),
                      grid_space->eNodeZ().at(k)};
  }
}

template <xfdtd::Axis::XYZ xyz, xfdtd::EMF::Attribute attribute>
XFDTD_CUDA_DEVICE static auto surfaceArea(Index i, Index j, Index k,
                                          const GridSpace* grid_space) -> Real {
  if constexpr (xyz == Axis::XYZ::X) {
    return grid_space->eSizeY().at(j) * grid_space->eSizeZ().at(k);
  } else if constexpr (xyz == Axis::XYZ::Y) {
    return grid_space->eSizeZ().at(k) * grid_space->eSizeX().at(i);
  } else if constexpr (xyz == Axis::XYZ::Z) {
    return grid_space->eSizeX().at(i) * grid_space->eSizeY().at(j);
  }
}

template <xfdtd::Axis::Direction D>
XFDTD_CUDA_DEVICE auto NF2FFTimeDomainData<D>::task() const -> IndexTask {
  const auto& node_task = _task;
  // blcok
  auto size_x = static_cast<Index>(gridDim.x);
  auto size_y = static_cast<Index>(gridDim.y);
  auto size_z = static_cast<Index>(gridDim.z);
  auto id =
      blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  auto block_task = decomposeTask(node_task, id, size_x, size_y, size_z);
  // thread
  size_x = static_cast<Index>(blockDim.x);
  size_y = static_cast<Index>(blockDim.y);
  size_z = static_cast<Index>(blockDim.z);
  id = threadIdx.x + threadIdx.y * blockDim.x +
       threadIdx.z * blockDim.x * blockDim.y;

  auto thread_task = decomposeTask(block_task, id, size_x, size_y, size_z);
  return thread_task;
}

template <xfdtd::Axis::Direction D>
template <xfdtd::EMF::Attribute attribute>
XFDTD_CUDA_DEVICE auto NF2FFTimeDomainData<D>::update() -> void {
  constexpr auto filed_a = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_a));
  constexpr auto filed_b = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_b));

  constexpr Real offset = (attribute == xfdtd::EMF::Attribute::E) ? 0.5 : 0.0;
  Real coeff_a = (attribute == xfdtd::EMF::Attribute::E) ? 1.0 : -1.0;
  if (Axis::directionNegative<D>()) {
    coeff_a *= -1.0;
  }

  Real coeff_b = (attribute == xfdtd::EMF::Attribute::E) ? -1.0 : 1.0;
  if (Axis::directionNegative<D>()) {
    coeff_b *= -1.0;
  }

  auto task = this->task();
  if (!task.valid()) {
    return;
  }

  const auto grid_space = _grid_space;
  const auto calculation_param = _calculation_param;
  const auto emf = _emf;

  const auto current_time_step =
      calculation_param->timeParam()->currentTimeStep();

  const auto& field_a_value = emf->field<filed_a>();
  const auto& field_b_value = emf->field<filed_b>();

  auto [potential_a, potential_b] = potential<attribute>();

  const auto dt = calculation_param->timeParam()->dt();

  const auto is = task.xRange().start();
  const auto js = task.yRange().start();
  const auto ks = task.zRange().start();
  const auto ie = task.xRange().end();
  const auto je = task.yRange().end();
  const auto ke = task.zRange().end();

  const auto offset_i = _task.xRange().start();
  const auto offset_j = _task.yRange().start();
  const auto offset_k = _task.zRange().start();

  for (auto i{is}; i < ie; ++i) {
    for (auto j{js}; j < je; ++j) {
      for (auto k{ks}; k < ke; ++k) {
        auto r_center = rVector<xyz, attribute>(i, j, k, grid_space);
        auto time_delay = timeDelay<attribute>(r_center);
        auto time_step_delay = time_delay / (C_0 * dt);
        auto nn = static_cast<Index>(
            std::floor(time_step_delay + offset + current_time_step));
        auto coeff_f = time_step_delay + offset + current_time_step - nn;

        auto ds = surfaceArea<xyz, attribute>(i, j, k, grid_space);

        const auto a_avg = interpolate::interpolateSurfaceCenter<xyz, filed_a>(
            field_a_value, i, j, k);
        const auto b_avg = interpolate::interpolateSurfaceCenter<xyz, filed_b>(
            field_b_value, i, j, k);

        auto delta_a = (a_avg - previousAValue<attribute>(
                                    i - offset_i, j - offset_j, k - offset_k));
        auto delta_b = (b_avg - previousBValue<attribute>(
                                    i - offset_i, j - offset_j, k - offset_k));

        // Here need to reduce sum
        potential_a.at(nn) += coeff_a * (1 - coeff_f) * ds * delta_b;
        potential_a.at(nn + 1) += coeff_a * coeff_f * ds * delta_b;

        potential_b.at(nn) += coeff_b * (1 - coeff_f) * ds * delta_a;
        potential_b.at(nn + 1) += coeff_b * coeff_f * ds * delta_a;

        setPreviousAValue<attribute>(i - offset_i, j - offset_j, k - offset_k,
                                     a_avg);
        setPreviousBValue<attribute>(i - offset_i, j - offset_j, k - offset_k,
                                     b_avg);
      }
    }
  }
}

// Agnecy

template <xfdtd::Axis::Direction D>
XFDTD_CUDA_GLOBAL auto __nF2FFTimeDomainUpdate(NF2FFTimeDomainData<D>* _data)
    -> void {
  _data->template update<xfdtd::EMF::Attribute::E>();
  _data->template update<xfdtd::EMF::Attribute::H>();
}

NF2FFTimeDoaminAgency::NF2FFTimeDoaminAgency(
    NF2FFTimeDomainData<xfdtd::Axis::Direction::XN>* data_xn,
    NF2FFTimeDomainData<xfdtd::Axis::Direction::XP>* data_xp,
    NF2FFTimeDomainData<xfdtd::Axis::Direction::YN>* data_yn,
    NF2FFTimeDomainData<xfdtd::Axis::Direction::YP>* data_yp,
    NF2FFTimeDomainData<xfdtd::Axis::Direction::ZN>* data_zn,
    NF2FFTimeDomainData<xfdtd::Axis::Direction::ZP>* data_zp)
    : _xn{data_xn},
      _xp{data_xp},
      _yn{data_yn},
      _yp{data_yp},
      _zn{data_zn},
      _zp{data_zp} {}

auto NF2FFTimeDoaminAgency::update(dim3 grid_dim, dim3 block_dim) -> void {
  // TODO(franzero): We don't implement the reduceSum, So we just call the
  grid_dim = {1, 1, 1};
  block_dim = {1, 1, 1};
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_xn);
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_xp);
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_yn);
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_yp);
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_zn);
  __nF2FFTimeDomainUpdate<<<grid_dim, block_dim>>>(_zp);
}

// explicit instantiation
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::XN>;
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::XP>;
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::YN>;
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::YP>;
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::ZN>;
template class NF2FFTimeDomainData<xfdtd::Axis::Direction::ZP>;

}  // namespace xfdtd::cuda
