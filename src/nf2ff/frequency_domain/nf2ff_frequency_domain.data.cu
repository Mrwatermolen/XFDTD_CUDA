#include <xfdtd/common/type_define.h>
#include <xfdtd/coordinate_system/coordinate_system.h>
#include <xfdtd/electromagnetic_field/electromagnetic_field.h>

#include <cstdio>
#include <cstdlib>

#include "nf2ff/frequency_domain/nf2ff_frequency_domain_agency.cuh"
#include "nf2ff/frequency_domain/nf2ff_frequency_domain_data.cuh"
#include "nf2ff/interpolate_scheme.cuh"
#include "xfdtd_cuda/common.cuh"
#include "xfdtd_cuda/tensor.cuh"

namespace xfdtd::cuda {

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

template <xfdtd::Axis::Direction D>
XFDTD_CUDA_DEVICE auto NF2FFFrequencyDomainData<D>::task() const -> IndexTask {
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
XFDTD_CUDA_DEVICE auto NF2FFFrequencyDomainData<D>::ds(Index i, Index j,
                                                       Index k) const -> Real {
  std::printf("Warning: NF2FFFrequencyDomainData::ds is not implemented\n");
  return 0.0;
}

template <xfdtd::Axis::Direction D>
XFDTD_CUDA_DEVICE auto NF2FFFrequencyDomainData<D>::calculateJ() -> void {
  auto task = this->task();
  if (!task.valid()) {
    return;
  }

  constexpr Real coff_ja = (Axis::directionPositive<D>()) ? -1.0 : 1.0;
  constexpr Real coff_jb = (Axis::directionPositive<D>()) ? 1.0 : -1.0;

  const auto is = task.xRange().start();
  const auto js = task.yRange().start();
  const auto ks = task.zRange().start();
  const auto ie = task.xRange().end();
  const auto je = task.yRange().end();
  const auto ke = task.zRange().end();

  auto&& ja = *_ja;
  auto&& jb = *_jb;
  // const auto grid_space = *_grid_space;
  const auto calculation_param = _calculation_param;
  const auto emf = _emf;
  // const auto& transform_e = *_transform_e;
  const auto& transform_h = *_transform_h;

  constexpr auto attribute = xfdtd::EMF::Attribute::H;
  constexpr auto field_a = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_a));
  constexpr auto field_b = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_b));

  const auto& ha = emf->field<field_a>();
  const auto& hb = emf->field<field_b>();

  const auto current_time_step =
      calculation_param->timeParam()->currentTimeStep();

  const auto offset_i = _task.xRange().start();
  const auto offset_j = _task.yRange().start();
  const auto offset_k = _task.zRange().start();

  for (auto i{is}; i < ie; ++i) {
    for (auto j{js}; j < je; ++j) {
      for (auto k{ks}; k < ke; ++k) {
        ja(i - offset_i, j - offset_j, k - offset_k) +=
            coff_ja *
            interpolate::interpolateSurfaceCenter<xyz, field_b>(hb, i, j, k) *
            transform_h(current_time_step);
        jb(i - offset_i, j - offset_j, k - offset_k) +=
            coff_jb *
            interpolate::interpolateSurfaceCenter<xyz, field_a>(ha, i, j, k) *
            transform_h(current_time_step);
      }
    }
  }
}

template <xfdtd::Axis::Direction D>
XFDTD_CUDA_DEVICE auto NF2FFFrequencyDomainData<D>::calculateM() -> void {
  const auto task = this->task();

  if (!task.valid()) {
    return;
  }

  Real coff_ma = (Axis::directionPositive<D>()) ? 1.0 : -1.0;
  Real coff_mb = (Axis::directionPositive<D>()) ? -1.0 : 1.0;

  const auto is = task.xRange().start();
  const auto js = task.yRange().start();
  const auto ks = task.zRange().start();
  const auto ie = task.xRange().end();
  const auto je = task.yRange().end();
  const auto ke = task.zRange().end();

  auto&& ma = *_ma;
  auto&& mb = *_mb;

  // const auto grid_space = *_grid_space;
  const auto calculation_param = _calculation_param;
  const auto emf = _emf;
  const auto& transform_e = *_transform_e;
  // const auto& transform_h = *_transform_h;

  constexpr auto attribute = xfdtd::EMF::Attribute::E;
  constexpr auto field_a = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_a));
  constexpr auto field_b = xfdtd::EMF::attributeComponentToField(
      attribute, xfdtd::EMF::xYZToComponent(xyz_b));

  const auto& ea = emf->field<field_a>();
  const auto& eb = emf->field<field_b>();

  const auto current_time_step =
      calculation_param->timeParam()->currentTimeStep();

  const auto offset_i = _task.xRange().start();
  const auto offset_j = _task.yRange().start();
  const auto offset_k = _task.zRange().start();

  for (auto i{is}; i < ie; ++i) {
    for (auto j{js}; j < je; ++j) {
      for (auto k{ks}; k < ke; ++k) {
        ma(i - offset_i, j - offset_j, k - offset_k) +=
            coff_ma *
            interpolate::interpolateSurfaceCenter<xyz, field_b>(eb, i, j, k) *
            transform_e(current_time_step);
        mb(i - offset_i, j - offset_j, k - offset_k) +=
            coff_mb *
            interpolate::interpolateSurfaceCenter<xyz, field_a>(ea, i, j, k) *
            transform_e(current_time_step);
      }
    }
  }
}

// Agency
template <xfdtd::Axis::Direction D>
XFDTD_CUDA_GLOBAL auto __nF2FFFrequencyDomainUpdate(
    NF2FFFrequencyDomainData<D>* data) -> void {
  data->calculateJ();
  data->calculateM();
}

NF2FFFrequencyDomainAgency::NF2FFFrequencyDomainAgency(
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XN>* data_xn,
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XP>* data_xp,
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YN>* data_yn,
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YP>* data_yp,
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZN>* data_zn,
    NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZP>* data_zp)
    : _data_xn{data_xn},
      _data_xp{data_xp},
      _data_yn{data_yn},
      _data_yp{data_yp},
      _data_zn{data_zn},
      _data_zp{data_zp} {}

auto NF2FFFrequencyDomainAgency::update(dim3 grid_dim, dim3 block_dim) -> void {
  auto grid_dim_x = dim3{1, grid_dim.y, grid_dim.z};
  auto grid_dim_y = dim3{grid_dim.x, 1, grid_dim.z};
  auto grid_dim_z = dim3{grid_dim.x, grid_dim.y, 1};
  auto block_dim_x = dim3{1, block_dim.y, block_dim.z};
  auto block_dim_y = dim3{block_dim.x, 1, block_dim.z};
  auto block_dim_z = dim3{block_dim.x, block_dim.y, 1};

  __nF2FFFrequencyDomainUpdate<<<grid_dim_x, block_dim_x>>>(_data_xn);
  __nF2FFFrequencyDomainUpdate<<<grid_dim_x, block_dim_x>>>(_data_xp);
  __nF2FFFrequencyDomainUpdate<<<grid_dim_y, block_dim_y>>>(_data_yn);
  __nF2FFFrequencyDomainUpdate<<<grid_dim_y, block_dim_y>>>(_data_yp);
  __nF2FFFrequencyDomainUpdate<<<grid_dim_z, block_dim_z>>>(_data_zn);
  __nF2FFFrequencyDomainUpdate<<<grid_dim_z, block_dim_z>>>(_data_zp);
}

// explicit instantiation
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XN>;
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::XP>;
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YN>;
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::YP>;
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZN>;
template class NF2FFFrequencyDomainData<xfdtd::Axis::Direction::ZP>;

}  // namespace xfdtd::cuda
