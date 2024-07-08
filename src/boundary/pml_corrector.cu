#include <cstdio>
#include <xfdtd_cuda/tensor.cuh>

#include "boundary/pml_corrector.cuh"
#include "boundary/pml_corrector_agency.cuh"

namespace xfdtd::cuda {

template <xfdtd::EMF::Attribute attribute, Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto correctPML(auto&& field, auto&& psi, const auto& coeff_a,
                                  const auto& coeff_b, const auto& dual_field,
                                  const auto& c_psi, Index is, Index ie,
                                  Index js, Index je, Index ks, Index ke,
                                  Index pml_global_start, Index pml_node_start,
                                  Index offset_c) {
  for (Index i = is; i < ie; ++i) {
    for (Index j = js; j < je; ++j) {
      for (Index k = ks; k < ke; ++k) {
        auto [a, b, c] = transform::xYZToABC<Index, xyz>(i, j, k);

        const auto global_c = c + offset_c;
        const auto coeff_a_v = coeff_a(global_c - pml_global_start);
        const auto coeff_b_v = coeff_b(global_c - pml_global_start);

        const auto layer_index = c - pml_node_start;
        auto [i_l, j_l, k_l] =
            transform::aBCToXYZ<Index, xyz>(a, b, layer_index);
        const auto c_psi_v = c_psi(i_l, j_l, k_l);
        auto&& psi_v = psi(i_l, j_l, k_l);

        auto&& f_v = field(i, j, k);

        const auto dual_f_p_v = dual_field(i, j, k);
        constexpr auto offset = attribute == xfdtd::EMF::Attribute::E ? -1 : 1;
        auto [i_dual, j_dual, k_dual] =
            transform::aBCToXYZ<Index, xyz>(a, b, c + offset);
        const auto dual_f_q_v = dual_field(i_dual, j_dual, k_dual);

        if constexpr (attribute == xfdtd::EMF::Attribute::E) {
          psi_v = coeff_b_v * psi_v + coeff_a_v * (dual_f_p_v - dual_f_q_v);
        } else {
          psi_v = coeff_b_v * psi_v + coeff_a_v * (dual_f_q_v - dual_f_p_v);
        }
        f_v += c_psi_v * psi_v;
      }
    }
  }
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto PMLCorrector<xyz>::correctE() -> void {
  constexpr auto attribute = xfdtd::EMF::Attribute::E;
  constexpr auto dual_attribute = xfdtd::EMF::dualAttribute(attribute);
  constexpr auto xyz_a = Axis::tangentialAAxis<xyz>();
  constexpr auto xyz_b = Axis::tangentialBAxis<xyz>();
  const auto pml_global_start = _pml_global_e_start;
  const auto pml_node_start = _pml_node_e_start;
  const auto task = this->task();
  if (!task.valid()) {
    return;
  }

  auto is = task.xRange().start();
  auto js = task.yRange().start();
  auto ks = task.zRange().start();

  auto main_axis_offset = 0;

  {
    // c == 0?
    auto [a, b, c] = transform::xYZToABC<Index, xyz>(
        _task.xRange().start(), _task.yRange().start(), _task.zRange().start());
    if (c == 0) {
      auto [a, b, c] = transform::xYZToABC<Index, xyz>(is, js, ks);
      auto [i, j, k] = transform::aBCToXYZ<Index, xyz>(a, b, c + 1);
      main_axis_offset = 1;
      is = i;
      js = j;
      ks = k;
    }
  }

  const auto offset_c = _offset_c;

  // static bool is_print = false;

  // if (!is_print) {
  //   std::printf(
  //       "PML blockIdx: %d, %d, %d, threadIdx: %d, %d, %d. Task: [%lu, %lu), "
  //       "[%lu, "
  //       "%lu), [%lu, %lu)\n",
  //       blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
  //       threadIdx.z, task.xRange().start(), task.xRange().end(),
  //       task.yRange().start(), task.yRange().end(), task.zRange().start(),
  //       task.zRange().end());
  //   is_print = true;
  //   return;
  // }
  // return;

  {
    auto&& field = _emf->field<attribute, xyz_a>();
    auto&& psi = this->psi<attribute, xyz_a>();
    const auto& coeff_a = coeffA<attribute>();
    const auto& coeff_b = coeffB<attribute>();
    const auto& dual_field = _emf->field<dual_attribute, xyz_b>();
    const auto& c_psi = cPsi<attribute, xyz_a>();
    // correct EA: [a_s, a_e), [b_s, b_e + 1), [c_s, c_e)
    auto [a, b, c] = transform::xYZToABC<Index, xyz>(
        task.xRange().end(), task.yRange().end(), task.zRange().end());

    // auto [ie, je, ke] =
    //     transform::aBCToXYZ<Index, xyz>(a, b + 1, c + main_axis_offset);
    Index ie = 0;
    Index je = 0;
    Index ke = 0;

    auto [ae, be, ce] = transform::xYZToABC<Index, xyz>(
        _task.xRange().end(), _task.yRange().end(), _task.zRange().end());

    if (c == ce) {
      auto [i, j, k] =
          transform::aBCToXYZ<Index, xyz>(a, b + 1, c + main_axis_offset);
      ie = i;
      je = j;
      ke = k;
    } else {
      auto [i, j, k] =
          transform::aBCToXYZ<Index, xyz>(a, b, c + main_axis_offset);
      ie = i;
      je = j;
      ke = k;
    }

    correctPML<attribute, xyz>(field, psi, coeff_a, coeff_b, dual_field, c_psi,
                               is, ie, js, je, ks, ke, pml_global_start,
                               pml_node_start, offset_c);
  }

  {
    auto&& field = _emf->field<attribute, xyz_b>();
    auto&& psi = this->psi<attribute, xyz_b>();
    const auto& coeff_a = coeffA<attribute>();
    const auto& coeff_b = coeffB<attribute>();
    const auto& dual_field = _emf->field<dual_attribute, xyz_a>();
    const auto& c_psi = cPsi<attribute, xyz_b>();
    // correct EB: [a_s, a_e + 1), [b_s, b_e), [c_s, c_e)
    auto [a, b, c] = transform::xYZToABC<Index, xyz>(
        task.xRange().end(), task.yRange().end(), task.zRange().end());
    // auto [ie, je, ke] =
    //     transform::aBCToXYZ<Index, xyz>(a + 1, b, c + main_axis_offset);

    Index ie = 0;
    Index je = 0;
    Index ke = 0;

    auto [ae, be, ce] = transform::xYZToABC<Index, xyz>(
        _task.xRange().end(), _task.yRange().end(), _task.zRange().end());

    if (c == ce) {
      auto [i, j, k] =
          transform::aBCToXYZ<Index, xyz>(a + 1, b, c + main_axis_offset);
      ie = i;
      je = j;
      ke = k;
    } else {
      auto [i, j, k] =
          transform::aBCToXYZ<Index, xyz>(a, b, c + main_axis_offset);
      ie = i;
      je = j;
      ke = k;
    }

    correctPML<attribute, xyz>(field, psi, coeff_a, coeff_b, dual_field, c_psi,
                               is, ie, js, je, ks, ke, pml_global_start,
                               pml_node_start, offset_c);
  }
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto PMLCorrector<xyz>::correctH() -> void {
  constexpr auto attribute = xfdtd::EMF::Attribute::H;
  constexpr auto dual_attribute = xfdtd::EMF::dualAttribute(attribute);
  constexpr auto xyz_a = Axis::tangentialAAxis<xyz>();
  constexpr auto xyz_b = Axis::tangentialBAxis<xyz>();
  const auto pml_global_start = _pml_global_h_start;
  const auto pml_node_start = _pml_node_h_start;
  const auto task = this->task();
  if (!task.valid()) {
    return;
  }

  const auto is = task.xRange().start();
  const auto js = task.yRange().start();
  const auto ks = task.zRange().start();

  const auto offset_c = _offset_c;

  {
    auto&& field = _emf->field<attribute, xyz_a>();
    auto&& psi = this->psi<attribute, xyz_a>();
    const auto& coeff_a = coeffA<attribute>();
    const auto& coeff_b = coeffB<attribute>();
    const auto& dual_field = _emf->field<dual_attribute, xyz_b>();
    const auto& c_psi = cPsi<attribute, xyz_a>();
    // correct HA: [a_s, a_e + 1), [b_s, b_e), [c_s, c_e)
    auto [a, b, c] = transform::xYZToABC<Index, xyz>(
        task.xRange().end(), task.yRange().end(), task.zRange().end());
    // auto [ie, je, ke] = transform::aBCToXYZ<Index, xyz>(a + 1, b, c);

    Index ie = 0;
    Index je = 0;
    Index ke = 0;

    auto [ae, be, ce] = transform::xYZToABC<Index, xyz>(
        _task.xRange().end(), _task.yRange().end(), _task.zRange().end());

    if (c == ce) {
      auto [i, j, k] = transform::aBCToXYZ<Index, xyz>(a + 1, b, c);
      ie = i;
      je = j;
      ke = k;
    } else {
      auto [i, j, k] = transform::aBCToXYZ<Index, xyz>(a, b, c);
      ie = i;
      je = j;
      ke = k;
    }

    correctPML<attribute, xyz>(field, psi, coeff_a, coeff_b, dual_field, c_psi,
                               is, ie, js, je, ks, ke, pml_global_start,
                               pml_node_start, offset_c);
  }

  {
    auto&& field = _emf->field<attribute, xyz_b>();
    auto&& psi = this->psi<attribute, xyz_b>();
    const auto& coeff_a = coeffA<attribute>();
    const auto& coeff_b = coeffB<attribute>();
    const auto& dual_field = _emf->field<dual_attribute, xyz_a>();
    const auto& c_psi = cPsi<attribute, xyz_b>();
    // correct HB: [a_s, a_e), [b_s, b_e + 1), [c_s, c_e)
    auto [a, b, c] = transform::xYZToABC<Index, xyz>(
        task.xRange().end(), task.yRange().end(), task.zRange().end());
    // auto [ie, je, ke] = transform::aBCToXYZ<Index, xyz>(a, b + 1, c);

    Index ie = 0;
    Index je = 0;
    Index ke = 0;

    auto [ae, be, ce] = transform::xYZToABC<Index, xyz>(
        _task.xRange().end(), _task.yRange().end(), _task.zRange().end());

    if (c == ce) {
      auto [i, j, k] = transform::aBCToXYZ<Index, xyz>(a, b + 1, c);
      ie = i;
      je = j;
      ke = k;
    } else {
      auto [i, j, k] = transform::aBCToXYZ<Index, xyz>(a, b, c);
      ie = i;
      je = j;
      ke = k;
    }

    correctPML<attribute, xyz>(field, psi, coeff_a, coeff_b, dual_field, c_psi,
                               is, ie, js, je, ks, ke, pml_global_start,
                               pml_node_start, offset_c);
  }
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_DEVICE auto PMLCorrector<xyz>::task() const -> IndexTask {
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

// Agency

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_GLOBAL void __PMLCorrectE(PMLCorrector<xyz>* pml_corrector) {
  pml_corrector->correctE();
}

template <xfdtd::Axis::XYZ xyz>
XFDTD_CUDA_GLOBAL void __PMLCorrectH(PMLCorrector<xyz>* pml_corrector) {
  pml_corrector->correctH();
}

template <xfdtd::Axis::XYZ xyz>
PMLCorrectorAgency<xyz>::PMLCorrectorAgency(PMLCorrector<xyz>* pml_corrector)
    : _pml_corrector{pml_corrector} {}

template <xfdtd::Axis::XYZ xyz>
auto PMLCorrectorAgency<xyz>::correctE(dim3 grid_size,
                                       dim3 block_size) -> void {
  __PMLCorrectE<xyz><<<grid_size, block_size>>>(_pml_corrector);
}

template <xfdtd::Axis::XYZ xyz>
auto PMLCorrectorAgency<xyz>::correctH(dim3 grid_size,
                                       dim3 block_size) -> void {
  __PMLCorrectH<xyz><<<grid_size, block_size>>>(_pml_corrector);
}

// explicit instantiation
template class PMLCorrector<Axis::XYZ::X>;
template class PMLCorrector<Axis::XYZ::Y>;
template class PMLCorrector<Axis::XYZ::Z>;

template class PMLCorrectorAgency<Axis::XYZ::X>;
template class PMLCorrectorAgency<Axis::XYZ::Y>;
template class PMLCorrectorAgency<Axis::XYZ::Z>;

}  // namespace xfdtd::cuda
