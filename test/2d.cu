#include <xfdtd/common/type_define.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <xfdtd_cuda/index_task.cuh>
#include <xfdtd_cuda/tensor.cuh>
#include <xfdtd_cuda/tensor_hd.cuh>

using Real = float;
using SizeType = std::size_t;

struct FDTD2DScatterProblem {
  using Array1D = xfdtd::cuda::Array1D<Real>;
  using Array2D = xfdtd::cuda::Tensor<Real, 2>;
  using Array3D = xfdtd::cuda::Tensor<Real, 3>;

  SizeType _nt{}; // time steps
  SizeType _nx{}; // x size
  SizeType _ny{}; // y size

  // update coefficients
  Array2D *_ceze{};
  Array2D *_cezhx{};
  Array2D *_cezhy{};
  Array2D *_chxh{};
  Array2D *_chxez{};
  Array2D *_chyh{};
  Array2D *_chyez{};

  // tfsf
  SizeType _tfsf_margin_x{};
  SizeType _tfsf_margin_y{};
  SizeType _tfsf_size_x{};
  SizeType _tfsf_size_y{};
  Array1D *_tfsf_projection_integrated_x{};
  Array1D *_tfsf_projection_integrated_y{};
  Array1D *_tfsf_projection_half_x{};
  Array1D *_tfsf_projection_half_y{};
  Array2D *_tfsf_ez{};
  Array2D *_tfsf_hx{};
  Array2D *_tfsf_hy{};
  Real _tfsf_cax{};
  Real _tfsf_cay{};
  Real _tfsf_cbx{};
  Real _tfsf_cby{};

  // field
  Array2D *_ez{};
  Array2D *_hx{};
  Array2D *_hy{};

  // monintor
  SizeType _movie_frame{};
  Array3D *_movie{};

  // cuda config
  dim3 _block{};
  dim3 _grid{};
};

XFDTD_CUDA_DUAL void runSimulateSingle(FDTD2DScatterProblem *problem) {
  const auto nt = problem->_nt;
  const auto nx = problem->_nx;
  const auto ny = problem->_ny;

  const auto &ceze = *problem->_ceze;
  const auto &cezhx = *problem->_cezhx;
  const auto &cezhy = *problem->_cezhy;
  const auto &chxh = *problem->_chxh;
  const auto &chxez = *problem->_chxez;
  const auto &chyh = *problem->_chyh;
  const auto &chyez = *problem->_chyez;

  const auto &tfsf_projection_integrated_x =
      *problem->_tfsf_projection_integrated_x;
  const auto &tfsf_projection_integrated_y =
      *problem->_tfsf_projection_integrated_y;
  const auto &tfsf_projection_half_x = *problem->_tfsf_projection_half_x;
  const auto &tfsf_projection_half_y = *problem->_tfsf_projection_half_y;

  struct {
    SizeType _x_start;
    SizeType _x_end;
    SizeType _y_start;
    SizeType _y_end;
  } tfsf_task{
      problem->_tfsf_margin_x, problem->_tfsf_margin_x + problem->_tfsf_size_x,
      problem->_tfsf_margin_y, problem->_tfsf_margin_y + problem->_tfsf_size_y};
  const auto &tfsf_ez = *problem->_tfsf_ez;
  const auto &tfsf_hx = *problem->_tfsf_hx;
  const auto &tfsf_hy = *problem->_tfsf_hy;
  const auto tfsf_cax = problem->_tfsf_cax;
  const auto tfsf_cay = problem->_tfsf_cay;
  const auto tfsf_cbx = problem->_tfsf_cbx;
  const auto tfsf_cby = problem->_tfsf_cby;

  auto &&ez = *problem->_ez;
  auto &&hx = *problem->_hx;
  auto &&hy = *problem->_hy;

  const auto &movie_frame = problem->_movie_frame;
  auto &&movie = *problem->_movie;

  for (SizeType t = 0; t < nt; ++t) {
    // update hx hy
    for (SizeType i = 0; i < nx; ++i) {
      for (SizeType j = 0; j < ny; ++j) {
        hx(i, j) =
            chxh(i, j) * hx(i, j) + chxez(i, j) * (ez(i, j + 1) - ez(i, j));
        hy(i, j) =
            chyh(i, j) * hy(i, j) + chyez(i, j) * (ez(i + 1, j) - ez(i, j));
      }
    }
    // correct hx hy
    // x backward
    {
      auto task = tfsf_task;
      task._y_end += 1;
      const auto is = task._x_start;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType j = js; j < je; ++j) {
        auto projection = tfsf_projection_integrated_x.at(is - is) +
                          tfsf_projection_integrated_y.at(j - js);
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                    tfsf_ez.at(t, index + 1) * weight;
        hy(is - 1, j) -= tfsf_cbx * ez_i;
      }
    }
    // x forward
    {
      auto task = tfsf_task;
      task._y_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType j = js; j < je; ++j) {
        auto projection = tfsf_projection_integrated_x.at(ie - is) +
                          tfsf_projection_integrated_y.at(j - js);
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                    tfsf_ez.at(t, index + 1) * weight;
        hy(ie, j) += tfsf_cbx * ez_i;
      }
    }
    // y backward
    {
      auto task = tfsf_task;
      task._x_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      for (SizeType i = is; i < ie; ++i) {
        auto projection = tfsf_projection_integrated_x.at(i - is) +
                          tfsf_projection_integrated_y.at(js - js);
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                    tfsf_ez.at(t, index + 1) * weight;
        hx(i, js - 1) += tfsf_cby * ez_i;
      }
    }
    // y forward
    {
      auto task = tfsf_task;
      task._x_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType i = is; i < ie; ++i) {
        auto projection = tfsf_projection_integrated_x.at(i - is) +
                          tfsf_projection_integrated_y.at(je - js);
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                    tfsf_ez.at(t, index + 1) * weight;
        hx(i, je) -= tfsf_cby * ez_i;
      }
    }

    // update ez
    for (SizeType i = 1; i < nx; ++i) {
      for (SizeType j = 1; j < ny; ++j) {
        ez(i, j) = ceze(i, j) * ez(i, j) +
                   cezhy(i, j) * (hy(i, j) - hy(i - 1, j)) +
                   cezhx(i, j) * (hx(i, j) - hx(i, j - 1));
      }
    }

    //
    {
      auto task = tfsf_task;
      task._y_end += 1;
      const auto is = task._x_start;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType j = js; j < je; ++j) {
        auto projection = tfsf_projection_half_x.at(is - 1 - is + 1) +
                          tfsf_projection_integrated_y.at(j - js) - 0.5;
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto hy_i = tfsf_hy.at(t, index) * (1 - weight) +
                    tfsf_hy.at(t, index + 1) * weight;
        ez(is, j) -= tfsf_cax * hy_i;
      }
    }
    {
      auto task = tfsf_task;
      task._y_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType j = js; j < je; ++j) {
        auto projection = tfsf_projection_half_x.at(ie - is + 1) +
                          tfsf_projection_integrated_y.at(j - js) - 0.5;
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto hy_i = tfsf_hy.at(t, index) * (1 - weight) +
                    tfsf_hy.at(t, index + 1) * weight;
        ez(ie, j) += tfsf_cax * hy_i;
      }
    }
    {
      auto task = tfsf_task;
      task._x_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      for (SizeType i = is; i < ie; ++i) {
        auto projection = tfsf_projection_integrated_x.at(i - is) +
                          tfsf_projection_half_y.at(js - 1 - js + 1) - 0.5;
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto hx_i = tfsf_hx.at(t, index) * (1 - weight) +
                    tfsf_hx.at(t, index + 1) * weight;
        ez(i, js) += tfsf_cay * hx_i;
      }
    }
    {
      auto task = tfsf_task;
      task._x_end += 1;
      const auto is = task._x_start;
      const auto ie = task._x_end;
      const auto js = task._y_start;
      const auto je = task._y_end;
      for (SizeType i = is; i < ie; ++i) {
        auto projection = tfsf_projection_integrated_x.at(i - js) +
                          tfsf_projection_half_y.at(je - js + 1) - 0.5;
        auto index = static_cast<SizeType>(projection);
        auto weight = projection - index;
        auto hx_i = tfsf_hx.at(t, index) * (1 - weight) +
                    tfsf_hx.at(t, index + 1) * weight;
        ez(i, je) -= tfsf_cay * hx_i;
      }
    }

    if (t % movie_frame == 0) {
      for (SizeType i = 0; i < nx + 1; ++i) {
        for (SizeType j = 0; j < ny + 1; ++j) {
          movie(t / movie_frame, i, j) = ez(i, j);
        }
      }
    }
  }
}

__global__ void run(FDTD2DScatterProblem *problem) {
  // printf("GridDim: {%d, %d, %d} BlockDim: {%d, %d, %d}\n", gridDim.x,
  // gridDim.y,
  //        gridDim.z, blockDim.x, blockDim.y, blockDim.z);
  const auto id_x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto id_y = blockIdx.y * blockDim.y + threadIdx.y;
  // printf("ThreadIdx: {%d, %d, %d} BlockIdx: {%d, %d, %d} id_x: %d id_y:
  // %d\n",
  //        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
  //        blockIdx.z, id_x, id_y);
  struct Range {
    SizeType _start;
    SizeType _end;
  };

  auto split_task = [](const decltype(id_x) id, const decltype(id_x) num_thread,
                       const SizeType problem_size) -> Range {
    auto quotient = problem_size / num_thread;
    auto remainder = problem_size % num_thread;
    SizeType start = 0;
    SizeType end = 0;
    if (id < remainder) {
      start = id * (quotient + 1);
      end = start + quotient + 1;
    } else {
      start = id * quotient + remainder;
      end = start + quotient;
    }

    return {start, end};
  };

  auto range_x = split_task(id_x, blockDim.x, problem->_nx);
  auto range_y = split_task(id_y, blockDim.y, problem->_ny);
  auto range_x_h = range_x;
  auto range_y_h = range_y;
  auto range_x_e_z = range_x;
  auto range_y_e_z = range_y;

  if (range_x_e_z._start == 0) {
    range_x_e_z._start = 1;
  }

  if (range_y_e_z._start == 0) {
    range_y_e_z._start = 1;
  }

  printf("ThreadIdx: {%d, %d, %d} BlockIdx: {%d, %d, %d} id_x: %d id_y: %d "
         "task_x: [%lu, %lu) task_y: [%lu, %lu)\n",
         threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
         blockIdx.z, id_x, id_y, range_x._start, range_x._end, range_y._start,
         range_y._end);

  const auto nt = problem->_nt;
  // const auto nx = problem->_nx;
  // const auto ny = problem->_ny;

  const auto &ceze = *problem->_ceze;
  const auto &cezhx = *problem->_cezhx;
  const auto &cezhy = *problem->_cezhy;
  const auto &chxh = *problem->_chxh;
  const auto &chxez = *problem->_chxez;
  const auto &chyh = *problem->_chyh;
  const auto &chyez = *problem->_chyez;

  const auto &tfsf_projection_integrated_x =
      *problem->_tfsf_projection_integrated_x;
  const auto &tfsf_projection_integrated_y =
      *problem->_tfsf_projection_integrated_y;
  const auto &tfsf_projection_half_x = *problem->_tfsf_projection_half_x;
  const auto &tfsf_projection_half_y = *problem->_tfsf_projection_half_y;

  struct {
    SizeType _x_start;
    SizeType _x_end;
    SizeType _y_start;
    SizeType _y_end;
  } tfsf_task{
      problem->_tfsf_margin_x, problem->_tfsf_margin_x + problem->_tfsf_size_x,
      problem->_tfsf_margin_y, problem->_tfsf_margin_y + problem->_tfsf_size_y};
  const auto &tfsf_ez = *problem->_tfsf_ez;
  const auto &tfsf_hx = *problem->_tfsf_hx;
  const auto &tfsf_hy = *problem->_tfsf_hy;
  const auto tfsf_cax = problem->_tfsf_cax;
  const auto tfsf_cay = problem->_tfsf_cay;
  const auto tfsf_cbx = problem->_tfsf_cbx;
  const auto tfsf_cby = problem->_tfsf_cby;

  auto &&ez = *problem->_ez;
  auto &&hx = *problem->_hx;
  auto &&hy = *problem->_hy;

  const auto &movie_frame = problem->_movie_frame;
  auto &&movie = *problem->_movie;

  for (SizeType t = 0; t < nt; ++t) {
    // update hx hy
    const auto is = range_x_h._start;
    const auto ie = range_x_h._end;
    const auto js = range_y_h._start;
    const auto je = range_y_h._end;
    for (SizeType i = is; i < ie; ++i) {
      for (SizeType j = js; j < je; ++j) {
        hx(i, j) =
            chxh(i, j) * hx(i, j) + chxez(i, j) * (ez(i, j + 1) - ez(i, j));
        hy(i, j) =
            chyh(i, j) * hy(i, j) + chyez(i, j) * (ez(i + 1, j) - ez(i, j));
      }
    }

    __syncthreads();

    // correct hx hy
    if (id_x == 0 && id_y == 0) {
      // x backward
      {
        auto task = tfsf_task;
        task._y_end += 1;
        const auto is = task._x_start;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType j = js; j < je; ++j) {
          auto projection = tfsf_projection_integrated_x.at(is - is) +
                            tfsf_projection_integrated_y.at(j - js);
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                      tfsf_ez.at(t, index + 1) * weight;
          hy(is - 1, j) -= tfsf_cbx * ez_i;
        }
      }
      // x forward
      {
        auto task = tfsf_task;
        task._y_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType j = js; j < je; ++j) {
          auto projection = tfsf_projection_integrated_x.at(ie - is) +
                            tfsf_projection_integrated_y.at(j - js);
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                      tfsf_ez.at(t, index + 1) * weight;
          hy(ie, j) += tfsf_cbx * ez_i;
        }
      }
      // y backward
      {
        auto task = tfsf_task;
        task._x_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        for (SizeType i = is; i < ie; ++i) {
          auto projection = tfsf_projection_integrated_x.at(i - is) +
                            tfsf_projection_integrated_y.at(js - js);
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                      tfsf_ez.at(t, index + 1) * weight;
          hx(i, js - 1) += tfsf_cby * ez_i;
        }
      }
      // y forward
      {
        auto task = tfsf_task;
        task._x_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType i = is; i < ie; ++i) {
          auto projection = tfsf_projection_integrated_x.at(i - is) +
                            tfsf_projection_integrated_y.at(je - js);
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto ez_i = tfsf_ez.at(t, index) * (1 - weight) +
                      tfsf_ez.at(t, index + 1) * weight;
          hx(i, je) -= tfsf_cby * ez_i;
        }
      }
    }

    __syncthreads();

    // update ez
    const auto is_e_z = range_x_e_z._start;
    const auto ie_e_z = range_x_e_z._end;
    const auto js_e_z = range_y_e_z._start;
    const auto je_e_z = range_y_e_z._end;
    for (SizeType i = is_e_z; i < ie_e_z; ++i) {
      for (SizeType j = js_e_z; j < je_e_z; ++j) {
        ez(i, j) = ceze(i, j) * ez(i, j) +
                   cezhy(i, j) * (hy(i, j) - hy(i - 1, j)) +
                   cezhx(i, j) * (hx(i, j) - hx(i, j - 1));
      }
    }

    __syncthreads();

    //
    if (id_x == 0 && id_y == 0) {
      {
        auto task = tfsf_task;
        task._y_end += 1;
        const auto is = task._x_start;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType j = js; j < je; ++j) {
          auto projection = tfsf_projection_half_x.at(is - 1 - is + 1) +
                            tfsf_projection_integrated_y.at(j - js) - 0.5;
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto hy_i = tfsf_hy.at(t, index) * (1 - weight) +
                      tfsf_hy.at(t, index + 1) * weight;
          ez(is, j) -= tfsf_cax * hy_i;
        }
      }
      {
        auto task = tfsf_task;
        task._y_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType j = js; j < je; ++j) {
          auto projection = tfsf_projection_half_x.at(ie - is + 1) +
                            tfsf_projection_integrated_y.at(j - js) - 0.5;
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto hy_i = tfsf_hy.at(t, index) * (1 - weight) +
                      tfsf_hy.at(t, index + 1) * weight;
          ez(ie, j) += tfsf_cax * hy_i;
        }
      }
      {
        auto task = tfsf_task;
        task._x_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        for (SizeType i = is; i < ie; ++i) {
          auto projection = tfsf_projection_integrated_x.at(i - is) +
                            tfsf_projection_half_y.at(js - 1 - js + 1) - 0.5;
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto hx_i = tfsf_hx.at(t, index) * (1 - weight) +
                      tfsf_hx.at(t, index + 1) * weight;
          ez(i, js) += tfsf_cay * hx_i;
        }
      }
      {
        auto task = tfsf_task;
        task._x_end += 1;
        const auto is = task._x_start;
        const auto ie = task._x_end;
        const auto js = task._y_start;
        const auto je = task._y_end;
        for (SizeType i = is; i < ie; ++i) {
          auto projection = tfsf_projection_integrated_x.at(i - js) +
                            tfsf_projection_half_y.at(je - js + 1) - 0.5;
          auto index = static_cast<SizeType>(projection);
          auto weight = projection - index;
          auto hx_i = tfsf_hx.at(t, index) * (1 - weight) +
                      tfsf_hx.at(t, index + 1) * weight;
          ez(i, je) -= tfsf_cay * hx_i;
        }
      }
    }

    __syncthreads();
    if (t % movie_frame == 0) {
      const auto is = range_x._start;
      const auto ie = range_x._end;
      const auto js = range_y._start;
      const auto je = range_y._end;

      for (SizeType i = is; i < ie; ++i) {
        for (SizeType j = js; j < je; ++j) {
          movie(t / movie_frame, i, j) = ez(i, j);
        }
      }
    }
  }
}

void FDTD2d() {
  const SizeType nx = 180;
  const SizeType ny = 180;
  const SizeType nt = 800;
  constexpr double center_frequency{12e9};
  constexpr double max_frequency{20e9};
  constexpr double min_lambda{3e8 / max_frequency};
  constexpr double dx{min_lambda / 20};
  constexpr double dy{dx};
  constexpr double tau{1.7 / (max_frequency - center_frequency)};
  constexpr double t_0{0.8 * tau};
  constexpr double dt{dx / (1.414 * 3e8)};
  constexpr Real cylinder_radius{0.03};
  // constexpr SizeType cylinder_radius_in_cells{
  //     static_cast<SizeType>(cylinder_radius / dx)};

  constexpr Real EPSILON_0 = 8.854187817e-12;
  constexpr Real MU_0 = 4 * 3.14159265 * 1e-7;

  // construct Domain
  auto coord_x = xfdtd::cuda::Tensor<Real, 1>({nx + 1});
  auto coord_y = xfdtd::cuda::Tensor<Real, 1>({ny + 1});
  auto min_x = -dx * (nx / 2.0);
  auto min_y = -dy * (ny / 2.0);
  for (SizeType i = 0; i < nx + 1; ++i) {
    coord_x(i) = min_x + i * dx;
  }
  for (SizeType i = 0; i < ny + 1; ++i) {
    coord_y(i) = min_y + i * dy;
  }
  printf("Domain Size: %lu x %lu\n", nx, ny);
  printf("dx = %f, dy = %f\n", dx, dy);
  printf("min_x = %f, max_x = %f\n", coord_x(0), coord_x(nx));
  printf("min_y = %f, max_y = %f\n", coord_y(0), coord_y(ny));

  // construct Material Space
  auto epsilon_z_xtensor = xfdtd::Array2D<Real>({nx + 1, ny + 1});
  epsilon_z_xtensor.fill(1.0 * EPSILON_0);

  auto mu_x_xtensor = xfdtd::Array2D<Real>({nx, ny + 1});
  mu_x_xtensor.fill(1.0 * MU_0);

  auto mu_y_xtensor = xfdtd::Array2D<Real>({nx + 1, ny});
  mu_y_xtensor.fill(1.0 * MU_0);

  auto sigma_e_z_xtensor = xfdtd::Array2D<Real>({nx + 1, ny + 1});
  sigma_e_z_xtensor.fill(0.0);

  auto sigma_m_x_xtensor = xfdtd::Array2D<Real>({nx, ny + 1});
  sigma_m_x_xtensor.fill(0.0);

  auto sigma_m_y_xtensor = xfdtd::Array2D<Real>({nx + 1, ny});
  sigma_m_y_xtensor.fill(0.0);

  // construct Cylinder
  constexpr Real cylinder_center_x{0.0};
  constexpr Real cylinder_center_y{0.0};
  constexpr Real cylinder_sigma_e = 1e10;
  auto cylinder_shape_func = [cylinder_center_x, cylinder_center_y,
                              cylinder_radius](Real x, Real y,
                                               Real eps = 1e-6) {
    auto dis = (x - cylinder_center_x) * (x - cylinder_center_x) +
               (y - cylinder_center_y) * (y - cylinder_center_y);

    return dis < (cylinder_radius * cylinder_radius + eps);
  };
  // correct the sigma_e in the cylinder
  {
    auto &&sigma = sigma_e_z_xtensor;
    for (SizeType i = 0; i < sigma_e_z_xtensor.shape()[0]; ++i) {
      for (SizeType j = 0; j < sigma_e_z_xtensor.shape()[1]; ++j) {
        if (cylinder_shape_func(coord_x(i), coord_y(j))) {
          sigma(i, j) = cylinder_sigma_e;
        }
      }
    }
  }

  // output material space
  {
    // auto&& epsilon = epsilon_z_xtensor;
    auto &&sigma_e = sigma_e_z_xtensor;
    auto outdir = std::filesystem::path("tmp");
    if (!std::filesystem::exists(outdir)) {
      std::filesystem::create_directory(outdir);
    }
    printf("Output Dir: %s\n", std::filesystem::absolute(outdir).c_str());

    auto sigma_e_out = outdir / "sigma_e_z.dat";
    auto coord_x_out = outdir / "coord_x.dat";
    auto coord_y_out = outdir / "coord_y.dat";
    std::fstream sigma_e_out_file(sigma_e_out, std::ios::out);
    std::fstream coord_x_out_file(coord_x_out, std::ios::out);
    std::fstream coord_y_out_file(coord_y_out, std::ios::out);
    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        sigma_e_out_file << sigma_e(i, j) << " ";
      }
      sigma_e_out_file << "\n";
    }
    for (SizeType i = 0; i < nx + 1; ++i) {
      coord_x_out_file << coord_x(i) << " ";
    }
    for (SizeType i = 0; i < ny + 1; ++i) {
      coord_y_out_file << coord_y(i) << " ";
    }
  }

  // coefficients
  auto ceze_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny + 1}));
  auto cezhx_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny + 1}));
  auto cezhy_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny + 1}));
  {
    auto &&ceze = ceze_hd.tensor();
    auto &&cezhx = cezhx_hd.tensor();
    auto &&cezhy = cezhy_hd.tensor();
    auto &&epsilon = epsilon_z_xtensor;
    auto &&sigma_e = sigma_e_z_xtensor;

    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        ceze(i, j) = (2 * epsilon(i, j) - dt * sigma_e(i, j)) /
                     (2 * epsilon(i, j) + dt * sigma_e(i, j));
        cezhx(i, j) = -(2 * dt / dy) / (2 * epsilon(i, j) + dt * sigma_e(i, j));
        cezhy(i, j) = (2 * dt / dx) / (2 * epsilon(i, j) + dt * sigma_e(i, j));
      }
    }
  }
  auto chxh_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx, ny + 1}));
  auto chxez_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx, ny + 1}));
  {
    auto &&chxh = chxh_hd.tensor();
    auto &&chxez = chxez_hd.tensor();
    auto &&mu = mu_x_xtensor;
    auto &&sigma = sigma_m_x_xtensor;

    for (SizeType i = 0; i < nx; ++i) {
      for (SizeType j = 0; j < ny + 1; ++j) {
        chxh(i, j) = (2 * mu(i, j) - dt * sigma(i, j)) /
                     (2 * mu(i, j) + dt * sigma(i, j));
        chxez(i, j) = -(2 * dt / dy) / (2 * mu(i, j) + dt * sigma(i, j));
      }
    }
  }
  auto chyh_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny}));
  auto chyez_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny}));
  {
    auto &&chyh = chyh_hd.tensor();
    auto &&chyez = chyez_hd.tensor();
    auto &&mu = mu_y_xtensor;
    auto &&sigma = sigma_m_y_xtensor;

    for (SizeType i = 0; i < nx + 1; ++i) {
      for (SizeType j = 0; j < ny; ++j) {
        chyh(i, j) = (2 * mu(i, j) - dt * sigma(i, j)) /
                     (2 * mu(i, j) + dt * sigma(i, j));
        chyez(i, j) = (2 * dt / dx) / (2 * mu(i, j) + dt * sigma(i, j));
      }
    }
  }

  // source
  auto source_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array1D<Real>, 1>(
      xfdtd::Array1D<Real>::from_shape({nt}));
  {
    auto &src = source_hd.tensor();
    for (SizeType i = 0; i < nt; ++i) {
      src(i) = std::exp(-0.5 * std::pow((t_0 - (i + 0.5) * dt) / tau, 2)) *
               std::cos(2 * 3.14159265 * center_frequency * (i + 0.5) * dt);
    }
  }

  // TF/SF
  struct {
    Real _theta{3.14159265 / 2};
    Real _phi{3 * 3.14159265 / 5};
    Real _psi{0};
  } tfsf_plane_wave;
  constexpr SizeType tfsf_margin_x = 30;
  constexpr SizeType tfsf_margin_y = 30;
  const auto tfsf_size_x = nx - 2 * tfsf_margin_x;
  const auto tfsf_size_y = ny - 2 * tfsf_margin_y;
  // determine the TF/SF start point
  struct {
    Real _x{};
    Real _y{};
    Real _z{};
  } tfsf_k;
  tfsf_k._x = std::sin(tfsf_plane_wave._theta) * std::cos(tfsf_plane_wave._phi);
  tfsf_k._y = std::sin(tfsf_plane_wave._theta) * std::sin(tfsf_plane_wave._phi);
  tfsf_k._z = std::cos(tfsf_plane_wave._theta);
  if (std::abs(tfsf_k._x) < 1e-6) {
    tfsf_k._x = 0;
  }
  if (std::abs(tfsf_k._y) < 1e-6) {
    tfsf_k._y = 0;
  }
  if (std::abs(tfsf_k._z) < 1e-6) {
    tfsf_k._z = 0;
  }
  SizeType tfsf_start_x_index = tfsf_margin_x;
  SizeType tfsf_start_y_index = tfsf_margin_y;
  if (0 <= tfsf_k._x && 0 <= tfsf_k._y) {
    tfsf_start_x_index = tfsf_margin_x;
    tfsf_start_y_index = tfsf_margin_y;
  } else if (0 <= tfsf_k._x && tfsf_k._y < 0) {
    tfsf_start_x_index = tfsf_margin_x;
    tfsf_start_y_index = ny - tfsf_margin_y;
  } else if (tfsf_k._x < 0 && 0 <= tfsf_k._y) {
    tfsf_start_x_index = nx - tfsf_margin_x;
    tfsf_start_y_index = tfsf_margin_y;
  } else if (tfsf_k._x < 0 && tfsf_k._y < 0) {
    tfsf_start_x_index = nx - tfsf_margin_x;
    tfsf_start_y_index = ny - tfsf_margin_y;
  }
  // Use Aux Field
  const Real tfsf_ratio =
      1.0 / (std::sqrt(std::pow(std::sin(tfsf_plane_wave._theta), 4) *
                           (std::pow(std::cos(tfsf_plane_wave._phi), 4) +
                            std::pow(std::sin(tfsf_plane_wave._phi), 4)) +
                       std::pow(std::cos(tfsf_plane_wave._theta), 4)));
  const SizeType aux_size_arr =
      static_cast<SizeType>(
          std::ceil(tfsf_ratio * std::sqrt(tfsf_size_x * tfsf_size_x +
                                           tfsf_size_y * tfsf_size_y))) +
      4 + 1;
  auto extra_dis_x = 2 * tfsf_k._x / tfsf_ratio;
  auto extra_dis_y = 2 * tfsf_k._y / tfsf_ratio;
  auto incident_point_x = tfsf_start_x_index - extra_dis_x;
  auto incident_point_y = tfsf_start_y_index - extra_dis_y;
  // Calculate projection
  auto projection_integrated_func = [](auto &&res, Real incident_point_componet,
                                       SizeType start_index, Real k_component,
                                       Real ratio) {
    for (SizeType i = 0; i < res.tensor().size(); ++i) {
      res.tensor().at(i) =
          (i + start_index - incident_point_componet) * k_component * ratio;
    }
  };
  auto projection_half_func = [](auto &&res, Real incident_point_componet,
                                 SizeType start_index, Real k_component,
                                 Real ratio) {
    for (SizeType i = 0; i < res.tensor().size(); ++i) {
      res.tensor().at(i) = (i + start_index - incident_point_componet - 0.5) *
                           k_component * ratio;
    }
  };
  auto tfsf_projection_integrated_x_hd =
      xfdtd::cuda::TensorHDWrapped<xfdtd::Array1D<Real>, 1>(
          xfdtd::Array1D<Real>::from_shape({aux_size_arr + 1}));
  auto tfsf_projection_integrated_y_hd =
      xfdtd::cuda::TensorHDWrapped<xfdtd::Array1D<Real>, 1>(
          xfdtd::Array1D<Real>::from_shape({aux_size_arr + 1}));
  auto tfsf_projection_half_x_hd =
      xfdtd::cuda::TensorHDWrapped<xfdtd::Array1D<Real>, 1>(
          xfdtd::Array1D<Real>::from_shape({aux_size_arr + 2}));
  auto tfsf_projection_half_y_hd =
      xfdtd::cuda::TensorHDWrapped<xfdtd::Array1D<Real>, 1>(
          xfdtd::Array1D<Real>::from_shape({aux_size_arr + 2}));
  projection_integrated_func(tfsf_projection_integrated_x_hd, incident_point_x,
                             tfsf_margin_x, tfsf_k._x, tfsf_ratio);
  projection_integrated_func(tfsf_projection_integrated_y_hd, incident_point_y,
                             tfsf_margin_y, tfsf_k._y, tfsf_ratio);
  projection_half_func(tfsf_projection_half_x_hd, incident_point_x,
                       tfsf_margin_x, tfsf_k._x, tfsf_ratio);
  projection_half_func(tfsf_projection_half_y_hd, incident_point_y,
                       tfsf_margin_y, tfsf_k._y, tfsf_ratio);
  // incident arr
  auto tfsf_e_i_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nt, aux_size_arr}));
  auto tfsf_h_i_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nt, aux_size_arr - 1}));
  // preprocess incident field
  auto fdtd1d_preprocess_incident_field_func = [](auto &&e, auto &&h,
                                                  const Real dt, const Real dl,
                                                  const Real epsilon,
                                                  const auto &source) {
    const auto nt = h.shape()[0];
    const auto nl = h.shape()[1];

    auto ceie = 1.0;
    auto ceih = -dt / (epsilon * dl);
    auto chih = 1.0;
    auto chie = -dt / (MU_0 * dl);

    // abc in end
    const auto c = sqrt(1 / (epsilon * MU_0));
    const auto abc_coeff_0 = (c * dt - dl) / (c * dt + dl);
    const auto abc_coeff_1 = 2 * dl / (c * dt + dl);

    // run
    e(0, 0) = source(0);
    for (SizeType l = 1; l < nl; ++l) {
      h(0, l) = chih * h(0, l) + chie * (e(0, l) - e(0, l - 1));
    }

    for (SizeType t = 1; t < nt; ++t) {
      for (SizeType l = 1; l < nl; ++l) {
        e(t, 0) = source(t);
        e(t, l) = ceie * e(t - 1, l) + ceih * (h(t - 1, l) - h(t - 1, l - 1));
      }

      // abc
      Real e_l_prev_t_prev_prev = (t < 2) ? 0.0 : e(t - 2, nl - 1);
      Real e_l_t_prev_prev = (t < 2) ? 0.0 : e(t - 2, nl);
      e(t, nl) = -e_l_prev_t_prev_prev +
                 abc_coeff_0 * (e(t, nl - 1) + e_l_t_prev_prev) +
                 abc_coeff_1 * (e(t - 1, nl) + e(t - 1, nl - 1));

      for (SizeType l = 0; l < nl; ++l) {
        h(t, l) = chih * h(t - 1, l) + chie * (e(t, l + 1) - e(t, l));
      }
    }
  };
  fdtd1d_preprocess_incident_field_func(
      tfsf_e_i_hd.tensor(), tfsf_h_i_hd.tensor(), dt, dx / tfsf_ratio,
      EPSILON_0, source_hd.tensor());
  // output tfsf
  {
    auto tfsf_e_i = tfsf_e_i_hd.tensor();
    auto outdir = std::filesystem::path("tmp");
    if (!std::filesystem::exists(outdir)) {
      std::filesystem::create_directory(outdir);
    }
    printf("Output Dir: %s\n", std::filesystem::absolute(outdir).c_str());
    std::fstream tfsf_e_i_out(outdir / "tfsf_e_i.dat", std::ios::out);
    for (SizeType i = 0; i < tfsf_e_i.shape()[0]; ++i) {
      for (SizeType j = 0; j < tfsf_e_i.shape()[1]; ++j) {
        tfsf_e_i_out << tfsf_e_i(i, j) << " ";
      }
      tfsf_e_i_out << "\n";
    }
    tfsf_e_i_out.close();
  }
  // tfsf k_e
  auto cos_theta = std::cos(tfsf_plane_wave._theta);
  auto sin_theta = std::sin(tfsf_plane_wave._theta);
  auto cos_phi = std::cos(tfsf_plane_wave._phi);
  auto sin_phi = std::sin(tfsf_plane_wave._phi);
  auto cos_psi = std::cos(tfsf_plane_wave._psi);
  auto sin_psi = std::sin(tfsf_plane_wave._psi);
  auto k_e_x = sin_psi;
  auto k_e_y = cos_psi;
  // auto k_e_z = 0.0;
  auto transform_e_x = k_e_x * (-sin_phi) + k_e_y * (cos_phi * cos_theta);
  auto transform_e_y = k_e_x * cos_phi + k_e_y * (sin_phi * cos_theta);
  auto transform_e_z = k_e_y * (-sin_theta);
  // transform h = k \times transform_e
  auto transform_h_x = tfsf_k._y * transform_e_z - tfsf_k._z * transform_e_y;
  auto transform_h_y = tfsf_k._z * transform_e_x - tfsf_k._x * transform_e_z;
  auto transform_h_z = tfsf_k._x * transform_e_y - tfsf_k._y * transform_e_x;
  // ez hx hy
  auto tfsf_ez_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nt, aux_size_arr}));
  auto tfsf_hx_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nt, aux_size_arr - 1}));
  auto tfsf_hy_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nt, aux_size_arr - 1}));

  // preprocess tfsf field
  for (SizeType t = 0; t < nt; ++t) {
    for (SizeType l = 0; l < aux_size_arr; ++l) {
      tfsf_ez_hd.tensor().at(t, l) =
          tfsf_e_i_hd.tensor().at(t, l) * transform_e_z;
    }
    for (SizeType l = 0; l < aux_size_arr - 1; ++l) {
      tfsf_hx_hd.tensor().at(t, l) =
          tfsf_h_i_hd.tensor().at(t, l) * transform_h_x;
      tfsf_hy_hd.tensor().at(t, l) =
          tfsf_h_i_hd.tensor().at(t, l) * transform_h_y;
    }
  }
  const Real tfsf_cax = dt / (EPSILON_0 * dx);
  const Real tfsf_cbx = dt / (MU_0 * dx);
  const Real tfsf_cay = dt / (EPSILON_0 * dy);
  const Real tfsf_cby = dt / (MU_0 * dy);

  // run simulation
  auto ez_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny + 1}));
  auto hx_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx, ny + 1}));
  auto hy_hd = xfdtd::cuda::TensorHDWrapped<xfdtd::Array2D<Real>, 2>(
      xfdtd::Array2D<Real>::from_shape({nx + 1, ny}));

  SizeType gif_step = 10;
  auto movie = xfdtd::cuda::TensorHDWrapped<xfdtd::Array3D<Real>, 3>(
      xfdtd::Array3D<Real>::from_shape({nt / gif_step, nx + 1, ny + 1}));

  auto problem = FDTD2DScatterProblem{
      nt,          nx,          ny,      nullptr,  nullptr,       nullptr,
      nullptr,     nullptr,     nullptr, nullptr,  tfsf_margin_x, tfsf_margin_y,
      tfsf_size_x, tfsf_size_y, nullptr, nullptr,  nullptr,       nullptr,
      nullptr,     nullptr,     nullptr, tfsf_cax, tfsf_cay,      tfsf_cbx,
      tfsf_cby,    nullptr,     nullptr, nullptr,  gif_step,      nullptr};
  FDTD2DScatterProblem *problem_decive = nullptr;
  {
    auto err = cudaMalloc(&problem_decive, sizeof(FDTD2DScatterProblem));
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      return;
    }
  }
  ceze_hd.copyHostToDevice();
  cezhx_hd.copyHostToDevice();
  cezhy_hd.copyHostToDevice();
  chxh_hd.copyHostToDevice();
  chxez_hd.copyHostToDevice();
  chyh_hd.copyHostToDevice();
  chyez_hd.copyHostToDevice();
  tfsf_projection_integrated_x_hd.copyHostToDevice();
  tfsf_projection_integrated_y_hd.copyHostToDevice();
  tfsf_projection_half_x_hd.copyHostToDevice();
  tfsf_projection_half_y_hd.copyHostToDevice();
  tfsf_ez_hd.copyHostToDevice();
  tfsf_hx_hd.copyHostToDevice();
  tfsf_hy_hd.copyHostToDevice();
  ez_hd.copyHostToDevice();
  hx_hd.copyHostToDevice();
  hy_hd.copyHostToDevice();
  movie.copyHostToDevice();
  problem._ceze = ceze_hd.device();
  problem._cezhx = cezhx_hd.device();
  problem._cezhy = cezhy_hd.device();
  problem._chxh = chxh_hd.device();
  problem._chxez = chxez_hd.device();
  problem._chyh = chyh_hd.device();
  problem._chyez = chyez_hd.device();
  problem._tfsf_projection_integrated_x =
      tfsf_projection_integrated_x_hd.device();
  problem._tfsf_projection_integrated_y =
      tfsf_projection_integrated_y_hd.device();
  problem._tfsf_projection_half_x = tfsf_projection_half_x_hd.device();
  problem._tfsf_projection_half_y = tfsf_projection_half_y_hd.device();
  problem._tfsf_ez = tfsf_ez_hd.device();
  problem._tfsf_hx = tfsf_hx_hd.device();
  problem._tfsf_hy = tfsf_hy_hd.device();
  problem._ez = ez_hd.device();
  problem._hx = hx_hd.device();
  problem._hy = hy_hd.device();
  problem._movie = movie.device();
  problem._block = dim3{16, 16};
  problem._grid = dim3{1, 1};
  {
    auto err = cudaMemcpy(problem_decive, &problem,
                          sizeof(FDTD2DScatterProblem), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      return;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  const auto block_nx = problem._block.x;
  const auto block_ny = problem._block.y;
  const auto grid_nx = problem._grid.x;
  const auto grid_ny = problem._grid.y;
  run<<<dim3(grid_nx, grid_ny), dim3(block_nx, block_ny)>>>(problem_decive);
  cudaDeviceSynchronize();
  {
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Error: %s\n", cudaGetErrorString(err));
      return;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Duration: %ld ms\n", duration.count());
  // copy back
  movie.copyDeviceToHost();

  // Run FDTD 2D in host
  // runSimulate(&problem_host);

  {
    auto outdir = std::filesystem::path("tmp") / "ez";
    if (!std::filesystem::exists(outdir)) {
      std::filesystem::create_directory(outdir);
    }
    printf("Output Dir: %s\n", std::filesystem::absolute(outdir).c_str());
    for (SizeType t = 0; t < nt / gif_step; ++t) {
      std::stringstream ss;
      ss << std::setw(4) << std::setfill('0') << t;
      auto out = std::ofstream(outdir / (ss.str() + ".dat"));
      for (SizeType i = 0; i < nx + 1; ++i) {
        for (SizeType j = 0; j < ny + 1; ++j) {
          out << movie.tensor().at(t, i, j) << " ";
        }
        out << "\n";
      }
    }
  }
  return;

  // constexpr auto MB = 1024 * 1024;
  // constexpr auto float_size = sizeof(Real);
  // constexpr auto movie_nt = nt / 10;
  // constexpr auto movie_size = nx * ny * movie_nt * float_size / MB;
}

int main() {
  FDTD2d();
  return 0;
}
