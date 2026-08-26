#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

double entropy(const std::vector<double>& counts) {
    double total = std::accumulate(counts.begin(), counts.end(), 0.0);
    if (total == 0.0) {
        return 0.0;
    }

    double result = 0.0;
    for (double count : counts) {
        if (count > 0.0) {
            const double probability = count / total;
            result -= probability * std::log2(probability);
        }
    }
    return result;
}

py::tuple best_continuous_split(
    py::array_t<double, py::array::c_style | py::array::forcecast> x,
    py::array_t<long long, py::array::c_style | py::array::forcecast> y,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    int n_classes) {
    const auto x_view = x.unchecked<1>();
    const auto y_view = y.unchecked<1>();
    const auto weight_view = weights.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(x_view.shape(0));

    std::vector<std::size_t> order;
    order.reserve(n);
    std::vector<double> total_counts(static_cast<std::size_t>(n_classes), 0.0);
    double total_weight = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isnan(x_view(i))) {
            order.push_back(i);
            total_counts[static_cast<std::size_t>(y_view(i))] += weight_view(i);
            total_weight += weight_view(i);
        }
    }

    if (order.size() < 2 || total_weight == 0.0) {
        return py::make_tuple(py::none(), 0.0);
    }

    std::sort(order.begin(), order.end(), [&x_view](std::size_t left, std::size_t right) {
        return x_view(left) < x_view(right);
    });

    const double base_entropy = entropy(total_counts);
    std::vector<double> left_counts(static_cast<std::size_t>(n_classes), 0.0);
    std::vector<double> right_counts = total_counts;
    std::size_t pointer = 0;
    double best_gain_ratio = -std::numeric_limits<double>::infinity();
    double best_threshold = 0.0;
    bool found = false;

    for (std::size_t i = 0; i + 1 < order.size(); ++i) {
        const double current = x_view(order[i]);
        const double next = x_view(order[i + 1]);
        if (current >= next) {
            continue;
        }

        const double threshold = (current + next) / 2.0;
        while (pointer < order.size() && x_view(order[pointer]) <= threshold) {
            const std::size_t row = order[pointer];
            const std::size_t label = static_cast<std::size_t>(y_view(row));
            left_counts[label] += weight_view(row);
            right_counts[label] -= weight_view(row);
            ++pointer;
        }

        const double left_weight = std::accumulate(left_counts.begin(), left_counts.end(), 0.0);
        const double right_weight = std::accumulate(right_counts.begin(), right_counts.end(), 0.0);
        if (left_weight == 0.0 || right_weight == 0.0) {
            continue;
        }

        const double after_entropy =
            (left_weight / total_weight) * entropy(left_counts) +
            (right_weight / total_weight) * entropy(right_counts);
        const double information_gain = base_entropy - after_entropy;
        const double left_probability = left_weight / total_weight;
        const double right_probability = right_weight / total_weight;
        const double split_info =
            -left_probability * std::log2(left_probability) -
            right_probability * std::log2(right_probability);
        if (split_info == 0.0) {
            continue;
        }

        const double gain_ratio = information_gain / split_info;
        if (gain_ratio > best_gain_ratio) {
            best_gain_ratio = gain_ratio;
            best_threshold = threshold;
            found = true;
        }
    }

    if (!found) {
        return py::make_tuple(py::none(), 0.0);
    }
    return py::make_tuple(best_threshold, std::max(best_gain_ratio, 0.0));
}

}  // namespace

PYBIND11_MODULE(_splitter_fast, module) {
    module.doc() = "Native numeric splitter for c5tree.";
    module.def("best_continuous_split", &best_continuous_split);
}
