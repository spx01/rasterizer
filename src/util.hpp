#pragma once

#include <span>
#include <variant>
#include <vector>

template <typename T>
using CSpan = std::span<const T>;

namespace buf {

template <typename T>
using Buf = std::variant<std::vector<T>, std::span<const T>>;

template <typename T>
std::span<const T> AsSpan(const Buf<T>& foo) {
  return std::visit(
      [](auto const& data) -> std::span<const T> {
        using U = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<U, std::span<const T>>) {
          return data;
        } else {
          return std::span<const T>(data.data(), data.size());
        }
      },
      foo);
}

template <typename T>
std::vector<T> TakeOwnership(Buf<T>&& foo) {
  return std::visit(
      [](auto&& data) -> std::vector<T> {
        using U = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<U, std::vector<T>>) {
          return std::forward<decltype(data)>(data);
        } else {
          return std::vector<T>(data.begin(), data.end());
        }
      },
      std::move(foo));
}

}  // namespace buf
