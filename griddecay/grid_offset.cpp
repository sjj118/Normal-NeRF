#include <torch/extension.h>

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

inline uint32_t powi(uint32_t base, uint32_t exponent)
{
	uint32_t result = 1;
	for (uint32_t i = 0; i < exponent; ++i)
	{
		result *= base;
	}

	return result;
}

template <typename T>
T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}

template <typename T>
T next_multiple(T val, T divisor)
{
	return div_round_up(val, divisor) * divisor;
}

inline float grid_scale(uint32_t level, float log2_per_level_scale, uint32_t base_resolution)
{
	// The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
	// than the number of cells. This is slightly different from the notation in the paper,
	// but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
	return exp2f(level * log2_per_level_scale) * base_resolution - 1.0f;
}

inline uint32_t grid_resolution(float scale)
{
	return (uint32_t)ceilf(scale) + 1;
}

std::vector<uint32_t> grid_offset(
	uint32_t grid_type,
	uint32_t n_levels,
	uint32_t log2_hashmap_size,
	uint32_t base_resolution,
	float per_level_scale,
	uint32_t n_features_per_level)
{
	std::vector<uint32_t> offset_table;
	uint32_t offset = 0;
	for (uint32_t i = 0; i < n_levels; ++i)
	{
		// Compute number of dense params required for the given level
		const uint32_t resolution = grid_resolution(grid_scale(i, std::log2(per_level_scale), base_resolution));

		uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
		uint32_t params_in_level = std::pow((float)resolution, 3) > (float)max_params ? max_params : powi(resolution, 3);

		// Make sure memory accesses will be aligned
		params_in_level = next_multiple(params_in_level, 8u);

		if (grid_type == 0)
		{
			// No-op
		}
		else if (grid_type == 1)
		{
			// If tiled grid needs fewer params than dense, then use fewer and tile.
			params_in_level = std::min(params_in_level, powi(base_resolution, 3));
		}
		else if (grid_type == 2)
		{
			// If hash table needs fewer params than dense, then use fewer and rely on the hash.
			params_in_level = std::min(params_in_level, (1u << log2_hashmap_size));
		}
		offset_table.push_back(offset);
		offset += params_in_level * n_features_per_level;
	}
	offset_table.push_back(offset);

	return offset_table;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("grid_offset", &grid_offset,
		  "grid offset");
}