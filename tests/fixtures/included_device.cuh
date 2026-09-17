#pragma once

struct Pixel {
  int value;
};

__device__ int from_header(Pixel pixel) { return pixel.value; }
