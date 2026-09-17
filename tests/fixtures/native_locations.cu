#include "included_device.cuh"

// UTF-8 must not disturb Clang's byte-based source locations: naïve café 🚀
#define READ_PIXEL(pixel) ((pixel).value)

namespace image {
__global__ void tint(Pixel *pixels, Pixel amount) {
  int index = threadIdx.x;
  pixels[index].value = from_header(amount) + READ_PIXEL(amount);
}
} // namespace image
