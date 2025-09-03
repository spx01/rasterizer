#include "mesh.hpp"

#include <algorithm>

using namespace mesh;

static std::vector<float4> Widen(std::span<const float3> buf,
                                 const float fill = 1.F) {
  std::vector<float4> result;
  std::ranges::transform(buf, std::back_inserter(result), [fill](float3 in) {
    return make_float4(in.x, in.y, in.z, fill);
  });
  return result;
}

Mesh::PipelineHandles Mesh::UploadTo(UploadMode mode, pipeline::Pipeline& p) {
  switch (mode) {
    case kTrianglesPadded: {
      PipelineHandles handles;
      auto vb_buf = GetVertexData();
      auto positions_w = Widen(buf::AsSpan(vb_buf.positions));
      auto normals_w = Widen(buf::AsSpan(vb_buf.normals));
      auto colors_w = Widen(buf::AsSpan(vb_buf.colors));
      pipeline::VertexData<CSpan> vb_span{
          .positions = std::span(positions_w),
          .normals = std::span(normals_w),
          .texcoords = buf::AsSpan(vb_buf.texcoords),
          .colors = std::span(colors_w),
      };
      handles.vb = p.UploadVertexBuffer(vb_span);
      handles.ib = p.UploadIndexBuffer(buf::AsSpan(GetTrianglesPadded()));
      return handles;
    }
  }
  __builtin_unreachable();
}
