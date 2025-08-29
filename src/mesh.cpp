#include "mesh.hpp"

#include <algorithm>

using namespace mesh;

static std::vector<float4> Widen(std::span<const float3> buf) {
  std::vector<float4> result;
  std::ranges::transform(buf, std::back_inserter(result), [](float3 in) {
    return float4(in.x, in.y, in.z, 0.F);
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
      pipeline::VertexData<CSpan> vb_span{
          .positions = std::span(positions_w),
          .normals = std::span(normals_w),
          .texcoords = buf::AsSpan(vb_buf.texcoords),
          .colors = buf::AsSpan(vb_buf.colors),
      };
      handles.vb = p.UploadVertexBuffer(vb_span);
      handles.ib = p.UploadIndexBuffer(buf::AsSpan(GetTrianglesPadded()));
      return handles;
    }
  }
  __builtin_unreachable();
}
