#pragma once

#include "pipeline.hpp"
#include "util.hpp"

namespace mesh {

struct VertexData {
  buf::Buf<float3> positions;
  buf::Buf<float3> normals;
  buf::Buf<float2> texcoords;
  buf::Buf<float3> colors;
};

enum UploadMode {
  kTrianglesPadded,
};

struct Mesh {
  virtual ~Mesh() = default;
  virtual VertexData GetVertexData() const = 0;
  virtual buf::Buf<uint> GetTrianglesPadded() const = 0;

  struct PipelineHandles {
    pipeline::Handle<pipeline::kVertexBuffer> vb;
    pipeline::Handle<pipeline::kIndexBuffer> ib;
  };
  [[nodiscard]] PipelineHandles UploadTo(UploadMode mode,
                                         pipeline::Pipeline &p);
};

}  // namespace mesh
