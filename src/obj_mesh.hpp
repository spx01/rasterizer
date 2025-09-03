#pragma once

#include "mesh.hpp"

struct ObjMesh : mesh::Mesh {
  struct Repr;
  std::unique_ptr<Repr> repr_;

  ObjMesh(const char *path, const char *mtl_path = nullptr);
  ~ObjMesh() override;

  mesh::VertexData GetVertexData() const override;
  buf::Buf<uint> GetTrianglesPadded() const override;
};
