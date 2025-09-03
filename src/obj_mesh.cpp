#include "obj_mesh.hpp"

#include "tiny_obj_loader.h"

struct ObjMesh::Repr {
  size_t num_vertices;
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<uint> triangles_padded;
  std::vector<float3> fetched_normals;
  std::vector<float2> fetched_uvs;
  std::vector<float4> fetched_colors;
  void FixupMissingAttribs() {
    std::vector<float> *attribs[] = {
        &attrib.colors,
    };
    for (auto *a : attribs) {
      if (a->empty()) {
        a->resize(3 * num_vertices, 0.F);
      } else if (a->size() != 3 * num_vertices) {
        throw std::runtime_error("inconsistent attribute data");
      }
    }
  }
};

ObjMesh::ObjMesh(const char *path, const char *mtl_path)
    : repr_(std::make_unique<Repr>()) {
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&repr_->attrib, &repr_->shapes, &materials, &warn,
                              &err, path, mtl_path, true);
  if (!warn.empty()) std::cerr << "(tiny_obj_loader) WARN: " << warn << "\n";
  if (!err.empty()) std::cerr << "(tiny_obj_loader) ERR: " << err << "\n";
  if (!ret) throw std::runtime_error("failed to load obj file");
  repr_->num_vertices = repr_->attrib.vertices.size() / 3;
  repr_->fetched_normals.resize(repr_->num_vertices);
  repr_->fetched_uvs.resize(repr_->num_vertices);
  for (auto &shape : repr_->shapes) {
    for (auto &face_count : shape.mesh.num_face_vertices) {
      if (face_count != 3) {
        throw std::runtime_error("only triangle meshes are supported");
      }
    }
    for (auto &idx : shape.mesh.indices) {
      auto [vi, ni, ti] = idx;
      repr_->fetched_normals[vi] =
          ni >= 0 ? make_float3(repr_->attrib.normals[3 * ni + 0],
                                repr_->attrib.normals[3 * ni + 1],
                                repr_->attrib.normals[3 * ni + 2])
                  : make_float3(0.F, 0.F, 0.F);
      repr_->fetched_uvs[vi] =
          ti >= 0 ? make_float2(repr_->attrib.texcoords[2 * ti + 0],
                                repr_->attrib.texcoords[2 * ti + 1])
                  : make_float2(0.F, 0.F);
      repr_->triangles_padded.push_back(vi);
      if (repr_->triangles_padded.size() % 4 == 3) {
        repr_->triangles_padded.push_back(0);
      }
    }
  }
  repr_->FixupMissingAttribs();
}

ObjMesh::~ObjMesh() {}

mesh::VertexData ObjMesh::GetVertexData() const {
  return mesh::VertexData{
      .positions = std::span(
          reinterpret_cast<const float3 *>(repr_->attrib.vertices.data()),
          repr_->num_vertices),
      .normals = std::span(repr_->fetched_normals),
      .texcoords = std::span(repr_->fetched_uvs),
      .colors = std::span(
          reinterpret_cast<const float3 *>(repr_->attrib.colors.data()),
          repr_->num_vertices)};
}

buf::Buf<uint> ObjMesh::GetTrianglesPadded() const {
  return std::span(repr_->triangles_padded);
}
