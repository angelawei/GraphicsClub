#ifndef AO_H
#define AO_H

#include <OpenGL/gl3.h>
#include <string>
#include <vector>
#include "glm/glm.hpp"
#include "tiny_obj_loader.h"

static const int32_t plane_vert_count = 6;
static float plane_vertices[] = {
     1.0f,  1.0f,
    -1.0f,  1.0f,
    -1.0f, -1.0f,

     1.0f,  1.0f,
    -1.0f, -1.0f,
     1.0f, -1.0f
};

class Camera {
public:
    Camera(glm::vec3 start_pos, glm::vec3 start_dir, glm::vec3 up) :
        pos(start_pos), up(up),
        start_dir(start_dir), start_left(glm::cross(start_dir,up)), rx(0), ry(0) { }

    // void adjust(float dx, float dy, float dz, float tx, float ty, float tz);
    void adjust(float dx);

    glm::mat4x4 get_view();

    float rx;
    float ry;
    glm::vec3 pos;
    glm::vec3 up;
    glm::vec3 start_left;
    glm::vec3 start_dir;
};

namespace mesh_attributes {
    enum {
        POSITION,
        NORMAL,
        TEXCOORD
    };
}

typedef struct {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<unsigned short> indices;
    std::string texname;
    glm::vec3 color;
} mesh_t;

typedef struct {
    unsigned int vertex_array;
    unsigned int vbo_indices;
    unsigned int num_indices;
    unsigned int vbo_vertices;
    unsigned int vbo_normals;
    unsigned int vbo_texcoords;
    unsigned int texture;
    std::string texname;
    glm::vec3 color;
} device_mesh_t;

typedef struct {
    GLuint fb;
    GLuint depthTexture;
    GLuint normalTexture;
    GLuint colorTexture;
} gbuffer_t;

typedef struct {
    int32_t is_running;
    int64_t frame;

    int32_t window_width;
    int32_t window_height;


    gbuffer_t gbuffer;
    GLuint gbuffer_program;
    GLint mesh_uniform_matrix_location;
    GLint mesh_uniform_mv_matrix_location;
    GLint mesh_uniform_normal_matrix_location;
    GLint mesh_uniform_color_location;
    GLint mesh_uniform_texture_assigned_color_location;
    GLint mesh_uniform_texture_sampler_location;
    GLuint ao_program;
    GLint ao_uniform_depth_texture_sampler_location;
    GLint ao_uniform_normal_texture_sampler_location;
    GLint ao_uniform_color_texture_sampler_location;
    GLint uniform_inverse_viewport_resolution_location;
    GLint uniform_inverse_projection_matrix_location;

    GLuint plane_vao;
    GLuint plane_vbo;

    // GLuint ao_fb;
    GLuint accum_texture;
    GLuint revealage_texture;

    GLuint crate_texture;

} ao_memory_t;

void ao_init(ao_memory_t* mem);
void ao_update_frame(ao_memory_t* mem);

typedef struct {
    void* contents;
    int32_t size;
} loaded_file_t;

loaded_file_t platform_load_file(const char* filename);

#endif

