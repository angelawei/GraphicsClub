#include "ao.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_TGA
#include "stb_image.h"

#define STATUS_CASE(enum) case (enum): return #enum
static const char* get_gl_error_string(GLenum err) {
  switch(err) {
    STATUS_CASE(GL_NO_ERROR);
    STATUS_CASE(GL_INVALID_ENUM);
    STATUS_CASE(GL_INVALID_OPERATION);
    STATUS_CASE(GL_INVALID_FRAMEBUFFER_OPERATION);
    STATUS_CASE(GL_OUT_OF_MEMORY);
    STATUS_CASE(GL_FRAMEBUFFER_COMPLETE);
    STATUS_CASE(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
    STATUS_CASE(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
    STATUS_CASE(GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE);
    STATUS_CASE(GL_FRAMEBUFFER_UNSUPPORTED);
  }
  return NULL;
}

#define COLORED 0

static GLuint gl_create_shader_program(const char* vertex_shader_src, const char* geometry_shader_src, const char* fragment_shader_src)
{
    GLuint program = glCreateProgram();

    GLchar log[500];
    GLsizei log_length;

  if (vertex_shader_src) {
    GLint did_vertex_shader_compile = GL_FALSE;
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_src, NULL);
    glCompileShader(vertex_shader);

    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &did_vertex_shader_compile);
    if (did_vertex_shader_compile == GL_FALSE) {
      glGetShaderInfoLog(vertex_shader, sizeof(log), &log_length, log);
      printf("vertex shader log:\n%s\n", log);
      glDeleteShader(vertex_shader);
      glDeleteProgram(program);
      return (0);
    }

    glAttachShader(program, vertex_shader);
    glDeleteShader(vertex_shader);
  }

  if (geometry_shader_src) {
    GLint did_geometry_shader_compile = GL_FALSE;
    GLuint geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometry_shader, 1, &geometry_shader_src, NULL);
    glCompileShader(geometry_shader);


    glGetShaderiv(geometry_shader, GL_COMPILE_STATUS, &did_geometry_shader_compile);
    if (did_geometry_shader_compile == GL_FALSE) {
      glGetShaderInfoLog(geometry_shader, sizeof(log), &log_length, log);
      printf("geometry shader log:\n%s\n", log);
      glDeleteShader(geometry_shader);
      glDeleteProgram(program);
      return (0);
    }

    printf("attaching the following geometry shader\n%s", geometry_shader_src);
    glAttachShader(program, geometry_shader);
    glDeleteShader(geometry_shader);
  }

  if (fragment_shader_src) {
    GLint did_fragment_shader_compile = GL_FALSE;
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_src, NULL);
    glCompileShader(fragment_shader);

    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &did_fragment_shader_compile);
    if (did_fragment_shader_compile == GL_FALSE) {
      glGetShaderInfoLog(fragment_shader, sizeof(log), &log_length, log);
      printf("fragment shader log:\n%s\n", log);
      glDeleteShader(fragment_shader);
      glDeleteProgram(program);
      return (0);
    }

    glAttachShader(program, fragment_shader);
    glDeleteShader(fragment_shader);
  }

    glLinkProgram(program);

    GLint did_program_link = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &did_program_link);
    if (did_program_link == GL_FALSE) {
        glGetProgramInfoLog(program, sizeof(log), &log_length, log);
        printf("program log:\n%s\n", log);
        glDeleteProgram(program);
        return (0);
    }

    glUseProgram(program);
    glUseProgram(0);

    return (program);
}

std::vector<tinyobj::shape_t> shapes;

device_mesh_t uploadMesh(const mesh_t & mesh) {
    device_mesh_t out;
    //Allocate vertex array
    //Vertex arrays encapsulate a set of generic vertex
    //attributes and the buffers they are bound to
    //Different vertex array per mesh.
    glGenVertexArrays(1, &(out.vertex_array));
    glBindVertexArray(out.vertex_array);

    //Allocate vbos for data
    glGenBuffers(1, &(out.vbo_vertices));
    glGenBuffers(1, &(out.vbo_normals));
    glGenBuffers(1, &(out.vbo_indices));
    glGenBuffers(1, &(out.vbo_texcoords));

    //Upload vertex data
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo_vertices);
    glBufferData(GL_ARRAY_BUFFER, mesh.vertices.size()*sizeof(glm::vec3),
            &mesh.vertices[0], GL_STATIC_DRAW);
    glVertexAttribPointer(mesh_attributes::POSITION, 3, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(mesh_attributes::POSITION);
    //cout << mesh.vertices.size() << " verts:" << endl;
    //for(int i = 0; i < mesh.vertices.size(); ++i)
    //    cout << "    " << mesh.vertices[i][0] << ", " << mesh.vertices[i][1] << ", " << mesh.vertices[i][2] << endl;

    //Upload normal data
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo_normals);
    glBufferData(GL_ARRAY_BUFFER, mesh.normals.size()*sizeof(glm::vec3),
            &mesh.normals[0], GL_STATIC_DRAW);
    glVertexAttribPointer(mesh_attributes::NORMAL, 3, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(mesh_attributes::NORMAL);
    //cout << mesh.normals.size() << " norms:" << endl;
    //for(int i = 0; i < mesh.normals.size(); ++i)
    //    cout << "    " << mesh.normals[i][0] << ", " << mesh.normals[i][1] << ", " << mesh.normals[i][2] << endl;

    //Upload texture coord data
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo_texcoords);
    glBufferData(GL_ARRAY_BUFFER, mesh.texcoords.size()*sizeof(glm::vec2),
            &mesh.texcoords[0], GL_STATIC_DRAW);
    glVertexAttribPointer(mesh_attributes::TEXCOORD, 2, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(mesh_attributes::TEXCOORD);
    //cout << mesh.texcoords.size() << " texcos:" << endl;
    //for(int i = 0; i < mesh.texcoords.size(); ++i)
    //    cout << "    " << mesh.texcoords[i][0] << ", " << mesh.texcoords[i][1] << endl;

    //indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.vbo_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size()*sizeof(GLushort),
            &mesh.indices[0], GL_STATIC_DRAW);
    out.num_indices = mesh.indices.size();
    //Unplug Vertex Array
    glBindVertexArray(0);


    out.texname = mesh.texname;
    // if (!mesh.texname.empty()) {
    //     GLint texture_width;
    //     GLint texture_height;
    //     GLint texture_channels;
    //     loaded_file_t file = platform_load_file(("crytek-sponza/" + mesh.texname).c_str());
    //     uint8_t* texture_data = stbi_load_from_memory((uint8_t*)file.contents, file.size, &texture_width, &texture_height, &texture_channels, 0);
    //     assert(texture_width > 0 && texture_height > 0 && texture_data != NULL);

    //     glGenTextures(1, &(out.texture));
    //     glBindTexture(GL_TEXTURE_2D, out.texture);

    //     GLint format = texture_channels < 4 ? GL_RGB : GL_RGBA;
    //     glTexImage2D(GL_TEXTURE_2D, 0, format, texture_width, texture_height, 0, format, GL_UNSIGNED_BYTE, texture_data);
    //     glGenerateMipmap(GL_TEXTURE_2D);
    //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    //     glBindTexture(GL_TEXTURE_2D, 0);
    //     free(file.contents);
    //     file.contents = NULL;
    //     free(texture_data);
    //     texture_data = NULL;
    // }

    out.color = mesh.color;
    return out;
}

std::vector<device_mesh_t> draw_meshes;
void initMesh() {
    stbi_set_flip_vertically_on_load(true);

    for(std::vector<tinyobj::shape_t>::iterator it = shapes.begin();
            it != shapes.end(); ++it)
    {
        tinyobj::shape_t shape = *it;
        int totalsize = shape.mesh.indices.size() / 3;
        int f = 0;
        while(f<totalsize){
            mesh_t mesh;
            int process = std::min(10000, totalsize-f);
            int point = 0;
            for(int i=f; i<process+f; i++){
                int idx0 = shape.mesh.indices[3*i];
                int idx1 = shape.mesh.indices[3*i+1];
                int idx2 = shape.mesh.indices[3*i+2];
                glm::vec3 p0 = glm::vec3(shape.mesh.positions[3*idx0],
                               shape.mesh.positions[3*idx0+1],
                               shape.mesh.positions[3*idx0+2]);
                glm::vec3 p1 = glm::vec3(shape.mesh.positions[3*idx1],
                               shape.mesh.positions[3*idx1+1],
                               shape.mesh.positions[3*idx1+2]);
                glm::vec3 p2 = glm::vec3(shape.mesh.positions[3*idx2],
                               shape.mesh.positions[3*idx2+1],
                               shape.mesh.positions[3*idx2+2]);

                mesh.vertices.push_back(p0);
                mesh.vertices.push_back(p1);
                mesh.vertices.push_back(p2);

                if(shape.mesh.normals.size() > 0)
                {
                    mesh.normals.push_back(glm::vec3(shape.mesh.normals[3*idx0],
                                                shape.mesh.normals[3*idx0+1],
                                                shape.mesh.normals[3*idx0+2]));
                    mesh.normals.push_back(glm::vec3(shape.mesh.normals[3*idx1],
                                                shape.mesh.normals[3*idx1+1],
                                                shape.mesh.normals[3*idx1+2]));
                    mesh.normals.push_back(glm::vec3(shape.mesh.normals[3*idx2],
                                                shape.mesh.normals[3*idx2+1],
                                                shape.mesh.normals[3*idx2+2]));
                }
                else
                {
                    glm::vec3 norm = glm::normalize(glm::cross(glm::normalize(p1-p0), glm::normalize(p2-p0)));
                    mesh.normals.push_back(norm);
                    mesh.normals.push_back(norm);
                    mesh.normals.push_back(norm);
                }

                if(shape.mesh.texcoords.size() > 0)
                {
                    mesh.texcoords.push_back(glm::vec2(shape.mesh.texcoords[2*idx0],
                                                  shape.mesh.texcoords[2*idx0+1]));
                    mesh.texcoords.push_back(glm::vec2(shape.mesh.texcoords[2*idx1],
                                                  shape.mesh.texcoords[2*idx1+1]));
                    mesh.texcoords.push_back(glm::vec2(shape.mesh.texcoords[2*idx2],
                                                  shape.mesh.texcoords[2*idx2+1]));
                }
                else
                {
                    glm::vec2 tex(0.0);
                    mesh.texcoords.push_back(tex);
                    mesh.texcoords.push_back(tex);
                    mesh.texcoords.push_back(tex);
                }
                mesh.indices.push_back(point++);
                mesh.indices.push_back(point++);
                mesh.indices.push_back(point++);
            }

            mesh.color = glm::vec3(shape.material.diffuse[0],
                              shape.material.diffuse[1],
                              shape.material.diffuse[2]);
            mesh.texname = shape.material.diffuse_texname;
            draw_meshes.push_back(uploadMesh(mesh));
            f=f+process;
        }
        // break;
    }
}

Camera cam(glm::vec3(0.0, 0.0, 1.5),
        glm::normalize(glm::vec3(1,0,0)),
        glm::normalize(glm::vec3(0,0,1)));

void Camera::adjust(float dx) {
    rx += dx;
    rx = fmod(rx,360.0f);
}

glm::mat4 Camera::get_view() {
    // glm::vec3 inclin = glm::gtx::rotate_vector::rotate(start_dir,ry,start_left);
    // glm::vec3 spun = glm::gtx::rotate_vector::rotate(inclin,rx,up);
    // glm::vec3 cent(pos, z);
    // return lookAt(cent, cent + spun, up);
    glm::vec3 inclin = glm::rotate(start_dir,0.2f,start_left);
    glm::vec3 spun = glm::rotate(inclin,rx,up);
    return glm::lookAt(pos, pos + spun, up);
}

void setupGbuffer(gbuffer_t *gbuffer, int32_t w, int32_t h) {
    /* scene depth and framebuffer setup */
    glGenFramebuffers(1, &gbuffer->fb);
    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer->fb);

    glGenTextures(1, &gbuffer->depthTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->depthTexture);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB32F, w, h, 2, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures(1, &gbuffer->normalTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->normalTexture);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB32F, w, h, 2, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures(1, &gbuffer->colorTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->colorTexture);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGB32F, w, h, 2, 0, GL_RGBA, GL_FLOAT, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gbuffer->depthTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, gbuffer->normalTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, gbuffer->colorTexture, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ao_init(ao_memory_t* mem)
{
    assert(mem != NULL);

    std::cout << "version " << glGetString(GL_VERSION) << std::endl;
	/* opaque shader program */

    char* solid_vertex_shader_src = (char*)R"(
        #version 410
        uniform mat4 u_matrix;
        uniform mat4 u_mv_matrix;
        uniform mat4 u_normal_matrix;
        layout (location = 0) in vec3 a_position;
        layout (location = 1) in vec3 a_normal;
        layout (location = 2) in vec2 a_uv_coord;

        out VS_OUT {
          vec3 position;
          vec3 normal;
          vec2 uv_coord;
        } vs_out;

        void main()
        {
          vs_out.position = (u_mv_matrix * vec4(a_position, 1.0)).xyz;
          vs_out.normal = (u_normal_matrix * vec4(a_normal, 0.0)).xyz;
          vs_out.uv_coord = a_uv_coord;
          gl_Position = u_matrix * vec4(a_position, 1.0);
        }
    )";

    char* solid_geometry_shader_src = (char*)R"(
      #version 410
      layout(triangles) in;
      layout(triangle_strip, max_vertices = 3) out;

      in VS_OUT {
        vec3 position;
        vec3 normal;
        vec2 uv_coord;
      } gs_in[];

      out vec3 v_position;
      out vec3 v_normal;
      out vec2 v_uv_coord;
      out float v_layer;

      void main()
      {
        for(int layer = 0; layer < 2; ++layer) {
            gl_Layer = layer;
            v_layer = float(layer);
            for (int i = 0; i < 3; ++i) {
              v_position = gs_in[i].position;
              v_normal = gs_in[i].normal;
              v_uv_coord = gs_in[i].uv_coord;
              gl_Position = gl_in[i].gl_Position;
              EmitVertex();
            }
            EndPrimitive();
        }
      }
    )";

    char* solid_fragment_shader_src = (char*)R"(
        #version 410
        in vec3 v_position;
        in vec3 v_normal;
        in vec2 v_uv_coord;
        in float v_layer;
        layout (location = 0) out vec4 output_normal;
        layout (location = 1) out vec4 output_color;
        uniform vec3 u_color;
        uniform bool u_texture_assigned_color;
        uniform sampler2D u_texture_sampler;
        void main()
		{
            if (v_layer == 0.0) {
                // output_normal = vec4(v_position, 1.0);
                output_normal = vec4(normalize(v_normal), 1.0);
                vec3 color = u_texture_assigned_color ? texture(u_texture_sampler, v_uv_coord).rgb : u_color;
                output_color = vec4(color, 1.0);
            } else {
                if (false) discard;
                output_normal = vec4(-normalize(v_normal), 1.0);
                vec3 color = u_texture_assigned_color ? texture(u_texture_sampler, v_uv_coord).rgb : u_color;
                output_color = vec4(color, 1.0);
            }
        }
    )";

	GLuint solid_program = gl_create_shader_program(solid_vertex_shader_src, solid_geometry_shader_src, solid_fragment_shader_src);
	assert(solid_program != 0);

	GLint mesh_matrix_location = glGetUniformLocation(solid_program, "u_matrix");
	assert(mesh_matrix_location >= 0);
    GLint mesh_mv_matrix_location = glGetUniformLocation(solid_program, "u_mv_matrix");
    assert(mesh_mv_matrix_location >= 0);
    GLint mesh_normal_matrix_location = glGetUniformLocation(solid_program, "u_normal_matrix");
    assert(mesh_normal_matrix_location >= 0);
    GLint mesh_color_location = glGetUniformLocation(solid_program, "u_color");
    assert(mesh_color_location >= 0);
    GLint mesh_texture_assigned_color_location = glGetUniformLocation(solid_program, "u_texture_assigned_color");
    assert(mesh_texture_assigned_color_location >= 0);
    GLint mesh_texture_sampler_location = glGetUniformLocation(solid_program, "u_texture_sampler");
    assert(mesh_texture_sampler_location >= 0);


    char* ao_vertex_shader_src = (char*)""
        "#version 410\n"
        "layout(location = 0) in vec2 a_position;\n"
        "void main()\n"
        "{\n"
        "\tgl_Position = vec4(a_position.xy, 0.0, 1.0);\n"
        "}\n";

    char* ao_fragment_shader_src = (char*)R"(
        #version 410
        layout(location = 0) out vec4 output_color;
        uniform sampler2DArray u_depth_texture_sampler;
        uniform sampler2DArray u_normal_texture_sampler;
        uniform sampler2DArray u_color_texture_sampler;
        uniform vec2 u_inverse_viewport_resolution;
        uniform mat4 u_inverse_projection_matrix;

        float linearizeDepth(float exp_depth, float near, float far) {
            return  (2 * near) / (far + near - (exp_depth*2.0-1.0) * (far - near));
        }

        void main() {
            ivec3 C = ivec3(gl_FragCoord.xy, 0);
            float depth0 = texelFetch(u_depth_texture_sampler, C, 0).x;
            float depthLinear0 = linearizeDepth(depth0, 0.1, 100.0);
            vec3 normal0 = texelFetch(u_normal_texture_sampler, C, 0).xyz;
            vec3 color0 = texelFetch(u_color_texture_sampler, C, 0).xyz;
            C.z = 1;
            float depth1 = texelFetch(u_depth_texture_sampler, C, 0).x;
            float depthLinear1 = linearizeDepth(depth1, 0.1, 100.0);
            vec3 normal1 = texelFetch(u_normal_texture_sampler, C, 0).xyz;
            vec3 color1 = texelFetch(u_color_texture_sampler, C, 0).xyz;

            vec2 positionScreenSpace = (gl_FragCoord.xy * u_inverse_viewport_resolution) * 2.0 - 1.0;
            vec4 farPlaneViewSpace = u_inverse_projection_matrix * vec4(positionScreenSpace, 1.0, 1.0);
            farPlaneViewSpace.xyz /= farPlaneViewSpace.w;
            vec3 position0 = farPlaneViewSpace.xyz * depthLinear0;
            vec3 position1 = farPlaneViewSpace.xyz * depthLinear1;

            output_color = vec4(normal0, 1.0);
        }
    )";
    GLuint ao_program = gl_create_shader_program(ao_vertex_shader_src, NULL, ao_fragment_shader_src);
    assert(ao_program != 0);

    GLint ao_depth_texture_sampler_location = glGetUniformLocation(ao_program, "u_depth_texture_sampler");
    assert(ao_depth_texture_sampler_location >= 0);
    GLint ao_normal_texture_sampler_location = glGetUniformLocation(ao_program, "u_normal_texture_sampler");
    assert(ao_normal_texture_sampler_location >= 0);
    GLint ao_color_texture_sampler_location = glGetUniformLocation(ao_program, "u_color_texture_sampler");
    assert(ao_color_texture_sampler_location >= 0);

    GLint inverse_viewport_resolution_location = glGetUniformLocation(ao_program, "u_inverse_viewport_resolution");
    assert(inverse_viewport_resolution_location >= 0);
    GLint inverse_projection_matrix_location = glGetUniformLocation(ao_program, "u_inverse_projection_matrix");
    assert(inverse_projection_matrix_location >= 0);
    std::cout << "SHADERS COMPILED" << std::endl;


    std::cout << "LOADING" << std::endl;
    std::string err = tinyobj::LoadObj(shapes, "crytek-sponza/sponza.obj", "crytek-sponza/");
    if(!err.empty()) {
        std::cout << "MODEL LOAD FAIL" << std::endl;
        std::cerr << err << std::endl;
    } else {
        std::cout << "MODEL LOAD SUCCESS" << std::endl;
    }

    initMesh();
    assert(glGetError() == GL_NO_ERROR);
    std::cout << "INIT MESH COMPLETE" << std::endl;


    setupGbuffer(&mem->gbuffer, mem->window_width, mem->window_height);
    assert(glGetError() == GL_NO_ERROR);


	/* plane vertex array object */
	GLuint plane_vao;
	glGenVertexArrays(1, &plane_vao);
	glBindVertexArray(plane_vao);

	GLuint plane_vbo;
	glGenBuffers(1, &plane_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, plane_vbo);
    GLsizei plane_vertex_size = sizeof(GLfloat) * 2;
	glBufferData(GL_ARRAY_BUFFER, plane_vertex_size * plane_vert_count, (const void*)plane_vertices, GL_STATIC_DRAW);

	GLsizei plane_stride = plane_vertex_size;
	uintptr_t plane_offset = 0;
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, plane_stride, (const void*)plane_offset);

	glBindVertexArray(0);


	/* store init variables in mem */
	mem->frame = 0;
	mem->plane_vao = plane_vao;
	mem->plane_vbo = plane_vbo;
	mem->gbuffer_program = solid_program;
    mem->mesh_uniform_matrix_location = mesh_matrix_location;
    mem->mesh_uniform_mv_matrix_location = mesh_mv_matrix_location;
    mem->mesh_uniform_normal_matrix_location = mesh_normal_matrix_location;
    mem->mesh_uniform_color_location = mesh_color_location;
	mem->mesh_uniform_texture_assigned_color_location = mesh_texture_assigned_color_location;
	mem->mesh_uniform_texture_sampler_location = mesh_texture_sampler_location;
    mem->ao_program = ao_program;
    mem->ao_uniform_depth_texture_sampler_location = ao_depth_texture_sampler_location;
    mem->ao_uniform_normal_texture_sampler_location = ao_normal_texture_sampler_location;
    mem->ao_uniform_color_texture_sampler_location = ao_color_texture_sampler_location;
    mem->uniform_inverse_viewport_resolution_location = inverse_viewport_resolution_location;
    mem->uniform_inverse_projection_matrix_location = inverse_projection_matrix_location;

	assert(glGetError() == GL_NO_ERROR);
}

glm::mat4 get_mesh_world() {
    glm::vec3 tilt(1.0f,0.0f,0.0f);
    glm::mat4 translate_mat = glm::translate(glm::vec3(0.0f,.5f,0.0f));
    glm::mat4 tilt_mat = glm::rotate(glm::mat4(), (float)M_PI * 0.5f, tilt);
    glm::mat4 scale_mat = glm::scale(glm::mat4(), glm::vec3(0.01));
    return tilt_mat * scale_mat;
}

float FARP;
float NEARP;
void draw_mesh(ao_memory_t* mem) {
    glUseProgram(mem->gbuffer_program);

    FARP = 100.0f;
    NEARP = 0.1f;

    glm::mat4 model = get_mesh_world();
    glm::mat4 view = cam.get_view();
    glm::mat4 persp = glm::perspective(45.0f,(float)mem->window_width/(float)mem->window_height,NEARP,FARP);
    glm::mat4 mv_matrix = view * model;
    glm::mat4 mvp_matrix = persp * view * model;
    glm::mat4 inverse_transposed = glm::transpose(glm::inverse(view*model));

    glUniformMatrix4fv(mem->mesh_uniform_matrix_location, 1, GL_FALSE, &mvp_matrix[0][0]);
    glUniformMatrix4fv(mem->mesh_uniform_mv_matrix_location, 1, GL_FALSE, &mv_matrix[0][0]);
    glUniformMatrix4fv(mem->mesh_uniform_normal_matrix_location, 1, GL_FALSE, &inverse_transposed[0][0]);

    glUniform1i(mem->mesh_uniform_texture_sampler_location, 0);
    glActiveTexture(GL_TEXTURE0);

    GLenum draw_buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, draw_buffers);

    for(unsigned int i=0; i<draw_meshes.size(); i++){
        // glBindTexture(GL_TEXTURE_2D, draw_meshes[i].texture);
        glUniform3fv(mem->mesh_uniform_color_location, 1, &(draw_meshes[i].color[0]));
        glUniform1i(mem->mesh_uniform_texture_assigned_color_location, false);//!draw_meshes[i].texname.empty());

        glBindVertexArray(draw_meshes[i].vertex_array);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, draw_meshes[i].vbo_indices);
        glDrawElements(GL_TRIANGLES, draw_meshes[i].num_indices, GL_UNSIGNED_SHORT,0);
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void draw_plane(ao_memory_t* mem) {
    glUseProgram(mem->ao_program);

    glUniform2f(mem->uniform_inverse_viewport_resolution_location, 1.0 / mem->window_width, 1.0 / mem->window_height);
    glm::mat4 inversePersp = glm::inverse(glm::perspective(45.0f,(float)mem->window_width/(float)mem->window_height,NEARP,FARP));
    glUniformMatrix4fv(mem->uniform_inverse_projection_matrix_location, 1, GL_FALSE, &inversePersp[0][0]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mem->gbuffer.depthTexture);
    glUniform1i(mem->ao_uniform_depth_texture_sampler_location, 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mem->gbuffer.normalTexture);
    glUniform1i(mem->ao_uniform_normal_texture_sampler_location, 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mem->gbuffer.colorTexture);
    glUniform1i(mem->ao_uniform_color_texture_sampler_location, 2);

    glBindVertexArray(mem->plane_vao);
    glDrawArrays(GL_TRIANGLES, 0, plane_vert_count);

    glBindVertexArray(0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glUseProgram(0);
}

void ao_update_frame(ao_memory_t* mem)
{
    assert(mem != NULL);

    cam.adjust(0.01f);

    glBindFramebuffer(GL_FRAMEBUFFER, mem->gbuffer.fb);
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    draw_mesh(mem);
    assert(glGetError() == GL_NO_ERROR);

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // glBindFramebuffer(GL_READ_FRAMEBUFFER, mem->gbuffer.fb);
    // glReadBuffer(GL_COLOR_ATTACHMENT3);
    // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    // glBlitFramebuffer(0, 0, mem->window_width, mem->window_height, 0, 0, mem->window_width, mem->window_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    // glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    draw_plane(mem);

	assert(glGetError() == GL_NO_ERROR);

	++mem->frame;
}

