#include "ao.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/rotate_vector.hpp"

#define USE_TEXTURES false

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

    //Upload normal data
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo_normals);
    glBufferData(GL_ARRAY_BUFFER, mesh.normals.size()*sizeof(glm::vec3),
            &mesh.normals[0], GL_STATIC_DRAW);
    glVertexAttribPointer(mesh_attributes::NORMAL, 3, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(mesh_attributes::NORMAL);

    //Upload texture coord data
    glBindBuffer(GL_ARRAY_BUFFER, out.vbo_texcoords);
    glBufferData(GL_ARRAY_BUFFER, mesh.texcoords.size()*sizeof(glm::vec2),
            &mesh.texcoords[0], GL_STATIC_DRAW);
    glVertexAttribPointer(mesh_attributes::TEXCOORD, 2, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(mesh_attributes::TEXCOORD);

    //indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, out.vbo_indices);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size()*sizeof(GLushort),
            &mesh.indices[0], GL_STATIC_DRAW);
    out.num_indices = mesh.indices.size();
    //Unplug Vertex Array
    glBindVertexArray(0);


    out.texname = mesh.texname;
    if (USE_TEXTURES && !mesh.texname.empty()) {
        GLint texture_width;
        GLint texture_height;
        GLint texture_channels;
        loaded_file_t file = platform_load_file(("crytek-sponza/" + mesh.texname).c_str());
        uint8_t* texture_data = stbi_load_from_memory((uint8_t*)file.contents, file.size, &texture_width, &texture_height, &texture_channels, 0);
        assert(texture_width > 0 && texture_height > 0 && texture_data != NULL);

        glGenTextures(1, &(out.texture));
        glBindTexture(GL_TEXTURE_2D, out.texture);

        GLint format = texture_channels < 4 ? GL_RGB : GL_RGBA;
        glTexImage2D(GL_TEXTURE_2D, 0, format, texture_width, texture_height, 0, format, GL_UNSIGNED_BYTE, texture_data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

        glBindTexture(GL_TEXTURE_2D, 0);
        free(file.contents);
        file.contents = NULL;
        free(texture_data);
        texture_data = NULL;
    }

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
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, w, h, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glGenTextures(1, &gbuffer->depthTextureBack);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->depthTextureBack);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, w, h, 2, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

    glGenTextures(1, &gbuffer->normalTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->normalTexture);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, w, h, 2, 0, GL_RGBA, GL_FLOAT, 0);

    glGenTextures(1, &gbuffer->colorTexture);
    glBindTexture(GL_TEXTURE_2D_ARRAY, gbuffer->colorTexture);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, w, h, 2, 0, GL_RGBA, GL_FLOAT, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gbuffer->depthTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, gbuffer->normalTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, gbuffer->colorTexture, 0);

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void swapDepthTextures(gbuffer_t *gbuffer) {
    /* scene depth and framebuffer setup */
    glBindFramebuffer(GL_FRAMEBUFFER, gbuffer->fb);

    GLuint depthTextureTmp = gbuffer->depthTextureBack;
    gbuffer->depthTextureBack = gbuffer->depthTexture;
    gbuffer->depthTexture = depthTextureTmp;

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gbuffer->depthTexture, 0);

    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
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
        uniform mat4 u_mv_matrix_back;
        uniform mat4 u_normal_matrix;
        layout (location = 0) in vec3 a_position;
        layout (location = 1) in vec3 a_normal;
        layout (location = 2) in vec2 a_uv_coord;

        out VS_OUT {
          vec3 position;
          vec3 position_back;
          vec3 normal;
          vec2 uv_coord;
        } vs_out;

        void main()
        {
          vs_out.position = (u_mv_matrix * vec4(a_position, 1.0)).xyz;
          vs_out.position_back = (u_mv_matrix_back * vec4(a_position, 1.0)).xyz;
          vs_out.normal = (u_normal_matrix * vec4(a_normal, 0.0)).xyz;
          vs_out.uv_coord = a_uv_coord;
          gl_Position = u_matrix * vec4(a_position, 1.0);
        }
    )";

    char* solid_geometry_shader_src = (char*)R"(
        #version 410
        layout(invocations = 2, triangles) in;
        layout(triangle_strip, max_vertices = 3) out;

        in VS_OUT {
            vec3 position;
            vec3 position_back;
            vec3 normal;
            vec2 uv_coord;
        } gs_in[];

        out vec3 v_position;
        out vec3 v_position_back;
        out vec3 v_normal;
        out vec2 v_uv_coord;
        out float v_layer;

        void main()
        {
            gl_Layer = gl_InvocationID;
            v_layer = float(gl_InvocationID);
            for (int i = 0; i < 3; ++i) {
                v_position = gs_in[i].position;
                v_position_back = gs_in[i].position_back;
                v_normal = gs_in[i].normal;
                v_uv_coord = gs_in[i].uv_coord;
                gl_Position = gl_in[i].gl_Position;
                EmitVertex();
            }
            EndPrimitive();
        }
    )";

    char* solid_fragment_shader_src = (char*)R"(
        #version 410
        in vec3 v_position;
        in vec3 v_position_back;
        in vec3 v_normal;
        in vec2 v_uv_coord;
        in float v_layer;
        layout (location = 0) out vec4 output_normal;
        layout (location = 1) out vec4 output_color;
        uniform vec3 u_color;
        uniform bool u_texture_assigned_color;
        uniform sampler2D u_texture_sampler;
        uniform sampler2DArray u_depth_texture_back_sampler;
        uniform mat4 u_projection_matrix;

        float linearizeDepth(float exp_depth, float near, float far) {
            return  (2 * near) / (far + near - (exp_depth*2.0-1.0) * (far - near));
        }

        vec2 getSSPositionChange(vec3 csPosition, vec3 csPrevPosition, mat4 projectToScreenMatrix) {
            vec4 temp = projectToScreenMatrix * vec4(csPrevPosition, 1.0);

            // gl_FragCoord.xy has already been rounded to a pixel center, so regenerate the true projected position.
            // This is needed to generate correct velocity vectors in the presence of Projection::pixelOffset
            vec4 temp2 = projectToScreenMatrix * vec4(csPosition, 1.0);

            // We want the precision of division here and intentionally do not convert to multiplying by an inverse.
            // Expressing the two divisions as a single vector division operation seems to prevent the compiler from
            // computing them at different precisions, which gives non-zero velocity for static objects in some cases.
            vec4 ssPositions = vec4(temp.xy, temp2.xy) / vec4(temp.ww, temp2.ww);

            return ssPositions.zw - ssPositions.xy;
        }

        // z_t−1 > Z_t−1[0][x_t−1, y_t−1] + z_minSep
        bool isInFrontOfSecondLayer(in vec2 ssV, in float minZGap, in float prevZ) {
            vec2 prevSSC = gl_FragCoord.xy - ssV;
            ivec3 C = ivec3(prevSSC, 0);
            float depthBack = texelFetch(u_depth_texture_back_sampler, C, 0).x;
            float depthBackLinear = linearizeDepth(depthBack, 0.1, 100.0);

            return -prevZ <= depthBackLinear * 100.0 + minZGap;
        }

        void main()
		{
            vec2 ssPositionChange = getSSPositionChange(v_position, v_position_back, u_projection_matrix);
            if (v_layer != 0.0 && isInFrontOfSecondLayer(ssPositionChange, 0.5, v_position_back.z)) {
                discard;
            }

            output_normal = vec4(normalize(v_normal), 1.0);
            vec3 color = u_texture_assigned_color ? texture(u_texture_sampler, v_uv_coord).rgb : u_color;
            output_color = vec4(color, 1.0);
            // discard;
        }
    )";

	GLuint solid_program = gl_create_shader_program(solid_vertex_shader_src, solid_geometry_shader_src, solid_fragment_shader_src);
	assert(solid_program != 0);

	GLint mesh_matrix_location = glGetUniformLocation(solid_program, "u_matrix");
	assert(mesh_matrix_location >= 0);
    GLint mesh_mv_matrix_location = glGetUniformLocation(solid_program, "u_mv_matrix");
    assert(mesh_mv_matrix_location >= 0);
    GLint mesh_mv_matrix_back_location = glGetUniformLocation(solid_program, "u_mv_matrix_back");
    assert(mesh_mv_matrix_back_location >= 0);
    GLint mesh_normal_matrix_location = glGetUniformLocation(solid_program, "u_normal_matrix");
    assert(mesh_normal_matrix_location >= 0);
    GLint mesh_color_location = glGetUniformLocation(solid_program, "u_color");
    assert(mesh_color_location >= 0);
    GLint mesh_texture_assigned_color_location = glGetUniformLocation(solid_program, "u_texture_assigned_color");
    assert(mesh_texture_assigned_color_location >= 0);
    GLint mesh_texture_sampler_location = glGetUniformLocation(solid_program, "u_texture_sampler");
    assert(mesh_texture_sampler_location >= 0);

    GLint gbuffer_depth_texture_back_sampler_location = glGetUniformLocation(solid_program, "u_depth_texture_back_sampler");
    assert(gbuffer_depth_texture_back_sampler_location >= 0);
    GLint gbuffer_projection_matrix_location = glGetUniformLocation(solid_program, "u_projection_matrix");
    assert(gbuffer_projection_matrix_location >= 0);

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

        int NUM_SAMPLES = 10;
        int NUM_SPIRAL_TURNS = 10;

        float linearizeDepth(float exp_depth, float near, float far) {
            return  (2 * near) / (far + near - (exp_depth*2.0-1.0) * (far - near));
        }

        vec2 tapLocation(int sampleNumber, float spinAngle, out float screen_space_radius){
            float alpha = float(sampleNumber + 0.5) * (1.0 / NUM_SAMPLES);
            float angle = alpha * (NUM_SPIRAL_TURNS * 6.28) + spinAngle;

            screen_space_radius = alpha;
            return vec2(cos(angle), sin(angle));
        }

        void getOffsetPositions(ivec2 screen_space_coord, vec2 unitOffset, float screen_space_radius, out vec3 P0, out vec3 P1) {
            ivec2 screen_space_point = ivec2((screen_space_radius * unitOffset) + screen_space_coord);
            vec2 positionScreenSpace = (screen_space_point * u_inverse_viewport_resolution) * 2.0 - 1.0;

            float linear_depth0 = linearizeDepth(texelFetch(u_depth_texture_sampler, ivec3(screen_space_point, 0), 0).x, 0.1, 100.0);
            float linear_depth1 = linearizeDepth(texelFetch(u_depth_texture_sampler, ivec3(screen_space_point, 1), 0).x, 0.1, 100.0);

            vec4 farPlaneViewSpace = u_inverse_projection_matrix * vec4(positionScreenSpace, 1.0, 1.0);
            farPlaneViewSpace.xyz /= farPlaneViewSpace.w;

            P0 = farPlaneViewSpace.xyz * linear_depth0;
            P1 = farPlaneViewSpace.xyz * linear_depth1;
        }

        float fallOffFunction(float vv, float vn, float epsilon) {
            float invRadius2 = 0.5;
            float bias = 0.01;
            float f = max(1.0 - vv * invRadius2, 0.0);
            return f * max((vn - bias) * inversesqrt(epsilon + vv), 0.0);
        }

        float aoValueFromPositionsAndNormal(vec3 C, vec3 n_C, vec3 Q) {
            vec3 v = Q - C;
            float vv = dot(v, v);
            float vn = dot(v, n_C);
            const float epsilon = 0.001;

            return fallOffFunction(vv, vn, epsilon) * mix(1.0, max(0.0, 1.5 * n_C.z), 0.35);
        }

        float sampleAO(in ivec2 screen_space_coord, in vec3 camera_space_position, in vec3 camera_space_normal, in float screen_space_disk_radius, in int tapIndex, in float randomPatternRotationAngle, in float scale) {
            float screen_space_radius;
            vec2 unitOffset = tapLocation(tapIndex, randomPatternRotationAngle, screen_space_radius);

            // Ensure that the taps are at least 1 pixel away
            screen_space_radius = max(0.75, screen_space_radius * screen_space_disk_radius);

            vec3 Q0, Q1;
            getOffsetPositions(screen_space_coord, unitOffset, screen_space_radius, Q0, Q1);

            float AO0 = aoValueFromPositionsAndNormal(camera_space_position, camera_space_normal, Q0);
            float AO1 = aoValueFromPositionsAndNormal(camera_space_position, camera_space_normal, Q1);
            return max(AO0, AO1);
        }

        void main() {
            ivec3 screen_space_coord = ivec3(gl_FragCoord.xy, 0);
            float depth0 = texelFetch(u_depth_texture_sampler, screen_space_coord, 0).x;
            float depthLinear0 = linearizeDepth(depth0, 0.1, 100.0);
            vec3 normal0 = texelFetch(u_normal_texture_sampler, screen_space_coord, 0).xyz;
            vec3 color0 = texelFetch(u_color_texture_sampler, screen_space_coord, 0).xyz;
            screen_space_coord.z = 1;
            float depth1 = texelFetch(u_depth_texture_sampler, screen_space_coord, 0).x;
            float depthLinear1 = linearizeDepth(depth1, 0.1, 100.0);
            // vec3 normal1 = texelFetch(u_normal_texture_sampler, screen_space_coord, 0).xyz;
            // vec3 color1 = texelFetch(u_color_texture_sampler, screen_space_coord, 0).xyz;

            vec2 positionScreenSpace = (gl_FragCoord.xy * u_inverse_viewport_resolution) * 2.0 - 1.0;
            vec4 farPlaneViewSpace = u_inverse_projection_matrix * vec4(positionScreenSpace, 1.0, 1.0);
            farPlaneViewSpace.xyz /= farPlaneViewSpace.w;
			// NOTE(lars): these are camera space positions, thanks Angela!
            vec3 position0 = farPlaneViewSpace.xyz * depthLinear0;
            vec3 position1 = farPlaneViewSpace.xyz * depthLinear1;

            float position0z = depthLinear0 * -farPlaneViewSpace.z;
            float position1z = depthLinear1 * -farPlaneViewSpace.z;

            float projScale = 500.0;
            float radius = 1.0;
            float screen_space_disk_radius = projScale * radius / position0z;

            float min_disk_radius = 3.0;
            if (screen_space_disk_radius <= min_disk_radius) {
                // bail
                output_color = vec4(1.0, 0.0, 1.0, 1.0);
                return;
            }

            float randomPatternRotationAngle = (((3 * screen_space_coord.x) ^ (screen_space_coord.y + screen_space_coord.x * screen_space_coord.y))) * 10;

            float sum = 0.0;
            for (int i = 0; i < NUM_SAMPLES; ++i) {
                sum += sampleAO(screen_space_coord.xy, position0, normal0, screen_space_disk_radius, i, randomPatternRotationAngle, 1);
            }

            float intensity = 1.0;
            float A = pow(max(0.0, 1.0 - sqrt(sum * (3.0 / NUM_SAMPLES))), intensity);

            // Fade in as the radius reaches 2 pixels
            float visibility = mix(1.0, A, clamp(screen_space_disk_radius - min_disk_radius, 0.0, 1.0));

            output_color = vec4(visibility * color0, 1.0);
            // output_color = vec4(vec3(position1z - position0z), 1.0);
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

    GLint ao_inverse_viewport_resolution_location = glGetUniformLocation(ao_program, "u_inverse_viewport_resolution");
    assert(ao_inverse_viewport_resolution_location >= 0);
    GLint ao_inverse_projection_matrix_location = glGetUniformLocation(ao_program, "u_inverse_projection_matrix");
    assert(ao_inverse_projection_matrix_location >= 0);
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
	mem->gbuffer_program = solid_program;
    mem->mesh_uniform_matrix_location = mesh_matrix_location;
    mem->mesh_uniform_mv_matrix_location = mesh_mv_matrix_location;
    mem->mesh_uniform_mv_matrix_back_location = mesh_mv_matrix_back_location;
    mem->mesh_uniform_normal_matrix_location = mesh_normal_matrix_location;
    mem->mesh_uniform_color_location = mesh_color_location;
	mem->mesh_uniform_texture_assigned_color_location = mesh_texture_assigned_color_location;
    mem->mesh_uniform_texture_sampler_location = mesh_texture_sampler_location;
	mem->gbuffer_uniform_depth_texture_back_sampler_location = gbuffer_depth_texture_back_sampler_location;
    mem->gbuffer_uniform_projection_matrix_location = gbuffer_projection_matrix_location;

    mem->plane_vao = plane_vao;
    mem->plane_vbo = plane_vbo;
    mem->ao_program = ao_program;
    mem->ao_uniform_depth_texture_sampler_location = ao_depth_texture_sampler_location;
    mem->ao_uniform_normal_texture_sampler_location = ao_normal_texture_sampler_location;
    mem->ao_uniform_color_texture_sampler_location = ao_color_texture_sampler_location;
    mem->ao_uniform_inverse_viewport_resolution_location = ao_inverse_viewport_resolution_location;
    mem->ao_uniform_inverse_projection_matrix_location = ao_inverse_projection_matrix_location;

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
    glUniformMatrix4fv(mem->gbuffer_uniform_projection_matrix_location, 1, GL_FALSE, &persp[0][0]);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, mem->gbuffer.depthTextureBack);
    glUniform1i(mem->gbuffer_uniform_depth_texture_back_sampler_location, 1);

    glActiveTexture(GL_TEXTURE0);
    glUniform1i(mem->mesh_uniform_texture_sampler_location, 0);

    GLenum draw_buffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, draw_buffers);

    for(unsigned int i=0; i<draw_meshes.size(); i++){
        if (USE_TEXTURES) glBindTexture(GL_TEXTURE_2D, draw_meshes[i].texture);
        glUniform3fv(mem->mesh_uniform_color_location, 1, &(draw_meshes[i].color[0]));
        glUniform1i(mem->mesh_uniform_texture_assigned_color_location, USE_TEXTURES ? !draw_meshes[i].texname.empty() : false);

        glBindVertexArray(draw_meshes[i].vertex_array);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, draw_meshes[i].vbo_indices);
        glDrawElements(GL_TRIANGLES, draw_meshes[i].num_indices, GL_UNSIGNED_SHORT,0);
    }
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0);

    glUniformMatrix4fv(mem->mesh_uniform_mv_matrix_back_location, 1, GL_FALSE, &mv_matrix[0][0]);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    glUseProgram(0);
}

void draw_plane(ao_memory_t* mem) {
    glUseProgram(mem->ao_program);

    glUniform2f(mem->ao_uniform_inverse_viewport_resolution_location, 1.0 / mem->window_width, 1.0 / mem->window_height);
    glm::mat4 inversePersp = glm::inverse(glm::perspective(45.0f,(float)mem->window_width/(float)mem->window_height,NEARP,FARP));
    glUniformMatrix4fv(mem->ao_uniform_inverse_projection_matrix_location, 1, GL_FALSE, &inversePersp[0][0]);

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

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mem->gbuffer.depthTexture, 0, 0);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mem->gbuffer.normalTexture, 0, 0);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mem->gbuffer.colorTexture, 0, 0);

    // float scene_texture_clear[4] = {0.55f, 0.65f, 0.85f, 1.0f};
    // glClearBufferfv(GL_COLOR, 0, scene_texture_clear);
    glClearColor(0.55f, 0.65f, 0.85f, 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mem->gbuffer.depthTexture, 0, 1);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mem->gbuffer.normalTexture, 0, 1);
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mem->gbuffer.colorTexture, 0, 1);

    // float scene_texture_clear[4] = {0.55f, 0.65f, 0.85f, 1.0f};
    // glClearBufferfv(GL_COLOR, 0, scene_texture_clear);
    glClearColor(0.55f, 0.65f, 0.00f, 1.0f);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, mem->gbuffer.depthTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mem->gbuffer.normalTexture, 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, mem->gbuffer.colorTexture, 0);

    glEnable(GL_DEPTH_TEST);

    draw_mesh(mem);
    assert(glGetError() == GL_NO_ERROR);

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glClearColor(0.55f, 0.00f, 0.85f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    assert(glGetError() == GL_NO_ERROR);

    // glBindFramebuffer(GL_READ_FRAMEBUFFER, mem->gbuffer.fb);
    // glReadBuffer(GL_COLOR_ATTACHMENT3);
    // glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    // glBlitFramebuffer(0, 0, mem->window_width, mem->window_height, 0, 0, mem->window_width, mem->window_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    // glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    draw_plane(mem);
    assert(glGetError() == GL_NO_ERROR);

    swapDepthTextures(&mem->gbuffer);
    assert(glGetError() == GL_NO_ERROR);

	++mem->frame;
}

