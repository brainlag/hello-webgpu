#include "webgpu.h"
#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_wgpu.h"
#include <SDL.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include <array>
#include <set>
#include <algorithm>
#include "json.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define SOKOL_FETCH_IMPL
#include "sokol_fetch.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

WGPUDevice device;
WGPUQueue queue;
WGPUSwapChain swapchain;

WGPURenderPipeline pipeline;
WGPUBuffer vertBuf; // vertex buffer with triangle position and colours
WGPUBuffer indxBuf; // index buffer
WGPUBuffer cameraBuf; // uniform buffer (containing the camera)
WGPUBuffer samplerBuf;
WGPUBuffer texBuf;
WGPUBindGroup bindGroup;

WGPUTexture depthBuffer;

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;



	static WGPUVertexBufferLayout getLayout() {
		std::array<WGPUVertexAttribute, 3> attributeDescriptions = {};

		attributeDescriptions[0].format = WGPUVertexFormat_Float32x3;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);
		attributeDescriptions[0].shaderLocation = 0;
		attributeDescriptions[1].format = WGPUVertexFormat_Float32x3;
		attributeDescriptions[1].offset = offsetof(Vertex, color);
		attributeDescriptions[1].shaderLocation = 1;
		attributeDescriptions[2].format = WGPUVertexFormat_Float32x2;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
		attributeDescriptions[2].shaderLocation = 2;

		
		WGPUVertexBufferLayout vertexBufferLayout = {};
		vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
		vertexBufferLayout.arrayStride = sizeof(Vertex);
		vertexBufferLayout.attributeCount = attributeDescriptions.size();
		vertexBufferLayout.attributes = attributeDescriptions.data();
		return vertexBufferLayout;
	}

	bool operator==(const Vertex& other) const {
			return pos == other.pos && color == other.color && texCoord == other.texCoord;
		}
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct Model {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

struct Resources {
	WGPUTexture texture1;
	Model model;
	int count = 2;
};

static Resources resources;

static Model loadModel(std::istream *inStream) {
	Model model; 
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err, inStream, NULL )) {
		return model;
	}

	std::unordered_map<Vertex, uint32_t> uniqueVertices = {};

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex = {};


			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.texCoord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.color = {1.0f, 1.0f, 1.0f};

			if (uniqueVertices.count(vertex) == 0) {
				uniqueVertices[vertex] = static_cast<uint32_t>(model.vertices.size());
				model.vertices.push_back(vertex);
			}

			model.indices.push_back(uniqueVertices[vertex]);
		}
	}
	return model;
}


static bool endsWith(std::string_view str, std::string_view suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

static bool startsWith(std::string_view str, std::string_view prefix)
{
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

/**
 * WGSL equivalent of \c triangle_vert_spirv.
 */
static char const triangle_vert_wgsl[] = R"(
	struct VertexIn {
		@location(0) aPos : vec2<f32>,
		@location(1) aCol : vec3<f32>
	};
	struct VertexOut {
		@location(0) vCol : vec3<f32>,
		@builtin(position) Position : vec4<f32>
	};

	struct Rotation {
		@location(0) degs : f32
	};
	@group(0)  @binding(0) var<uniform> uRot : Rotation;
	@stage(vertex)
	fn main(input : VertexIn) -> VertexOut {
		var rads : f32 = radians(uRot.degs);
		var cosA : f32 = cos(rads);
		var sinA : f32 = sin(rads);
		var rot : mat3x3<f32> = mat3x3<f32>(
			vec3<f32>( cosA, sinA, 0.0),
			vec3<f32>(-sinA, cosA, 0.0),
			vec3<f32>( 0.0,  0.0,  1.0));
		var output : VertexOut;
		output.Position = vec4<f32>(rot * vec3<f32>(input.aPos, 1.0), 1.0);
		output.vCol = input.aCol;
		return output;
	}
)";

static char const triangle_vert_wgsl_new[] = R"(
	struct VertexIn {
		@location(0) inPosition : vec3<f32>,
		@location(1) inColor : vec3<f32>,
		@location(2) inTexCoord : vec2<f32>
	};
	struct VertexOut {
		@location(0) fragColor : vec3<f32>,
		@location(1) fragTexCoord : vec2<f32>,
		@builtin(position) Position : vec4<f32>
	};
	struct Camera {
		model : mat4x4<f32>,
		view : mat4x4<f32>,
		proj : mat4x4<f32>
	};
	@group(0)  @binding(0) var<uniform> camera : Camera;

	@vertex
	fn main(input : VertexIn) -> VertexOut {
		var output : VertexOut;
 		output.Position = camera.proj * camera.view * camera.model * vec4<f32>(input.inPosition, 1.0);
    	output.fragColor = input.inColor;
    	output.fragTexCoord = input.inTexCoord;
		return output;
	}
)";

static char const triangle_frag_wgsl_new[] = R"(
	struct FragIn {
		@location(0) fragColor : vec3<f32>,
		@location(1) fragTexCoord : vec2<f32>,
	}

	@group(0)  @binding(1) var t: texture_2d<f32>;
	@group(0)  @binding(2) var s: sampler;

	@fragment
	fn main(input : FragIn) -> @location(0) vec4<f32> {
		return textureSample(t, s, input.fragTexCoord);
	}
)";


/**
 * WGSL equivalent of \c triangle_frag_spirv.
 */
static char const triangle_frag_wgsl[] = R"(
	@stage(fragment)
	fn main(@location(0) vCol : vec3<f32>) -> @location(0) vec4<f32> {
		return vec4<f32>(vCol, 1.0);
	}
)";

/**
 * Helper to create a shader from WGSL source.
 *
 * \param[in] code WGSL shader source
 * \param[in] label optional shader name
 */
static WGPUShaderModule createShader(const char* const code, const char* label = nullptr) {
	WGPUShaderModuleWGSLDescriptor wgsl = {};
	wgsl.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
	wgsl.source = code;
	WGPUShaderModuleDescriptor desc = {};
	desc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgsl);
	desc.label = label;
	return wgpuDeviceCreateShaderModule(device, &desc);
}

static WGPUTexture createTexture(const void* data, uint32_t width, uint32_t height) {
	int size_pp = 4; // guess 4 bytes per pixel
	WGPUTextureDescriptor tex = {};
	tex.dimension = WGPUTextureDimension_2D;
	tex.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
	tex.format = WGPUTextureFormat_RGBA8Unorm;
	tex.size = { .width = width, .height = height, .depthOrArrayLayers = 1};
	tex.mipLevelCount = 1;
	tex.sampleCount = 1;
	WGPUTexture texture = wgpuDeviceCreateTexture(device, &tex);
	WGPUImageCopyTexture copyTex = {};
	copyTex.texture = texture;
	copyTex.mipLevel = 0;
	copyTex.origin = { 0, 0, 0 };
	copyTex.aspect = WGPUTextureAspect_All;
	WGPUTextureDataLayout layout = {};
	layout.offset = 0;
	layout.bytesPerRow = width * size_pp;
	layout.rowsPerImage = height;
	WGPUExtent3D size = {width, height, 1 };
	wgpuQueueWriteTexture(queue, &copyTex, data, width * height * size_pp, &layout, &size);
	return texture;
}

static WGPUTexture createDepthBuffer() {
	WGPUTextureDescriptor tex = {};
	tex.dimension = WGPUTextureDimension_2D;
	tex.usage = WGPUTextureUsage_RenderAttachment;
	tex.format = WGPUTextureFormat_Depth32Float;
	tex.size = { .width = 1900, .height = 937, .depthOrArrayLayers = 1};
	tex.mipLevelCount = 1;
	tex.sampleCount = 1;
	return wgpuDeviceCreateTexture(device, &tex);
}

static WGPUBuffer createBuffer(size_t size, WGPUBufferUsage usage) {
	WGPUBufferDescriptor desc = {};
	desc.usage = WGPUBufferUsage_CopyDst | usage;
	desc.size  = size;
	WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &desc);
	return buffer;
}

static WGPUBuffer createBuffer(const void* data, size_t size, WGPUBufferUsage usage) {
	WGPUBuffer buffer = createBuffer(size, usage);
	wgpuQueueWriteBuffer(queue, buffer, 0, data, size);
	return buffer;
}

static WGPUSampler createSampler() {
	WGPUSamplerDescriptor desc = {};
	desc.minFilter = WGPUFilterMode_Linear;
	desc.magFilter = WGPUFilterMode_Linear;
	WGPUSampler sampler = wgpuDeviceCreateSampler(device, &desc);
	return sampler;
}

/**
 * Bare minimum pipeline to draw a triangle using the above shaders.
 */
static void createPipelineAndBuffers() {
	// compile shaders
	// NOTE: these are now the WGSL shaders (tested with Dawn and Chrome Canary)
	WGPUShaderModule vertMod = createShader(triangle_vert_wgsl_new);
	WGPUShaderModule fragMod = createShader(triangle_frag_wgsl_new);
	
	printf("checkpoint 1\n");

	depthBuffer = createDepthBuffer();

	WGPUBufferBindingLayout buf = {};
	buf.type = WGPUBufferBindingType_Uniform;


	// bind group layout (used by both the pipeline layout and uniform bind group, released at the end of this function)
	WGPUBindGroupLayoutEntry bglVertexEntry = {};
	bglVertexEntry.binding = 0;
	bglVertexEntry.visibility = WGPUShaderStage_Vertex;
	bglVertexEntry.buffer.type = WGPUBufferBindingType_Uniform;

	WGPUBindGroupLayoutEntry bglFragmentEntry = {};
	bglFragmentEntry.binding = 1;
	bglFragmentEntry.visibility = WGPUShaderStage_Fragment;
	bglFragmentEntry.texture.sampleType = WGPUTextureSampleType_Float;
	bglFragmentEntry.texture.viewDimension = WGPUTextureViewDimension_2D;

	WGPUBindGroupLayoutEntry sampleEntry = {};
	sampleEntry.binding = 2;
	sampleEntry.visibility = WGPUShaderStage_Fragment;
	sampleEntry.sampler.type = WGPUSamplerBindingType_Filtering;

	WGPUBindGroupLayoutEntry ents[3] = {bglVertexEntry, bglFragmentEntry, sampleEntry};
	WGPUBindGroupLayoutDescriptor bglDesc = {};
	bglDesc.entryCount = 3;
	bglDesc.entries = ents;
	WGPUBindGroupLayout bindGroupLayout = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

	// pipeline layout (used by the render pipeline, released after its creation)
	WGPUPipelineLayoutDescriptor layoutDesc = {};
	layoutDesc.bindGroupLayoutCount = 1;
	layoutDesc.bindGroupLayouts = &bindGroupLayout;
	WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &layoutDesc);

	printf("checkpoint 2\n");

	// Fragment state
	WGPUBlendState blend = {};
	blend.color.operation = WGPUBlendOperation_Add;
	blend.color.srcFactor = WGPUBlendFactor_One;
	blend.color.dstFactor = WGPUBlendFactor_Zero;
	blend.alpha.operation = WGPUBlendOperation_Add;
	blend.alpha.srcFactor = WGPUBlendFactor_One;
	blend.alpha.dstFactor = WGPUBlendFactor_Zero;

	WGPUColorTargetState colorTarget = {};
	colorTarget.format = webgpu::getSwapChainFormat(device);
	colorTarget.blend = &blend;
	colorTarget.writeMask = WGPUColorWriteMask_All;

	WGPUFragmentState fragment = {};
	fragment.module = fragMod;
	fragment.entryPoint = "main";
	fragment.targetCount = 1;
	fragment.targets = &colorTarget;

	WGPURenderPipelineDescriptor desc = {};
	desc.fragment = &fragment;

	WGPUDepthStencilState depthState = {};
	depthState.depthWriteEnabled = true;
	depthState.depthCompare = WGPUCompareFunction_Less;
	depthState.format = WGPUTextureFormat_Depth32Float;

	// Other state
	desc.layout = pipelineLayout;
	desc.depthStencil = &depthState;

	std::array<WGPUVertexAttribute, 3> attributeDescriptions = {};

	attributeDescriptions[0].format = WGPUVertexFormat_Float32x3;
	attributeDescriptions[0].offset = offsetof(Vertex, pos);
	attributeDescriptions[0].shaderLocation = 0;
	attributeDescriptions[1].format = WGPUVertexFormat_Float32x3;
	attributeDescriptions[1].offset = offsetof(Vertex, color);
	attributeDescriptions[1].shaderLocation = 1;
	attributeDescriptions[2].format = WGPUVertexFormat_Float32x2;
	attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
	attributeDescriptions[2].shaderLocation = 2;


	WGPUVertexBufferLayout vertexBufferLayout = {};
	vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;
	vertexBufferLayout.arrayStride = sizeof(Vertex);
	vertexBufferLayout.attributeCount = attributeDescriptions.size();
	vertexBufferLayout.attributes = attributeDescriptions.data();

	desc.vertex.module = vertMod;
	desc.vertex.entryPoint = "main";
	desc.vertex.bufferCount = 1;
	desc.vertex.buffers = &vertexBufferLayout;

	desc.multisample.count = 1;
	desc.multisample.mask = 0xFFFFFFFF;
	desc.multisample.alphaToCoverageEnabled = false;

	desc.primitive.frontFace = WGPUFrontFace_CCW;
	desc.primitive.cullMode = WGPUCullMode_None;
	desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
	desc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;

	pipeline = wgpuDeviceCreateRenderPipeline(device, &desc);

	printf("checkpoint 3\n");

	// partial clean-up (just move to the end, no?)
	wgpuPipelineLayoutRelease(pipelineLayout);

	wgpuShaderModuleRelease(fragMod);
	wgpuShaderModuleRelease(vertMod);

	auto verts = resources.model.vertices;
	auto indices = resources.model.indices;

	auto vertBufferSize = sizeof(verts[0]) * verts.size();
	auto indicesBufferSize = sizeof(indices[0]) * indices.size();

	vertBuf = createBuffer(verts.data(), vertBufferSize, WGPUBufferUsage_Vertex);
	indxBuf = createBuffer(indices.data(), indicesBufferSize, WGPUBufferUsage_Index);


	// create the uniform bind group 
	auto cameraSize = sizeof(struct UniformBufferObject);
	cameraBuf = createBuffer(cameraSize, WGPUBufferUsage_Uniform);
	
	WGPUTextureViewDescriptor viewDesc = {};
	viewDesc.aspect = WGPUTextureAspect_All;
	viewDesc.format = WGPUTextureFormat_RGBA8Unorm;
	viewDesc.dimension = WGPUTextureViewDimension_2D;
	viewDesc.baseMipLevel = 0;
	viewDesc.mipLevelCount = 1;
	viewDesc.baseArrayLayer = 0;
	viewDesc.arrayLayerCount = 1;

	WGPUTextureView view = wgpuTextureCreateView(resources.texture1, &viewDesc);

	WGPUSamplerDescriptor samplerDesc = {};
	samplerDesc.minFilter = WGPUFilterMode_Linear;
	samplerDesc.magFilter = WGPUFilterMode_Linear;
	WGPUSampler sampler = wgpuDeviceCreateSampler(device, &samplerDesc);

	WGPUBindGroupEntry bgEntry = {};
	bgEntry.binding = 0;
	bgEntry.buffer = cameraBuf;
	bgEntry.offset = 0;
	bgEntry.size = cameraSize;

	WGPUBindGroupEntry bgEntry2 = {};
	bgEntry2.binding = 1;
	bgEntry2.textureView = view;

	WGPUBindGroupEntry bgEntry3 = {};
	bgEntry3.binding = 2;
	bgEntry3.sampler = sampler;

	WGPUBindGroupEntry bgEnts[3] = {bgEntry, bgEntry2, bgEntry3};
	WGPUBindGroupDescriptor bgDesc = {};
	bgDesc.layout = bindGroupLayout;
	bgDesc.entryCount = 3;
	bgDesc.entries = bgEnts;

	bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

	// last bit of clean-up
	wgpuBindGroupLayoutRelease(bindGroupLayout);
}

static void updateCamera() {
	static auto startTime = std::chrono::high_resolution_clock::now();

	auto currentTime = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

	UniformBufferObject ubo = {};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), 1900 / (float) 937, 0.1f, 10.0f);
	//ubo.proj[1][1] *= -1;

	wgpuQueueWriteBuffer(queue, cameraBuf, 0, &ubo, sizeof(ubo));
}

/**
 * Draws using the above pipeline and buffers.
 */

static bool show_demo_window = true;
static bool redraw() {

	sfetch_dowork();

 	SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL2_ProcessEvent(&event);	
	}

	ImGui_ImplWGPU_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();
  	ImGui::ShowDemoWindow(&show_demo_window);
	ImGui::Render();

	WGPUTextureView backBufView = wgpuSwapChainGetCurrentTextureView(swapchain);
	WGPURenderPassColorAttachment colorDesc = {};
	colorDesc.view    = backBufView;
	colorDesc.loadOp  = WGPULoadOp_Clear;
	colorDesc.storeOp = WGPUStoreOp_Store;


	WGPUTextureViewDescriptor viewDesc = {};
	viewDesc.aspect = WGPUTextureAspect_DepthOnly;
	viewDesc.format = WGPUTextureFormat_Depth32Float;
	viewDesc.dimension = WGPUTextureViewDimension_2D;
	viewDesc.baseMipLevel = 0;
	viewDesc.mipLevelCount = 1;
	viewDesc.baseArrayLayer = 0;
	viewDesc.arrayLayerCount = 1;
	WGPUTextureView depthView = wgpuTextureCreateView(depthBuffer, &viewDesc);

	WGPURenderPassDepthStencilAttachment depthDesc = {};
	depthDesc.view = depthView;
	depthDesc.depthClearValue = 1.0;
	depthDesc.depthLoadOp = WGPULoadOp_Clear;
	depthDesc.depthStoreOp = WGPUStoreOp_Store;
	depthDesc.stencilLoadOp = WGPULoadOp_Undefined;
	depthDesc.stencilStoreOp = WGPUStoreOp_Undefined;

	WGPURenderPassDescriptor renderPass = {};
	renderPass.colorAttachmentCount = 1;
	renderPass.colorAttachments = &colorDesc;
	renderPass.depthStencilAttachment = &depthDesc;

	WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, nullptr);			// create encoder
	WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPass);	// create pass

	updateCamera();

	wgpuRenderPassEncoderSetPipeline(pass, pipeline);
	wgpuRenderPassEncoderSetBindGroup(pass, 0, bindGroup, 0, 0);
	wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertBuf, 0, WGPU_WHOLE_SIZE);
	wgpuRenderPassEncoderSetIndexBuffer(pass, indxBuf, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
	wgpuRenderPassEncoderDrawIndexed(pass, resources.model.indices.size(), 1, 0, 0, 0);

	ImGui_ImplWGPU_RenderDrawData(ImGui::GetDrawData(), pass);
	wgpuRenderPassEncoderEnd(pass);
	WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);				// create commands

	wgpuQueueSubmit(queue, 1, &commands);
	wgpuRenderPassEncoderRelease(pass);														// release pass
	wgpuCommandEncoderRelease(encoder);														// release encoder
	wgpuCommandBufferRelease(commands);														// release commands
#ifndef __EMSCRIPTEN__
	/*
	 * TODO: wgpuSwapChainPresent is unsupported in Emscripten, so what do we do?
	 */
	wgpuSwapChainPresent(swapchain);
#endif
	wgpuTextureViewRelease(backBufView);													// release textureView

	return true;
}

void response_callback(const sfetch_response_t* response) {
	if (response->fetched) {
		// data has been loaded, and is available via
		// 'buffer_ptr' and 'fetched_size':
		void* data = response->buffer_ptr;
		uint64_t num_bytes = response->fetched_size;
		std::string path(response->path);
		if(endsWith(path, std::string("obj"))) {
			std::stringstream ss;
			ss.rdbuf()->sputn(static_cast<char*>(data), num_bytes);
		 	resources.model = loadModel(&ss);
		}
		if(endsWith(path, std::string("png"))) {
			int texWidth, texHeight, texChannels;
			stbi_uc* pixels = stbi_load_from_memory(static_cast<unsigned char*>(data), num_bytes, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
			resources.texture1 = createTexture(pixels, texWidth, texHeight);
		}
		resources.count--;
	}
	if (response->finished) {
			printf("Done\n");
		if (response->failed) {
			printf("Failed\n");
		}
	}
}

// 937 * 1920 
extern "C" int __main__(int /*argc*/, char* /*argv*/[]) {
	if (window::Handle wHnd = window::create()) {

		const sfetch_desc_t &a = {};
		sfetch_setup(a);

		static uint8_t buf[1024 * 1024 * 50];

		const sfetch_request_t &req = {
            .path = "data/viking_room.obj",
            .callback = response_callback,
            .buffer_ptr = buf,
            .buffer_size = sizeof(buf)
        };

        sfetch_send(req);

		static uint8_t buf2[1024 * 1024 * 3];

		const sfetch_request_t &req2 = {
            .path = "data/viking_room.png",
            .callback = response_callback,
            .buffer_ptr = buf2,
            .buffer_size = sizeof(buf2)
        };

        sfetch_send(req2);

		SDL_Init(SDL_INIT_NOPARACHUTE);
		SDL_Window *window = SDL_CreateWindow("Egal", 0, 0, 1900, 937, 0);

		IMGUI_CHECKVERSION();
    	ImGui::CreateContext();
    	ImGuiIO& io = ImGui::GetIO(); 
		io.IniFilename = NULL;
		ImGui::StyleColorsDark();

		ImGui_ImplSDL2_InitForSDLRenderer(window, NULL);
		if ((device = webgpu::create(wHnd))) {
			queue = wgpuDeviceGetQueue(device);
			swapchain = webgpu::createSwapChain(device);
    		ImGui_ImplWGPU_Init(device, 3, WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_Depth32Float);
			ImGui_ImplWGPU_CreateDeviceObjects();
			while(resources.count > 0) {
				sfetch_dowork();
				emscripten_sleep(100);
			}
			createPipelineAndBuffers();
			window::show(wHnd);
			window::loop(wHnd, redraw);

		#ifndef __EMSCRIPTEN__
			wgpuBindGroupRelease(bindGroup);
			wgpuBufferRelease(uRotBuf);
			wgpuBufferRelease(indxBuf);
			wgpuBufferRelease(vertBuf);
			wgpuRenderPipelineRelease(pipeline);
			wgpuSwapChainRelease(swapchain);
			wgpuQueueRelease(queue);
			wgpuDeviceRelease(device);
		#endif
		}
	#ifndef __EMSCRIPTEN__
		window::destroy(wHnd);
	#endif
	}
	return 0;
}
