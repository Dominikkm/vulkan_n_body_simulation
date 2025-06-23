use vulkano::{
  buffer::{
    Buffer,
    BufferContents,
    BufferCreateInfo,
    BufferUsage,
    Subbuffer,
  },
  command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder,
    CommandBufferUsage,
    CopyBufferInfo,
    PrimaryCommandBufferAbstract,
    RenderPassBeginInfo,
    SubpassBeginInfo,
    SubpassContents
  },
  descriptor_set::{
    allocator::StandardDescriptorSetAllocator,
    DescriptorSet,
    WriteDescriptorSet
  },
  device::{
    physical::PhysicalDeviceType,
    Device,
    DeviceCreateInfo,
    DeviceExtensions,
    Queue,
    QueueCreateInfo,
    QueueFlags,
  },
  image::{view::ImageView, Image, ImageUsage},
  instance::{
    Instance,
    InstanceCreateFlags,
    InstanceCreateInfo,
  },
  memory::allocator::{
    AllocationCreateInfo,
    MemoryTypeFilter,
    StandardMemoryAllocator,
  },
  pipeline::{
    compute::ComputePipelineCreateInfo,
    graphics::{
      color_blend::{ColorBlendAttachmentState, ColorBlendState},
      input_assembly::{InputAssemblyState, PrimitiveTopology},
      multisample::MultisampleState,
      rasterization::RasterizationState,
      vertex_input::{Vertex, VertexDefinition},
      viewport::{Viewport, ViewportState},
      GraphicsPipelineCreateInfo,
    },
    layout::PipelineDescriptorSetLayoutCreateInfo,
    ComputePipeline,
    DynamicState,
    GraphicsPipeline,
    Pipeline,
    PipelineBindPoint,
    PipelineLayout,
    PipelineShaderStageCreateInfo,
  },
  render_pass::{
    Framebuffer,
    FramebufferCreateInfo,
    RenderPass,
    Subpass,
  },
  shader::SpecializationConstant,
  swapchain::{
    acquire_next_image,
    Surface,
    Swapchain,
    SwapchainCreateInfo,
    SwapchainPresentInfo,
  },
  sync::{self, GpuFuture},
  DeviceSize,
  Validated,
  VulkanError,
  VulkanLibrary,
};
use winit::{
  application::ApplicationHandler,
  event::{ElementState, KeyEvent, WindowEvent},
  event_loop::{ActiveEventLoop, EventLoop},
  window::{Window, WindowId},
  keyboard::{Key, NamedKey},
};
use foldhash::HashMap;

use std::{sync::Arc, time::SystemTime};

use crate::galaxy;


const PARTICLES: usize = 128 * 200;


pub struct App {
  instance: Arc<Instance>,
  device: Arc<Device>,
  queue: Arc<Queue>,
  command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
  vertex_buffer: Subbuffer<[MyVertex]>,
  vertex_buffer_new: Subbuffer<[MyVertex]>,
  compute_pipeline: Arc<ComputePipeline>,
  descriptor_set: Arc<DescriptorSet>,
  rcx: Option<RenderContext>,
  paused: bool,
}

struct RenderContext {
  window: Arc<Window>,
  swapchain: Arc<Swapchain>,
  recreate_swapchain: bool,
  previous_frame_end: Option<Box<dyn GpuFuture>>,
  render_pass: Arc<RenderPass>,
  framebuffers: Vec<Arc<Framebuffer>>,
  pipeline: Arc<GraphicsPipeline>,
  viewport: Viewport,
  last_frame_time: SystemTime,
}


impl App {
  pub fn new(event_loop: &EventLoop<()>) -> Self {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(event_loop).unwrap();
    let instance = Instance::new(
      library,
      InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
      },
    )
      .unwrap();
    let device_extensions = DeviceExtensions {
      khr_swapchain: true,
      ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
      .enumerate_physical_devices()
      .unwrap()
      .filter(|p| {
        p.supported_extensions().contains(&device_extensions)
      })
      .filter_map(|p| {
        p.queue_family_properties()
          .iter()
          .enumerate()
          .position(|(i, q)| {
            q.queue_flags.intersects(QueueFlags::GRAPHICS)
            && p.presentation_support(i as u32, event_loop).unwrap()
          })
          .map(|i| (p, i as u32))
      })
      .max_by_key(|(p, _)| {
        match p.properties().device_type {
          PhysicalDeviceType::DiscreteGpu => 0,
          PhysicalDeviceType::IntegratedGpu => 1,
          PhysicalDeviceType::VirtualGpu => 2,
          PhysicalDeviceType::Cpu => 3,
          PhysicalDeviceType::Other => 4,
          _ => 5,
        }
      })
      .expect("No suitable physical device found");

    println!( "Using device: {} (type: {:?})",
      physical_device.properties().device_name,
      physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
      physical_device,
      DeviceCreateInfo {
        enabled_extensions: device_extensions,
        queue_create_infos: vec![QueueCreateInfo {
          queue_family_index,
          ..Default::default()
        }],
        ..Default::default()
      },
    )
      .unwrap();
    let queue = queues.next().unwrap();
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
      device.clone(),
      Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
      device.clone(),
      Default::default(),
    ));

    let vertices = {
      let mut vertices_0 = galaxy::generate(
        PARTICLES / 2,
        5,
        5.0,
        0.6,
        0.1,
        [0.9, 0.9],
        [0.00, 0.00],
        1E8,
      );
      let mut vertices_1 = galaxy::generate(
        PARTICLES / 2,
        3,
        3.0,
        0.6,
        0.1,
        [-0.9, -0.9],
        [-0.00, -0.00],
        1E8,
      );
      vertices_0.append(&mut vertices_1);

      vertices_0
    };


    let temporary_accesible_buffer = Buffer::from_iter(
      memory_allocator.clone(),
      BufferCreateInfo {
        usage: BufferUsage::TRANSFER_SRC,
        ..Default::default()
      },
      AllocationCreateInfo {
        memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        ..Default::default()
      },
      vertices,
    )
      .unwrap();

    let vertex_buffer = {
      let device_local_buffer = Buffer::new_slice::<MyVertex>(
        memory_allocator.clone(),
        BufferCreateInfo {
          usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::VERTEX_BUFFER,
          ..Default::default()
        },
        AllocationCreateInfo {
          memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
          ..Default::default()
        },
        PARTICLES as DeviceSize,
      )
        .unwrap();

      let mut cbb = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
      )
        .unwrap();
      cbb.copy_buffer(CopyBufferInfo::buffers(
        temporary_accesible_buffer.clone(), device_local_buffer.clone()
      ))
        .unwrap();
      let cb = cbb.build().unwrap();

      cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

      device_local_buffer
    };

    let vertex_buffer_new = {
      let device_local_buffer = Buffer::new_slice::<MyVertex>(
        memory_allocator,
        BufferCreateInfo {
          usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
          ..Default::default()
        },
        AllocationCreateInfo {
          memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
          ..Default::default()
        },
        PARTICLES as DeviceSize
      )
        .unwrap();

      let mut cbb = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
      )
        .unwrap();
      cbb.copy_buffer(CopyBufferInfo::buffers(
        temporary_accesible_buffer.clone(), device_local_buffer.clone()
      ))
        .unwrap();
      let cb = cbb.build().unwrap();

      cb.execute(queue.clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

      device_local_buffer
    };

    let compute_pipeline = {
      let cs = cs::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
      let stage = PipelineShaderStageCreateInfo::new(cs);
      let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
          .into_pipeline_layout_create_info(device.clone())
          .unwrap(),
      )
        .unwrap();

      ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
      )
        .unwrap()
    };

    let descriptor_set = DescriptorSet::new(
      descriptor_set_allocator.clone(),
      compute_pipeline.layout().set_layouts()[0].clone(),
      [
        WriteDescriptorSet::buffer(0, vertex_buffer.clone()),
        WriteDescriptorSet::buffer(1, vertex_buffer_new.clone()),
      ],
      [],
    )
      .unwrap();

    let rcx = None;
    let paused = false;

    Self {
      instance,
      device,
      queue,
      command_buffer_allocator,
      vertex_buffer,
      vertex_buffer_new,
      compute_pipeline,
      descriptor_set,
      rcx,
      paused,
    }
  }
}

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    let window = Arc::new(
      event_loop
        .create_window(Window::default_attributes())
        .unwrap(),
    );
    let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();
    let window_size = window.inner_size();

    let (swapchain, images) = {
      let surface_capabilities = self
        .device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();
      let (image_format, _) = self
        .device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .unwrap()[0];

      Swapchain::new(
        self.device.clone(),
        surface,
        SwapchainCreateInfo {
          min_image_count: surface_capabilities.min_image_count.max(2),
          image_format,
          image_extent: window_size.into(),
          image_usage: ImageUsage::COLOR_ATTACHMENT,
          composite_alpha: surface_capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .unwrap(),
          ..Default::default()
        },
      )
        .unwrap()
    };

    mod vs {
      vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
          #version 450

          layout(constant_id = 0) const float point = 1.0;
          layout(constant_id = 1) const float minMass = 0.9E5;
          layout(constant_id = 2) const float maxMass = 2.25E5;

          layout(location = 0) out vec3 fragColor;
          layout(location = 0) in vec2 pos;
          layout(location = 1) in vec2 vel;
          layout(location = 2) in float mass;
          layout(location = 3) in float _pad;

          void main() {
            gl_Position = vec4(pos, 0.0, 1.0);
            gl_PointSize = point;

            float norm = clamp((log(mass) - log(minMass)) / (log(maxMass) - log(minMass)), 0.0, 1.0);

            vec3 color;
            if (norm < 0.25) {
              // 0.0 - 0.25: white -> blue
              float t = norm / 0.25;
              color = mix(vec3(1.0), vec3(0.2, 0.4, 1.0), t);
            } else if (norm < 0.5) {
              // 0.25 - 0.5: blue -> green
              float t = (norm - 0.25) / 0.25;
              color = mix(vec3(0.2, 0.4, 1.0), vec3(0.2, 1.0, 0.2), t);
            } else if (norm < 0.75) {
              // 0.5 - 0.75: green -> red
              float t = (norm - 0.5) / 0.25;
              color = mix(vec3(0.2, 1.0, 0.2), vec3(1.0, 0.2, 0.2), t);
            } else {
              // 0.75 - 1.0: red -> black
              float t = (norm - 0.75) / 0.25;
              color = mix(vec3(1.0, 0.2, 0.2), vec3(0.0), t);
            }

            fragColor = color;
          }
        ",
      }
    }

    mod fs {
      vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
          #version 450

          layout(location = 0) in vec3 fragColor;
          layout(location = 0) out vec4 f_color;

          void main() {
            f_color = vec4(fragColor, 1.0);
          }
        ",
      }
    }

    let render_pass = vulkano::single_pass_renderpass!(
      self.device.clone(),
      attachments: {
        color: {
          format: swapchain.image_format(),
          samples: 1,
          load_op: Clear,
          store_op: Store,
        },
      },
      pass: {
        color: [color],
        depth_stencil: {},
      },
    )
      .unwrap();

    let framebuffers = window_size_dependent_setup(&images, &render_pass);

    let pipeline = {
      let raw_vs = vs::load(self.device.clone()).unwrap();
      let mut specialization_info = HashMap::default();

      specialization_info.insert(0, SpecializationConstant::F32(1.0));
      specialization_info.insert(1, SpecializationConstant::F32(0.9E5));
      specialization_info.insert(2, SpecializationConstant::F32(2.25E5));

      let specialization = raw_vs.specialize(specialization_info).unwrap();
      let vs = specialization.entry_point("main").unwrap();
      let fs = fs::load(self.device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
      let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
      let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
      ];
      let layout = PipelineLayout::new(
        self.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
          .into_pipeline_layout_create_info(self.device.clone())
          .unwrap(),
      )
        .unwrap();

      let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

      GraphicsPipeline::new(
        self.device.clone(),
        None,
        GraphicsPipelineCreateInfo {
          stages: stages.into_iter().collect(),
          vertex_input_state: Some(vertex_input_state),
          input_assembly_state: Some(InputAssemblyState {
            topology: PrimitiveTopology::PointList,
            ..Default::default()
          }),
          viewport_state: Some(ViewportState::default()),
          rasterization_state: Some(RasterizationState::default()),
          multisample_state: Some(MultisampleState::default()),
          color_blend_state: Some(ColorBlendState::with_attachment_states(
            subpass.num_color_attachments(),
            ColorBlendAttachmentState::default(),
          )),
          dynamic_state: [DynamicState::Viewport].into_iter().collect(),
          subpass: Some(subpass.into()),
          ..GraphicsPipelineCreateInfo::layout(layout)
        },
      )
        .unwrap()
    };

    let viewport = Viewport {
      offset: [0.0, 0.0],
      extent: window_size.into(),
      depth_range: 0.0 ..= 1.0,
    };
    let recreate_swapchain = false;
    let previous_frame_end = Some(sync::now(self.device.clone()).boxed());
    let last_frame_time = SystemTime::now();

    self.rcx = Some(RenderContext {
      window,
      swapchain,
      recreate_swapchain,
      previous_frame_end,
      render_pass,
      framebuffers,
      pipeline,
      viewport,
      last_frame_time,
    });
  }

  fn window_event(
    &mut self,
    event_loop: &ActiveEventLoop,
    _window_id: WindowId,
    event: WindowEvent,
  ) {
    let rcx = self.rcx.as_mut().unwrap();

    match event {
      WindowEvent::CloseRequested => event_loop.exit(),
      WindowEvent::Resized(_) => {
        rcx.recreate_swapchain = true;
      },
      WindowEvent::RedrawRequested => {
        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
          return;
        }
        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let now = SystemTime::now();
        let delta_time = if !self.paused {
          now
            .duration_since(rcx.last_frame_time)
            .unwrap()
            .as_secs_f32()
        } else {
          0.0
        };
        rcx.last_frame_time = now;
        let push_constants = cs::PushConstants {
          dt: delta_time,
          count: PARTICLES as u32,
        };

        if rcx.recreate_swapchain {
          let (new_swapchain, new_images) = rcx
            .swapchain
            .recreate(SwapchainCreateInfo {
              image_extent: window_size.into(),
              ..rcx.swapchain.create_info()
            })
            .expect("Failed to create swapchain");
          rcx.swapchain = new_swapchain;
          rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass);
          rcx.viewport.extent = window_size.into();
          rcx.recreate_swapchain = false;
        }

        let (image_indedx, suboptimal, acquire_future) = match acquire_next_image(
          rcx.swapchain.clone(),
          None,
        )
          .map_err(Validated::unwrap)
        {
          Ok(r) => r,
          Err(VulkanError::OutOfDate) => {
            rcx.recreate_swapchain = true;
            return;
          }
          Err(e) => panic!("failed to acquire next image: {}", e),
        };

        if suboptimal {
          rcx.recreate_swapchain = true;
        }


        let mut builder = AutoCommandBufferBuilder::primary(
          self.command_buffer_allocator.clone(),
          self.queue.queue_family_index(),
          CommandBufferUsage::OneTimeSubmit,
        )
          .unwrap();

        builder
          .push_constants(self.compute_pipeline.layout().clone(), 0, push_constants)
          .unwrap()
          .bind_pipeline_compute(self.compute_pipeline.clone())
          .unwrap()
          .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.compute_pipeline.layout().clone(),
            0,
            self.descriptor_set.clone(),
          )
            .unwrap();

        // TODO: Fix division
        unsafe { builder.dispatch([(PARTICLES / 128) as u32, 1, 1]) }.unwrap();

        builder.copy_buffer(CopyBufferInfo::buffers(
          self.vertex_buffer_new.clone(),
          self.vertex_buffer.clone(),
        ))
          .unwrap();

        builder
          .begin_render_pass(
            RenderPassBeginInfo {
              clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
              ..RenderPassBeginInfo::framebuffer(
                rcx.framebuffers[image_indedx as usize].clone(),
              )
            },
            SubpassBeginInfo {
              contents: SubpassContents::Inline,
              ..Default::default()
            },
          )
          .unwrap()
          .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
          .unwrap()
          .bind_pipeline_graphics(rcx.pipeline.clone())
          .unwrap()
          .bind_vertex_buffers(0, self.vertex_buffer.clone())
          .unwrap();

        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

        builder
          .end_render_pass(Default::default())
          .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = rcx
          .previous_frame_end
          .take()
          .unwrap()
          .join(acquire_future)
          .then_execute(self.queue.clone(), command_buffer)
          .unwrap()
          .then_swapchain_present(
            self.queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
              rcx.swapchain.clone(),
              image_indedx,
            ),
          )
          .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
          Ok(future) => {
            rcx.previous_frame_end = Some(future.boxed());
          }
          Err(VulkanError::OutOfDate) => {
            rcx.recreate_swapchain = true;
            rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
          }
          Err(e) => {
            panic!("Failed to flush future: {}", e);
          }
        }
      }

      WindowEvent::KeyboardInput {
        event: KeyEvent {
          logical_key: Key::Named(NamedKey::Space),
          state: ElementState::Pressed,
          ..
        }
        ,
        ..
      } => {
        self.paused = !self.paused;
      }
      _ => {}
    }
  }

  fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    let rcx = self.rcx.as_mut().unwrap();

    rcx.window.request_redraw();
  }
}


#[derive(BufferContents, Vertex, Debug)]
#[repr(C)]
pub struct MyVertex {
  /// Position
  #[format(R32G32_SFLOAT)]
  pos: [f32; 2],
  /// Velocity
  #[format(R32G32_SFLOAT)]
  vel: [f32; 2],
  // Mass
  #[format(R32_SFLOAT)]
  mass: f32,
  #[format(R32_SFLOAT)]
  _pad: f32,
}

impl MyVertex {
  pub fn new(pos: [f32; 2], vel: [f32; 2], mass: f32) -> Self {
    MyVertex {
      pos,
      vel,
      mass,
      _pad: 0.0,
    }
  }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
  images: &[Arc<Image>],
  render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
  images
    .iter()
    .map(|image| {
      let view = ImageView::new_default(image.clone()).unwrap();
      Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
          attachments: vec![view],
          ..Default::default()
        },
      )
        .unwrap()
    })
    .collect::<Vec<_>>()
}

mod cs {
  vulkano_shaders::shader! {
    ty: "compute",
    src: r"
        #version 450

        const float G = 6.67408E-11;

        layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

        struct Particle {
            vec2 pos;
            vec2 vel;
            float mass;
        };

        layout(set = 0, binding = 0) buffer InParticles {
            Particle particles[];
        } inBuf;

        layout(set = 0, binding = 1) buffer OutParticles {
            Particle particles[];
        } outBuf;

        layout (push_constant) uniform PushConstants {
          float dt;
          uint count;
        } push;

        float length2(vec2 v) {
          return v.x * v.x + v.y * v.y;
        }

        void main() {
          uint i = gl_GlobalInvocationID.x;
          Particle p = inBuf.particles[i];

          vec2 acc = vec2(0.0);

          for (uint j = 0; j < push.count; ++j) {
            if (i == j) continue;

            
            vec2 diff = inBuf.particles[j].pos - p.pos;
            float dist = dot(diff, diff);
            // normalize(0.0) is undefined
            if (dist > 1E-8) {
              acc += normalize(diff) * inBuf.particles[j].mass / (dist + 0.1);
            }
          }

          outBuf.particles[i].vel += vec2(acc * G * push.dt * 0.1);
          outBuf.particles[i].pos += outBuf.particles[i].vel * push.dt;
        }
    ",
  }
}

