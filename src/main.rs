// Build-in modules
use std::error::Error;
use std::sync::Arc;

// External modules
use image::{ImageBuffer, Rgba};
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::ClearValue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::Subpass;
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::Capabilities;
use vulkano::swapchain::Surface;
use vulkano::swapchain::{Swapchain, SurfaceTransform, PresentMode};
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;
use winit::EventsLoop;
use winit::Window;
use winit::WindowBuilder;

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

fn main() -> Result<(), Box<Error>> {
    let (
        instance, device, queue,
        surface, capabilities, mut events_loop
    ) = init()?;


    let image =
        StorageImage::new(
            device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
            Format::R8G8B8A8Unorm, Some(queue.family())
        )?;

    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(),
            vec![vertex1, vertex2, vertex3].into_iter()
        )?;

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}"
        }
    }

    let vs = vs::Shader::load(device.clone())?;
    let fs = fs::Shader::load(device.clone())?;

    let render_pass =
        Arc::new(
            vulkano::single_pass_renderpass!(
                device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: Format::R8G8B8A8Unorm,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
                }
            )?
        );

    let pipeline =
        Arc::new(
            GraphicsPipeline::start()
                // Defines what kind of vertex input is expected.
                .vertex_input_single_buffer::<Vertex>()
                // The vertex shader.
                .vertex_shader(vs.main_entry_point(), ())
                // Defines the viewport.
                .viewports_dynamic_scissors_irrelevant(1)
                // The fragment shader.
                .fragment_shader(fs.main_entry_point(), ())
                // This graphics pipeline object concerns the first pass of the render pass.
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                // Now that everything is specified, we call `build`.
                .build(device.clone())?
        );

    let framebuffer =
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone())?
                .build()?
        );

    let buf =
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(),
            (0 .. 1024 * 1024 * 4).map(|_| 0u8)
        )?;

    let dynamic_state =
        DynamicState {
            viewports: Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [1024.0, 1024.0],
                depth_range: 0.0 .. 1.0,
            }]),
            .. DynamicState::none()
        };

    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(
            device.clone(), queue.family()
        )?
            .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into()])?
            .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())?
            .end_render_pass()?
            .copy_image_to_buffer(image.clone(), buf.clone())?
            .build()?;;

    let finished = command_buffer.execute(queue.clone())?;
    finished.then_signal_fence_and_flush()?.wait(None)?;

    let buffer_content = buf.read()?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png")?;

    Ok(())
}

fn init() ->
    Result<
        (
            Arc<Instance>, Arc<Device>, Arc<Queue>,
            Arc<Surface<Window>>, Capabilities, EventsLoop
        ),
        Box<Error>
    >
{
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None)?
    };

    #[cfg(debug_assertions)]
    {
        println!("Listing available devices supporting Vulkan API: ");
        for device in PhysicalDevice::enumerate(&instance) {
            println!("{:?}: {:?}", device.name(), device);

            print!("Device contains queue families with this queue(s) amount: ");
            for family in device.queue_families() {
                print!("{:?} ", family.queues_count());
            }

            println!("---");
        }

        println!("");
    }

    let chosen_physical_device =
        PhysicalDevice::enumerate(&instance).nth(0)
            .expect("Error: NoneError: No physical devices supporting Vulkan API found");

    #[cfg(debug_assertions)]
    {
        println!(
            "Chosen device: {:?}: {:?}",
            chosen_physical_device.name(),
            chosen_physical_device);

        println!("");
    }

    let chosen_family = chosen_physical_device.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("Error: NoneError: No family supporting GRAPHICS_BIT found in chosen device");

    let (chosen_logical_device, mut queues) = {
        let mut chosen_extensions = DeviceExtensions::none();
        // // "khr_storage_buffer_storage_class" is required in vulkano="0.16.0"
        // chosen_extensions.khr_storage_buffer_storage_class = true;
        chosen_extensions.khr_swapchain = true;

        Device::new(
            chosen_physical_device,
            chosen_physical_device.supported_features(),
            &chosen_extensions,
            [(chosen_family, 0.5)].iter().cloned()
        )?
    };

    let chosen_queue = queues.next()
        .expect("Error: NoneError: No queue found in chosen family");


    let mut events_loop = EventsLoop::new();
    let surface =
        WindowBuilder::new().build_vk_surface(
            &events_loop, instance.clone()
        )?;

    let capabilities = surface.capabilities(chosen_physical_device)?;

    Ok((
        instance, chosen_logical_device, chosen_queue,
        surface, capabilities, events_loop
    ))
}
