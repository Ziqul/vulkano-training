use std::error::Error;
use std::sync::Arc;

use image::{ImageBuffer, Rgba};
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::ClearValue;
use vulkano::format::Format;
use vulkano::image::Dimensions;
use vulkano::image::StorageImage;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;


fn main() -> Result<(), Box<Error>> {

    let (instance, device, queue) = init()?;

    // Shading stuff [START]
    mod cs {
        vulkano_shaders::shader!{
            ty: "compute",
            src: "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

void main() {
    vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
    vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

    vec2 z = vec2(0.0, 0.0);
    float i;
    for (i = 0.0; i < 1.0; i += 0.005) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 4.0) {
            break;
        }
    }

    vec4 to_write = vec4(vec3(i), 1.0);
    imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
}"
        }
    }

    let shader = cs::Shader::load(device.clone())?;

    let compute_pipeline =
        Arc::new(
            ComputePipeline::new(
                device.clone(), &shader.main_entry_point(), &()
            )?
        );
    // Shading stuff [END]

    let image =
        StorageImage::new(
            device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
            Format::R8G8B8A8Unorm, Some(queue.family())
        )?;

    let set =
        Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline.clone(), 0
            ).add_image(image.clone())?.build()?
        );

    let buf =
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(),
            (0 .. 1024 * 1024 * 4).map(|_| 0u8)
        )?;

    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
            .dispatch([1024 / 8, 1024 / 8, 1], compute_pipeline.clone(), set.clone(), ())?
            .copy_image_to_buffer(image.clone(), buf.clone())?
            .build()?;

    let finished = command_buffer.execute(queue.clone())?;
    finished.then_signal_fence_and_flush()?.wait(None)?;

    let buffer_content = buf.read()?;
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("image.png")?;

    Ok(())
}

fn init() -> Result<(Arc<Instance>, Arc<Device>, Arc<Queue>), Box<Error>> {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)?;

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

    let mut chosen_extensions = DeviceExtensions::none();
    // // This field is required in 0.16.0 version of vulkano
    // chosen_extensions.khr_storage_buffer_storage_class = true;

    let (chosen_logical_device, mut queues) = {
        Device::new(
            chosen_physical_device, &Features::none(),
            &chosen_extensions,
            [(chosen_family, 0.5)].iter().cloned()
        )?
    };

    let chosen_queue = queues.next()
        .expect("Error: NoneError: No queue found in chosen family");

    Ok((instance, chosen_logical_device, chosen_queue))
}
