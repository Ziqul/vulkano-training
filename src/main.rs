use std::error::Error;
use std::sync::Arc;

use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;

fn main() -> Result<(), Box<Error>> {

    let (instance, device, queue) = init()?;

    let data = 0 .. 65536;
    let data_buffer =
        CpuAccessibleBuffer::from_iter(
            device.clone(), BufferUsage::all(), data)?;

    mod cs {
        vulkano_shaders::shader!{
            ty: "compute",
            src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
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

    let set =
        Arc::new(
            PersistentDescriptorSet::start(
                compute_pipeline.clone(), 0
            ).add_buffer(data_buffer.clone())?.build()?
        );

    let command_buffer =
        AutoCommandBufferBuilder::new(device.clone(), queue.family())?
            .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ())?
            .build()?;

    let finished = command_buffer.execute(queue.clone())?;
    finished.then_signal_fence_and_flush()?.wait(None)?;

    let content = data_buffer.read()?;
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    println!("Everything succeeded!");

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
