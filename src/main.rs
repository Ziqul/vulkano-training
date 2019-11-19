use std::error::Error;
use std::sync::Arc;

use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;

fn main() -> Result<(), Box<Error>> {

    let (instance, device, queue) = init()?;

    Ok(())
}

fn init() -> Result<(Arc<Instance>, Arc<Device>, Arc<Queue>), Box<Error>> {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)?;

    #[cfg(debug_assertions)]
    {
        println!("Listing available devices supporting Vulkan API:");
        for device in PhysicalDevice::enumerate(&instance) {
            println!("{:?}", device);

            println!("Found a queue family with this queue(s) amount:");
            for family in device.queue_families() {
                println!("{:?}", family.queues_count());
            }

        }

        println!("");
    }

    let chosen_physical_device =
        PhysicalDevice::enumerate(&instance).nth(0)
            .expect("Error: NoneError: No physical devices supporting Vulkan API found");

    #[cfg(debug_assertions)]
    {
        println!("Chosen device: {:?}", chosen_physical_device);

        println!("");
    }

    let chosen_family = chosen_physical_device.queue_families()
        .find(|&q| q.supports_graphics())
        .expect("Error: NoneError: No family supporting GRAPHICS_BIT found in chosen device");

    let (chosen_logical_device, mut queues) = {
        Device::new(
            chosen_physical_device, &Features::none(),
            &DeviceExtensions::none(),
            [(chosen_family, 0.5)].iter().cloned()
        )?
    };

    let chosen_queue = queues.next()
        .expect("Error: NoneError: No queue found in chosen family");

    Ok((instance, chosen_logical_device, chosen_queue))
}
