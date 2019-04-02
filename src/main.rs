extern crate portaudio;
use portaudio as pa;
use std::io;
use std::str::FromStr;

/// Print a prompt and read a value from stdin.

fn read_from_stdin<T: FromStr>(prompt: &str) -> Result<T, T::Err> {
    print!("{}", prompt);
    let mut s = String::new();
    io::stdin().read_line(&mut s)
        .expect("Failed to read user input");
    s.trim().parse::<T>()
}

/// Let the user choose an input device using stdin.

fn choose_device(pa: &pa::PortAudio) -> Result<pa::DeviceIndex, pa::Error> {
    let devices = pa.devices()?;
    let mut inputs = Vec::new();
    let mut i = 0;
    for d in devices {
        let (idx, info) = d?;
        let host = match pa.host_api_info(info.host_api) {
            Some(h) => h,
            None => {
                println!("Couldn't get host for '{}'. Ignoring.", info.name);
                continue
            }
        };
        let is_default = if let Some(default) = host.default_input_device {
            info.host_api == pa.default_host_api()? && default == idx
        } else { false };
        if info.max_input_channels >= 1 {
            if is_default {
                println!("* ({}) {} [{}]", i, info.name, host.name);
            } else {
                println!("  ({}) {} [{}]", i, info.name, host.name);
            }
            inputs.push(idx);
            i += 1;
        }
    }
    let choice: Result<u32,_> = read_from_stdin("Choose device (leave blank to use the one marked with '*'): ");
    let choice = choice.expect("Failed to parse user input");
    Ok(pa::DeviceIndex(choice))
}

fn main() {
    // Initialize PortAudio
    let pa = pa::PortAudio::new().expect("Failed to create PortAudio object");
    let pa_version = pa::version_text().expect("Failed to get PortAudio version");
    println!("Using PortAudio version '{}'.", pa_version);
    let input_idx = choose_device(&pa);

}
