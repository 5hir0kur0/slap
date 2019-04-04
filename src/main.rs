extern crate portaudio;
extern crate rustfft;
use portaudio as pa;
use std::io;
use std::io::Write;
use std::str::FromStr;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true;       // Shouldn't make a difference; we only have 1 channel
const DEFAULT_SAMPLE_RATE: f64 = 44100.0;
const FRAME_COUNT: u32 = 1024;
const THRESHOLD: f32 = 42.0;
type FORMAT = f32;

/// Print a prompt and read a value from stdin.

fn read_from_stdin<T: FromStr>(p: &str) -> Result<T, T::Err> {
    prompt(p);
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
    let mut default_index = -1_i32;
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
                default_index = i as i32;
                println!("* ({}) {} [{}]", i, info.name, host.name);
            } else {
                println!("  ({}) {} [{}]", i, info.name, host.name);
            }
            inputs.push(idx);
            i += 1;
        }
    }
    let choice: Result<u32,_> = read_from_stdin("Choose device (leave blank to use the one marked with '*'): ");
    // let choice = choice.expect("Failed to parse user input");
    Ok(inputs[choice.unwrap_or(default_index as u32) as usize])
}

fn choose_sample_rate(pa: &pa::PortAudio, input_params: pa::StreamParameters<FORMAT>) -> f64 {
    let choice: Result<f64,_> = read_from_stdin(&format!("Choose sample rate (default: {} Hz): ",
                                      DEFAULT_SAMPLE_RATE));
    let sample_rate = choice.unwrap_or(DEFAULT_SAMPLE_RATE);
    if pa.is_input_format_supported(input_params, sample_rate).is_err() {
        panic!("PortAudio says this sample rate won't work");
    }
    sample_rate
}

fn prompt(p: &str) {
    print!("{}", p);
    io::stdout().flush();
}

fn main() {
    // Initialize PortAudio
    let pa = pa::PortAudio::new().expect("Failed to create PortAudio object");
    let pa_version = pa::version_text().expect("Failed to get PortAudio version");
    println!();
    println!("Using PortAudio version '{}'.", pa_version);
    let input_idx = choose_device(&pa).unwrap();
    let input_info = pa.device_info(input_idx).unwrap();
    println!("Using input device: {}", input_info.name);
    let latency = input_info.default_low_input_latency;
    println!("Using latency: {}", latency);
    let input_params = pa::StreamParameters::<FORMAT>::new(input_idx, CHANNELS,
                                                        INTERLEAVED, latency);
    let sample_rate = choose_sample_rate(&pa, input_params);
    println!("Using sample rate: {}", sample_rate);
    let settings = pa::stream::InputSettings::new(input_params, sample_rate, FRAME_COUNT);
    let mut stream = pa.open_blocking_stream(settings).expect("Failed to open stream");
    stream.start().expect("Failed to start stream");

    println!();
    let mut fft_input = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let mut fft_output = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let fft = FFTplanner::new(false).plan_fft(FRAME_COUNT as usize);
    loop {
        let input_samples = stream.read(FRAME_COUNT);
        fft_input.clear();
        let input_samples = match input_samples {
            Ok(samples) => samples,
            Err(err) => {
                eprintln!("ERROR: {}", err);
                continue
            }
        };
        fft_input.extend(input_samples.iter().map(|x| Complex::new(*x, 0.0)));
        fft.process(&mut fft_input, &mut fft_output);
        let mut max = 0;
        for i in 1..fft_output.len() {
            if fft_output[i].norm() > fft_output[max].norm() {
                max = i;
            }
        }
        if fft_output[max].norm() > THRESHOLD {
            print!("\r{}", max as f64 * sample_rate / FRAME_COUNT as f64);
            io::stdout().flush().unwrap();
        }
    }
}
