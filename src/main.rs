extern crate sdl2;
extern crate portaudio;
extern crate rustfft;
use portaudio as pa;
use std::io;
use std::thread;
use std::io::Write;
use std::str::FromStr;
use std::sync::mpsc::{Sender,Receiver};
use std::sync::mpsc;
use std::collections::VecDeque;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use sdl2::pixels::Color;
use sdl2::event::Event;
use sdl2::event::WindowEvent;
use sdl2::event::WindowEvent::Resized;
use sdl2::keyboard::Keycode;
use std::time::Duration;
use sdl2::rect::Point;
use sdl2::render::Canvas;


const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true;       // Shouldn't make a difference; we only have 1 channel
const DEFAULT_SAMPLE_RATE: f64 = 10_000.0;
const FRAME_COUNT: u32 = 1024;
const THRESHOLD: f32 = 5.0;
const LOWEST_FREQUENCY: f32 = 70.0;
const HIGHEST_FREQUENCY: f32 = 1000.0;
const LONG_SILENCE_DURATION: i32 = 10; // Number of samples below threshold that consitute "long silence"
const NUMBER_OF_VALUES: usize = 100;
type FORMAT = f32;

enum DataPoint {
    Frequency{freq: FORMAT, norm: FORMAT},
    Nothing,
    LongSilence
}

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

/// Create the stream settings for the input stream
/// Return the sample rate and the settings
fn init_stream_settings(pa: &pa::PortAudio) -> Result<(f64, pa::stream::InputSettings<FORMAT>), pa::Error> {
    let input_idx = choose_device(pa)?;
    let input_info = pa.device_info(input_idx)?;
    println!("Using input device: {}", input_info.name);
    let latency = input_info.default_low_input_latency;
    println!("Using latency: {}", latency);
    let input_params = pa::StreamParameters::<FORMAT>::new(input_idx, CHANNELS,
                                                        INTERLEAVED, latency);
    let sample_rate = choose_sample_rate(pa, input_params);
    println!("Using sample rate: {}", sample_rate);
    let settings = pa::stream::InputSettings::new(input_params, sample_rate, FRAME_COUNT);
    Ok((sample_rate, settings))
}

/// Compute which indices in the output array should even be considered
/// Returns the lower bound and the upper bound
fn compute_significant_indices(sample_rate: f64) -> (usize,usize) {
    let lower = (LOWEST_FREQUENCY as f64 * FRAME_COUNT as f64 / sample_rate).ceil() as usize;
    let upper = (HIGHEST_FREQUENCY as f64 * FRAME_COUNT as f64 / sample_rate).floor() as usize;
    (lower, upper)
}

/// Process the results of the FFT. Consider only indices i with lower <= i <= upper.
/// Uses index_norm as a scratch space to store the mapping between indices and norms
/// of the output vectors.
/// Computes the average of the loudest n frequencies.
/// Returns this average and the average norm (considering the loudest n frequencies).
fn process_fft_results(lower: usize, upper: usize, n: usize, sample_rate: f64,
                       fft_output: &mut Vec<Complex<FORMAT>>,
                       index_norm: &mut Vec<(usize, f32)>) -> (f32, f32) {
    index_norm.clear();
    // let mut max = first_significant_index;
    // for i in first_significant_index..=last_significant_index {
    //     if fft_output[i].norm() > fft_output[max].norm() {
    //         max = i;
    //     }
    // }
    // let norm = fft_output[max].norm();
    // if norm > THRESHOLD {
    //     println!("\r{} @ {}", max as f64 * sample_rate / FRAME_COUNT as f64, norm);
    //     io::stdout().flush().unwrap();
    // }

    for i in 0..n {
        let mut max = lower;
        for j in lower + i..=upper {
            if fft_output[j].norm() > fft_output[max].norm() {
                max = j;
            }
        }
        index_norm.push((max, fft_output[max].norm()));
        fft_output[max] = Complex::zero();
    }

    let mut sum_norm: f32 = 0.0;
    let mut avg_freq: f32 = 0.0;
    let num = index_norm.len();
    for i in 0..num {
        sum_norm += index_norm[i].1;
    }
    for i in 0..num {
        let (idx, norm) = index_norm[i];
        avg_freq += idx as f32 * sample_rate as f32 / FRAME_COUNT as f32 * norm / sum_norm;
    }
    let avg_norm = sum_norm / n as f32;
    (avg_freq, avg_norm)
}

fn gui_thread(rx: Receiver<DataPoint>) -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    // anti-aliasing
    let gl_attr = video_subsystem.gl_attr();
    gl_attr.set_multisample_buffers(1);
    gl_attr.set_multisample_samples(4);

    let window = video_subsystem.window("SLAP", 800, 600)
        .position_centered()
        .resizable()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window.into_canvas().present_vsync().build().map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;
    let mut nothing_count = 0;
    let mut values =  VecDeque::<DataPoint>::new();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'running
                },
                Event::Window{win_event: Resized(width,height), ..} => {
                    println!("Resize to {}x{}", width, height);
                }
                _ => {}
            }
        }
        while let Ok(data) = rx.try_recv() {
            while values.len() > NUMBER_OF_VALUES {
                let _ = values.pop_front();
            }
            match data {
                DataPoint::Frequency{freq, norm} => {
                    println!("Got {} @ {}", freq, norm);
                    nothing_count = 0;
                    values.push_back(DataPoint::Frequency{freq,norm});
                },
                DataPoint::Nothing => {
                    println!("Got Nothing");
                    nothing_count += 1;
                    if nothing_count == 1 {
                        values.push_back(DataPoint::Nothing);
                    } else if nothing_count == LONG_SILENCE_DURATION {
                        values.push_back(DataPoint::LongSilence);
                    }
                }
                DataPoint::LongSilence => eprintln!("Got LongSilence (This can't happen in the current implementation)"),
            }
        }
        if let Err(mpsc::TryRecvError::Disconnected) = rx.try_recv() {
            break 'running
        }
        thread::sleep(Duration::from_millis(5));
        // The rest of the game loop goes here...
        // println!("{:?}", canvas.output_size())
        canvas.set_draw_color(Color::RGB(0x28, 0x28, 0x28));
        canvas.clear();
        draw_data_points(&mut canvas, &values)?;
        canvas.present();
    }

    Ok(())
}

fn draw_data_points(canvas: &mut Canvas<sdl2::video::Window>, values: &VecDeque<DataPoint>) -> Result<(), String> {
    let (width,height) = canvas.output_size()?;
    canvas.set_draw_color(Color::RGB(0xeb, 0xdb, 0xb2));
    let mut prev = None;
    for i in 0..values.len() {
        match &values[i] {
            DataPoint::Frequency{freq, ..} => {
                let x = i * width as usize / values.len();
                let y = height as f32 * (1.0 - (freq - LOWEST_FREQUENCY) / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let curr = Point::new(x as i32, y as i32);
                println!("freq: {}, x: {}, y: {} (h: {})", freq, x, y as i32, height);
                if let Some(p) = prev {
                    canvas.draw_line(p, curr)?;
                } else {
                    canvas.draw_point(curr)?;
                }
                prev = Some(curr);
            },
            DataPoint::Nothing => {
                prev = None;
            },
            _ => {}
        }
    }
    // canvas.draw_line(Point::new(0,0), Point::new(width as i32, height as i32));
    canvas.set_draw_color(Color::RGB(0xff, 0, 0));
    canvas.draw_point(Point::new(width as i32 - 1, height as i32 - 1));
    Ok(())
}

fn main() {
    // Initialize PortAudio
    let pa = pa::PortAudio::new().expect("Failed to create PortAudio object");
    let pa_version = pa::version_text().expect("Failed to get PortAudio version");
    println!();
    println!("Using PortAudio version '{}'.", pa_version);

    let (sample_rate, settings) = init_stream_settings(&pa).unwrap();
    let (first_significant_index, last_significant_index) = compute_significant_indices(sample_rate as f64);

    println!("Considering indices from {} to {}", first_significant_index, last_significant_index);
    let mut stream = pa.open_blocking_stream(settings).expect("Failed to open stream");
    stream.start().expect("Failed to start stream");

    let (tx, rx): (Sender<DataPoint>, Receiver<DataPoint>) = mpsc::channel();
    thread::spawn(move || gui_thread(rx));

    println!();
    let mut fft_input = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let mut fft_output = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let fft = FFTplanner::new(false).plan_fft(FRAME_COUNT as usize);
    let n = 3;
    let mut index_norm = Vec::<(usize,f32)>::new();
    let mut nothing_count = 0;
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

        let (freq,norm) = process_fft_results(first_significant_index, last_significant_index,
                                              n, sample_rate, &mut fft_output, &mut index_norm);
        let res;
        if norm > THRESHOLD {
            // println!("\r{} @ {}", freq, norm);
            res = tx.send(DataPoint::Frequency{freq,norm});
        } else {
            res = tx.send(DataPoint::Nothing);
        }
        if res.is_err() {
            println!("The GUI thread disconnected. Exiting.");
            break
        }

    }
}
