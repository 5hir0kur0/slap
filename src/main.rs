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
use sdl2::event::WindowEvent::Resized;
use sdl2::keyboard::Keycode;
use std::time::Duration;
use sdl2::render::Canvas;
use sdl2::gfx::primitives::DrawRenderer;


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
    /// values: (freq,norm) of all frequencies in the range being analyzed
    Frequency{freq: FORMAT, norm: FORMAT, values: Vec<(FORMAT, FORMAT)>},
    Nothing{values: Vec<(FORMAT, FORMAT)>},
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
    io::stdout().flush().unwrap();
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

fn process_fft_results(n: usize, sorted_freq_norm: &Vec<(f32, f32)>) -> (f32,f32) {
    let mut sum_norm: f32 = 0.0;
    let mut avg_freq: f32 = 0.0;

    for (_,n) in &sorted_freq_norm[0..n] {
        sum_norm += n;
    }

    for (f,n) in &sorted_freq_norm[0..n] {
        avg_freq += f * n / sum_norm;
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
    gl_attr.set_multisample_samples(16);

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
    let mut last_nothing = None;
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
            while values.len() >= NUMBER_OF_VALUES {
                let _ = values.pop_front();
            }
            match data {
                v@DataPoint::Frequency{..} => {
                    // println!("Got {} @ {}", freq, norm);
                    nothing_count = 0;
                    if let Some(nothing) = last_nothing {
                        if values.len() >= NUMBER_OF_VALUES { let _ = values.pop_front(); }
                        values.push_back(nothing);
                        last_nothing = None;
                        println!("pushing nothing");
                    }
                    values.push_back(v);
                },
                v@DataPoint::Nothing{..} => {
                    // println!("Got Nothing");
                    nothing_count += 1;
                    if nothing_count == 1 {
                        values.push_back(v);
                    } else if nothing_count >= LONG_SILENCE_DURATION {
                        if nothing_count == LONG_SILENCE_DURATION {
                            values.push_back(DataPoint::LongSilence);
                        }
                        last_nothing = Some(v);
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

fn draw_data_points(canvas: &mut Canvas<sdl2::video::Window>, points: &VecDeque<DataPoint>) -> Result<(), String> {
    let (width,height) = canvas.output_size()?;
    let mut prev = None;
    let line_color = Color::RGB(0xeb, 0xdb, 0xb2);
    let point_color = Color::RGB(0xfa, 0xbd, 0x2f);
    let silence_color = Color::RGB(0x8f, 0x3f, 0x71);
    let line_thickness = 4;
    let radius = 1;
    let mut nothing_or_freq_values;
    for i in 0..points.len() {
        let x = i * width as usize / points.len();
        let x = x as i16;
        match &points[i] {
            DataPoint::Frequency{freq, values, ..} => {
                let y = height as f32 * (1.0 - (freq - LOWEST_FREQUENCY) / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let y = y as i16;
                if let Some(p) = prev {
                    let (x_old, y_old) = p;
                    canvas.thick_line(x_old, y_old, x, y, line_thickness, line_color)?;
                } else {
                    canvas.circle(x, y, line_thickness as i16/2, line_color)?;
                }
                prev = Some((x,y));
                nothing_or_freq_values = Some(values);
            },
            DataPoint::Nothing{values} => {
                prev = None;
                nothing_or_freq_values = Some(values);
            },
            DataPoint::LongSilence => {
                nothing_or_freq_values = None;
                canvas.thick_line(x, 0, x, height as i16, 2*line_thickness/3, silence_color)?;
            }
        }
        if let Some(values) = nothing_or_freq_values {
            let max_norm: FORMAT = values[0].1;
            for v in values {
                let (f,n) = v;
                let col = Color::RGBA(point_color.r, point_color.g, point_color.b, (0xff as f32 * n/max_norm) as u8);
                let y2 = height as f32 * (1.0 - (f - LOWEST_FREQUENCY) / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let y2 = y2 as i16;
                canvas.circle(x, y2, radius, col)?;
            }
        }
    }
    Ok(())
}

fn index_to_freq(i: usize, sample_rate: f64) -> f32 {
    i as f32 * sample_rate as f32 / FRAME_COUNT as f32
}

fn main() {
    // Initialize PortAudio
    let pa = pa::PortAudio::new().expect("Failed to create PortAudio object");
    let pa_version = pa::version_text().expect("Failed to get PortAudio version");
    println!();
    println!("Using PortAudio version '{}'.", pa_version);

    let (sample_rate, settings) = init_stream_settings(&pa).unwrap();
    let (first_significant_index, last_significant_index) = compute_significant_indices(sample_rate as f64);

    println!("Considering indices from {} ({:.0}Hz) to {} ({:.0}Hz)",
             first_significant_index,
             index_to_freq(first_significant_index, sample_rate),
             last_significant_index,
             index_to_freq(last_significant_index, sample_rate));
    let mut stream = pa.open_blocking_stream(settings).expect("Failed to open stream");
    stream.start().expect("Failed to start stream");

    let (tx, rx): (Sender<DataPoint>, Receiver<DataPoint>) = mpsc::channel();
    thread::spawn(move || gui_thread(rx));

    println!();
    let mut fft_input = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let mut fft_output = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let fft = FFTplanner::new(false).plan_fft(FRAME_COUNT as usize);
    let n = 3;
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

        // Contains only the values in the range that is being analyzed
        let mut fft_freq_norm = Vec::with_capacity(last_significant_index - first_significant_index + 1);

        for (i,z) in fft_output[first_significant_index..=last_significant_index].iter().enumerate() {
            fft_freq_norm.push((index_to_freq(i + first_significant_index, sample_rate), z.norm()));
        }

        fft_freq_norm.sort_by(|(_,a), (_,b)| b.partial_cmp(a).unwrap());

        let (freq,norm) = process_fft_results(n, &fft_freq_norm);

        let res;
        if norm > THRESHOLD {
            // println!("\r{} @ {}", freq, norm);
            res = tx.send(DataPoint::Frequency{freq,norm, values: fft_freq_norm});
        } else {
            res = tx.send(DataPoint::Nothing{values: fft_freq_norm});
        }
        if res.is_err() {
            println!("The GUI thread disconnected. Exiting.");
            break
        }

    }
}
