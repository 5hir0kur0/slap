extern crate portaudio;
extern crate rustfft;
extern crate sdl2;
use portaudio as pa;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FftPlanner;
use sdl2::event::Event;
use sdl2::gfx::primitives::DrawRenderer;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::render::Canvas;
use std::collections::VecDeque;
use std::io;
use std::io::Write;
use std::str::FromStr;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::Duration;

const CHANNELS: i32 = 1;
const INTERLEAVED: bool = true; // Shouldn't make a difference; we only have 1 channel
const DEFAULT_SAMPLE_RATE: f64 = 11_025.0;
const FRAME_COUNT: u32 = 1024;
const THRESHOLD: f32 = 3.0;
const LOWEST_FREQUENCY: f32 = 50.0;
const HIGHEST_FREQUENCY: f32 = 550.0;
const LONG_SILENCE_DURATION: i32 = 15; // Number of samples below threshold that constitute "long silence"
const NUMBER_OF_VALUES: usize = 84;
const N: usize = 4; // The number of frequencies to consider for the graph (i.e. use the N loudest frequencies for the graph)
type FORMAT = f32;

/// Data structure used for passing values from the main thread to the gui thread
enum DataPoint {
    Frequency {
        freq: FORMAT,
        norm: FORMAT,
        values: Vec<(FORMAT, FORMAT)>,
    },
    Nothing {
        values: Vec<(FORMAT, FORMAT)>,
    },
    LongSilence,
}

/// Print a prompt and read a value from stdin.
fn read_from_stdin<T: FromStr>(p: &str) -> Result<T, T::Err> {
    prompt(p);
    let mut s = String::new();
    io::stdin()
        .read_line(&mut s)
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
                eprintln!("Couldn't get host for '{}'. Ignoring.", info.name);
                continue;
            }
        };
        let is_default = if let Some(default) = host.default_input_device {
            info.host_api == pa.default_host_api()? && default == idx
        } else {
            false
        };
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
    let choice: Result<u32, _> =
        read_from_stdin("Choose device (leave blank to use the one marked with '*'): ");
    // let choice = choice.expect("Failed to parse user input");
    Ok(inputs[choice.unwrap_or(default_index as u32) as usize])
}

/// Let the user choose a sample rate
fn choose_sample_rate(pa: &pa::PortAudio, input_params: pa::StreamParameters<FORMAT>) -> f64 {
    let choice: Result<f64, _> = read_from_stdin(&format!(
        "Choose sample rate (default: {} Hz): ",
        DEFAULT_SAMPLE_RATE
    ));
    let sample_rate = choice.unwrap_or(DEFAULT_SAMPLE_RATE);
    if pa
        .is_input_format_supported(input_params, sample_rate)
        .is_err()
    {
        panic!("PortAudio says this sample rate won't work");
    }
    sample_rate
}

/// Print a prompt and flush stdout
fn prompt(p: &str) {
    print!("{}", p);
    io::stdout().flush().unwrap();
}

/// Create the stream settings for the input stream
/// Return the sample rate and the settings
fn init_stream_settings(
    pa: &pa::PortAudio,
) -> Result<(f64, pa::stream::InputSettings<FORMAT>), pa::Error> {
    let input_idx = choose_device(pa)?;
    let input_info = pa.device_info(input_idx)?;
    println!("Using input device: {}", input_info.name);
    let latency = input_info.default_low_input_latency;
    println!("Using latency: {}", latency);
    let input_params =
        pa::StreamParameters::<FORMAT>::new(input_idx, CHANNELS, INTERLEAVED, latency);
    let sample_rate = choose_sample_rate(pa, input_params);
    println!("Using sample rate: {}", sample_rate);
    let settings = pa::stream::InputSettings::new(input_params, sample_rate, FRAME_COUNT);
    Ok((sample_rate, settings))
}

/// Compute which indices in the output array should even be considered
/// Returns the lower bound and the upper bound
fn compute_significant_indices(sample_rate: f64) -> (usize, usize) {
    let lower = (LOWEST_FREQUENCY as f64 * FRAME_COUNT as f64 / sample_rate).ceil() as usize;
    let upper = (HIGHEST_FREQUENCY as f64 * FRAME_COUNT as f64 / sample_rate).floor() as usize;
    println!(
        "Considering indices from {} ({:.0}Hz) to {} ({:.0}Hz)",
        lower,
        index_to_freq(lower, sample_rate),
        upper,
        index_to_freq(upper, sample_rate)
    );

    (lower, upper)
}

/// Process a single fft output. Only indices i where lower <= i <= upper are considered.
/// Returns the weighted average over the top n frequencies, the average of their norms
/// and a vector containing the frequencies and norms of all indices i
/// sorted in descending order by the norms
fn process_fft_results(
    n: usize,
    fft_output: &Vec<Complex<FORMAT>>,
    lower: usize,
    upper: usize,
    sample_rate: f64,
) -> (f32, f32, Vec<(f32, f32)>) {
    // Contains only the values in the range that is being analyzed
    let mut fft_freq_norm = Vec::with_capacity(upper - lower + 1);

    for (i, z) in fft_output[lower..=upper].iter().enumerate() {
        fft_freq_norm.push((index_to_freq(i + lower, sample_rate), z.norm()));
    }

    // sort by norms in reverse order (loudest first)
    fft_freq_norm.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let mut sum_norm: f32 = 0.0;
    let mut avg_freq: f32 = 0.0;

    for (_, n) in &fft_freq_norm[0..n] {
        sum_norm += n;
    }

    for (f, n) in &fft_freq_norm[0..n] {
        avg_freq += f * n / sum_norm;
    }

    let avg_norm = sum_norm / n as f32;

    (avg_freq, avg_norm, fft_freq_norm)
}

/// Initialize the gui and redraw in an infinite loop
fn gui_thread(rx: Receiver<DataPoint>) -> Result<(), String> {
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    // anti-aliasing
    let gl_attr = video_subsystem.gl_attr();
    gl_attr.set_multisample_buffers(1);
    gl_attr.set_multisample_samples(16);

    let window = video_subsystem
        .window("SLAP", 800, 600)
        .position_centered()
        .resizable()
        .opengl()
        .build()
        .map_err(|e| e.to_string())?;

    let mut canvas = window
        .into_canvas()
        .present_vsync()
        .build()
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump()?;
    let mut nothing_count = 0;
    let mut values = VecDeque::<DataPoint>::new();
    let mut last_nothing = None;
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                }
                | Event::KeyDown {
                    keycode: Some(Keycode::Q),
                    ..
                } => break 'running,
                Event::KeyDown {
                    keycode: Some(Keycode::R),
                    ..
                } => {
                    values.clear();
                }
                _ => {}
            }
        }
        while let Ok(data) = rx.try_recv() {
            while values.len() >= NUMBER_OF_VALUES {
                let _ = values.pop_front();
            }
            match data {
                v @ DataPoint::Frequency { .. } => {
                    // println!("Got {} @ {}", freq, norm);
                    nothing_count = 0;
                    if let Some(nothing) = last_nothing {
                        if values.len() >= NUMBER_OF_VALUES {
                            let _ = values.pop_front();
                        }
                        values.push_back(nothing);
                        last_nothing = None;
                    }
                    values.push_back(v);
                }
                v @ DataPoint::Nothing { .. } => {
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
                DataPoint::LongSilence => {
                    eprintln!("Got LongSilence (This can't happen in the current implementation)")
                }
            }
        }
        if let Err(mpsc::TryRecvError::Disconnected) = rx.try_recv() {
            break 'running;
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

/// Plot the values in the analyzed range as dots with varying alpha
/// channel according to their amplitude.
/// Also draw a line through the dominant frequencies and a smoothed
/// version of the line (average of the point and the prev/next point,
/// doesn't stop at a "nothing" value)
fn draw_data_points(
    canvas: &mut Canvas<sdl2::video::Window>,
    points: &VecDeque<DataPoint>,
) -> Result<(), String> {
    let (width, height) = canvas.output_size()?;
    let mut prev = None;
    let mut prev_smooth = None;
    let line_color = Color::RGBA(0x07, 0x66, 0x78, 0x80);
    let point_color = Color::RGB(0xfa, 0xbd, 0x2f);
    let silence_color = Color::RGB(0x8f, 0x3f, 0x71);
    let smooth_color = Color::RGB(0xeb, 0xdb, 0xb2);
    let line_thickness = 4;
    let radius = 2;
    let mut nothing_or_freq_values;
    for i in 0..points.len() {
        let x = i * width as usize / points.len();
        let x = x as i16;
        match &points[i] {
            DataPoint::Frequency { freq, values, .. } => {
                let y = height as f32
                    * (1.0 - (freq - LOWEST_FREQUENCY) / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let y = y as i16;
                let smooth_freq = smooth_value(i, *freq, &points);
                let smooth_y = height as f32
                    * (1.0
                        - (smooth_freq - LOWEST_FREQUENCY)
                            / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let smooth_y = smooth_y as i16;
                if let Some(p) = prev {
                    let (x_old, y_old) = p;
                    canvas.thick_line(x_old, y_old, x, y, line_thickness, line_color)?;
                } else {
                    canvas.filled_circle(x, y, line_thickness as i16, line_color)?;
                }
                if let Some(p) = prev_smooth {
                    let (x_old, y_old) = p;
                    canvas.thick_line(x_old, y_old, x, smooth_y, line_thickness, smooth_color)?;
                } else {
                    canvas.filled_circle(x, smooth_y, line_thickness as i16, smooth_color)?;
                }
                prev = Some((x, y));
                prev_smooth = Some((x, smooth_y));
                nothing_or_freq_values = Some(values);
            }
            DataPoint::Nothing { values } => {
                prev = None;
                // prev_smooth = None;
                nothing_or_freq_values = Some(values);
            }
            DataPoint::LongSilence => {
                prev = None;
                prev_smooth = None;
                nothing_or_freq_values = None;
                canvas.thick_line(
                    x,
                    0,
                    x,
                    height as i16,
                    2 * line_thickness / 3,
                    silence_color,
                )?;
            }
        }
        if let Some(values) = nothing_or_freq_values {
            let max_norm: FORMAT = values[0].1;
            for v in values {
                let (f, n) = v;
                let col = Color::RGBA(
                    point_color.r,
                    point_color.g,
                    point_color.b,
                    (0xff as f32 * n / max_norm) as u8,
                );
                let y2 = height as f32
                    * (1.0 - (f - LOWEST_FREQUENCY) / (HIGHEST_FREQUENCY - LOWEST_FREQUENCY));
                let y2 = y2 as i16;
                canvas.filled_circle(x, y2, radius, col)?;
            }
        }
    }
    Ok(())
}

/// Smooth the value by averaging it with the prev/next value.
fn smooth_value(i: usize, f: f32, points: &VecDeque<DataPoint>) -> f32 {
    let mut numerator = f;
    let mut denominator = 1.0;
    if i > 0 {
        if let DataPoint::Frequency {
            freq: freq_left, ..
        } = points[i - 1]
        {
            numerator += 2.0 * freq_left;
            denominator += 2.0;
        }
    }
    if i < points.len() - 1 {
        if let DataPoint::Frequency {
            freq: freq_right, ..
        } = points[i + 1]
        {
            numerator += 2.0 * freq_right;
            denominator += 2.0;
        }
    }
    numerator / denominator
}

/// Convert an index in the fft output to the corresponding frequency
fn index_to_freq(i: usize, sample_rate: f64) -> f32 {
    i as f32 * sample_rate as f32 / FRAME_COUNT as f32
}

/// Open a stream and return the sample rate and the stream
type Stream = pa::Stream<pa::Blocking<pa::stream::Buffer>, pa::Input<FORMAT>>;
fn initialize_stream(pa: &pa::PortAudio) -> Result<(f64, Stream), pa::Error> {
    let (sample_rate, settings) = init_stream_settings(&pa)?;

    Ok((sample_rate, pa.open_blocking_stream(settings)?))
}

fn main() -> Result<(), pa::Error> {
    // Initialize PortAudio
    let pa = pa::PortAudio::new()?;
    let pa_version = pa::version_text().unwrap();
    println!();
    println!("Using PortAudio version '{}'.", pa_version);

    let (sample_rate, mut stream) =
        initialize_stream(&pa).expect("Failed to open PortAudio stream");
    let (first_significant_index, last_significant_index) =
        compute_significant_indices(sample_rate as f64);

    stream.start().expect("Failed to start stream");

    let (tx, rx): (Sender<DataPoint>, Receiver<DataPoint>) = mpsc::channel();
    thread::spawn(move || gui_thread(rx));

    let mut fft_input_old = vec![Complex::<FORMAT>::zero(); FRAME_COUNT as usize];
    let mut fft_input_new = fft_input_old.clone();
    let mut fft_output = fft_input_old.clone();
    let mut fft_scratch = fft_input_old.clone();
    let fft = FftPlanner::new().plan_fft_forward(FRAME_COUNT as usize);
    let mut first_run = true;
    loop {
        let input_samples = stream.read(if first_run {
            FRAME_COUNT
        } else {
            FRAME_COUNT / 2
        });
        let input_samples = match input_samples {
            Ok(samples) => samples,
            Err(err) => {
                eprintln!("ERROR: {}", err);
                continue;
            }
        };
        if !first_run {
            std::mem::swap(&mut fft_input_old, &mut fft_input_new);
            fft_input_new.clear();
            fft_input_new.extend(&fft_input_old[FRAME_COUNT as usize / 2..]);
        } else {
            first_run = false;
            fft_input_new.clear();
        }
        fft_input_new.extend(input_samples.iter().map(|x| Complex::new(*x, 0.0)));
        fft_output.clear();
        fft_output.extend(fft_input_new.iter());
        fft_scratch.resize(fft.get_inplace_scratch_len(), Complex::<FORMAT>::zero());
        // This writes the output into the first argument.
        fft.process_with_scratch(&mut fft_output, &mut fft_scratch);
        let (freq, norm, freqs_norms) = process_fft_results(
            N,
            &fft_output,
            first_significant_index,
            last_significant_index,
            sample_rate,
        );

        let res;
        if norm > THRESHOLD {
            // println!("\r{} @ {}", freq, norm);
            res = tx.send(DataPoint::Frequency {
                freq,
                norm,
                values: freqs_norms,
            });
        } else {
            res = tx.send(DataPoint::Nothing {
                values: freqs_norms,
            });
        }
        if res.is_err() {
            println!("The GUI thread disconnected. Exiting.");
            break;
        }
    }
    Ok(())
}
