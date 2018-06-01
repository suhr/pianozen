#![allow(dead_code)]

extern crate jack;
extern crate miosc;
extern crate rosc;

use std::collections::VecDeque;

struct Noise {
    rand: u32,
}

impl Noise {
    fn new() -> Self {
        // XKCD random number
        let rand = 4;

        Noise {
            rand,
        }
    }
    fn generate(&mut self) -> f32 {
        // BSD variant of the linear congruential method
        self.rand = self.rand.wrapping_mul(1_103_515_245).wrapping_add(12345);
        self.rand %= 1 << 31;

        self.rand as f32 / 2_147_483_647.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum AdsrState {
    Idle,
    Press(u32),
    Sustain,
    Release(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Adsr {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    state: AdsrState,
}

impl Adsr {
    fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Adsr {
            attack, decay, sustain, release,
            state: AdsrState::Idle,
        }
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let ms = 1e-3 * parameters.sample_rate;

        let attack = parameters.attack * ms;
        let decay = parameters.decay * ms;
        let release = parameters.release * ms;

        let sustain = parameters.sustain;

        Adsr {
            attack, decay, sustain, release,
            state: AdsrState::Idle,
        }
    }

    fn note_on(&mut self) {
        self.state = AdsrState::Press(0)
    }

    fn note_off(&mut self) {
        self.state = AdsrState::Release(0)
    }

    fn is_active(&self) -> bool {
        self.state != AdsrState::Idle
    }

    fn increment_state(&mut self) {
        match self.state {
            AdsrState::Press(ref mut t) =>
                *t += 1,
            AdsrState::Release(ref mut t) =>
                *t += 1,
            _ => ()
        }
    }

    fn generate(&mut self) -> f32 {
        let adsr =
            match self.state {
                AdsrState::Idle =>
                    0.0,
                AdsrState::Press(mut t) if (t as f32) < self.attack => {
                    t as f32 / self.attack
                },
                AdsrState::Press(t) => {
                    let t = t as f32 - self.attack;

                    if t < self.decay {
                        1.0 - t * (1.0 - self.sustain) / self.decay
                    } else {
                        self.state = AdsrState::Sustain;

                        self.sustain
                    }
                },
                AdsrState::Sustain =>
                    self.sustain,
                AdsrState::Release(t) =>
                    if (t as f32) < self.release {
                        self.sustain * (1.0 - t as f32 / self.release)
                    } else {
                        self.state = AdsrState::Idle;

                        0.0
                    }
            };

        self.increment_state();

        adsr
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Filter {
    z1: f32,
}

impl Filter {
    fn new() -> Self {
        Filter {
            z1: 0.0,
        }
    }

    fn signal(&mut self, input: f32) -> f32 {
        let output = 0.5 * (input + self.z1);
        self.z1 = input;

        output
    }
}

struct Delay {
    queue: VecDeque<f32>,
}

impl Delay {
    fn new(len: usize) -> Self {
        Delay {
            queue: vec![0.0; len].into()
        }
    }
    fn signal(&mut self, input: f32) -> f32 {
        let z_out = self.queue.pop_front().unwrap();
        self.queue.push_back(input);

        z_out
    }
}

struct PowerMeter {
    left_sum: f32,
    right_sum: f32,
    length: usize,
    counter: usize,
}

impl PowerMeter {
    fn new(length: usize) -> Self {
        PowerMeter {
            left_sum: 0.0,
            right_sum: 0.0,
            length,
            counter: 0,
        }
    }
    fn power(&mut self, left: f32, right: f32) -> (f32, f32) {
        self.left_sum += left * left;
        self.right_sum += right * right;

        self.counter += 1;

        if self.counter == self.length {
            self.left_sum /= self.length as f32;
            self.right_sum /= self.length as f32;

            self.counter = 1
        }

        (
            self.left_sum / self.counter as f32,
            self.right_sum / self.counter as f32
        )
    }
}

struct Waveguide {
    queue: VecDeque<f32>,
    filter: Filter,
    leak: f32,
    pmeter: PowerMeter,
}

impl Waveguide {
    fn new(wavelength: usize) -> Self {
        let queue = vec![0.0; wavelength * 2].into();

        let pmeter = PowerMeter::new(wavelength);

        Waveguide {
            queue,
            filter: Filter::new(),
            leak: 0.995,
            pmeter
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.leak = parameters.leak
    }

    fn tune(&mut self, wavelength: f32) {
        unimplemented!()
    }

    fn signal(&mut self, input: f32) -> f32 {
        let z_out = self.queue.pop_front().unwrap();
        let feedback = self.leak * self.filter.signal(z_out);

        let (p_l, p_r) = self.pmeter.power(input, feedback);
        let p_l = 0.1;
        let k = (p_l - p_r).max(0.0);

        let z_in = k * input + feedback;

        self.queue.push_back(z_in);

        z_out
    }
}

fn gate(input: f32, value: f32) -> f32 {
    if input.abs() < value {
        input
    } else {
        -input
    }
}

fn curve(input: f32) -> f32 {
    if input.abs() > 1.0 {
        println!("{}", input);
    }

    input - input.powi(3)
}

struct Square {
    period: usize,
    phase: usize,
}

impl Square {
    fn new(wavelength: usize) -> Self {
        Square {
            period: wavelength,
            phase: 0
        }
    }
    fn generate(&mut self) -> f32 {
        let signal =
            if self.phase < self.period / 2 {
                1.0
            } else {
                -1.0
            };


        self.phase += 1;
        if self.phase == self.period {
            self.phase = 0
        }

        signal
    }
}

struct Synth {
    noise: Noise,
    square: Square,
    adsr: Adsr,
    waveguide: Waveguide,
    velocity: f32,
}

impl Synth {
    fn new() -> Self {
        let noise = Noise::new();
        let adsr = Adsr::new(200.0, 0.0, 0.0, 10000.0);
        let waveguide = Waveguide::new(150);

        let square = Square::new(301);

        Synth {
            noise, adsr, waveguide, square,
            velocity: 0.25,
        }
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let noise = Noise::new();
        let adsr = Adsr::from_parameters(parameters);
        let waveguide = Waveguide::new(150);

        let square = Square::new(300);
        let velocity = parameters.velocity;

        Synth {
            noise, adsr, waveguide, square, velocity,
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.waveguide.load_parameters(parameters);

        let ms = 1e-3 * parameters.sample_rate;
        let attack = parameters.attack * ms;
        let decay = parameters.decay * ms;

        self.adsr.attack = attack;
        self.adsr.decay = decay;
    }

    fn note_on(&mut self, wavelength: f32) {
        self.adsr.note_on()
    }

    fn note_off(&mut self) {
        self.adsr.note_off()
    }

    fn fill_buf(&mut self, buf: &mut [f32]) {
        if !self.adsr.is_active() {
            for v in buf { *v = 0.0 }
            return
        }

        for v in buf {
            let excite = self.velocity * (0.9 * self.square.generate() + 0.1 * self.noise.generate()) * self.adsr.generate();
            let output = self.waveguide.signal(excite);

            *v = output
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Parameters {
    sample_rate: f32,

    leak: f32,

    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,

    velocity: f32,
}

struct Engine {
    synth: Synth,
    parameters: Parameters,
}

impl Engine {
    fn new(sample_rate: f32) -> Self {
        let parameters = Parameters {
            sample_rate,

            leak: 0.99,

            attack: 5.0,
            decay: 10.0,
            sustain: 0.0,
            release: 10000.0,

            velocity: 0.25,
        };

        let synth = Synth::from_parameters(parameters);

        Engine {
            synth, parameters
        }
    }

    fn process_osc(&mut self, msg: rosc::OscMessage) {
        use rosc::OscType as Ty;

        if msg.args.is_none() { return }

        let (addr, mut args) = (msg.addr, msg.args.unwrap());

        match &*addr {
            "/leak" => {
                if let Some(Ty::Float(leak)) = args.pop() {
                    self.parameters.leak = leak
                }
            },

            "/attack" =>
                if let Some(Ty::Float(attack)) = args.pop() {
                    self.parameters.attack = attack
                },
            "/decay" =>
                if let Some(Ty::Float(decay)) = args.pop() {
                    self.parameters.decay = decay
                },
            "/sustain" =>
                if let Some(Ty::Float(sustain)) = args.pop() {
                    self.parameters.sustain = sustain
                },
            "/release" =>
                if let Some(Ty::Float(release)) = args.pop() {
                    self.parameters.release = release
                },

            "/velocity" =>
                if let Some(Ty::Float(velocity)) = args.pop() {
                    self.parameters.velocity = velocity
                },

            _ => {
                let msg =
                    rosc::OscMessage { addr, args: Some(args) };
                if let Ok(m) = miosc::into_miosc(msg) {
                    self.process_miosc(m);
                    return
                }
            },
        }

        self.update()
    }

    fn update(&mut self) {
        self.synth.load_parameters(self.parameters)
    }

    fn process_miosc(&mut self, miosc: miosc::MioscMessage) {
        use miosc::MioscMessage::*;

        match miosc {
            NoteOn(_, _, _) =>
                self.synth.note_on(100.0),
            NoteOff(_) =>
                self.synth.note_off(),
            _ => (),
        }
    }

    fn fill_buf(&mut self, buf: &mut [f32]) {
        self.synth.fill_buf(buf)
    }
}

enum OscIoError {
    IoError(::std::io::Error),
    OscError(rosc::OscError),
}

impl From<::std::io::Error> for OscIoError {
    fn from(source: std::io::Error) -> Self {
        OscIoError::IoError(source)
    }
}

impl From<::rosc::OscError> for OscIoError {
    fn from(source: rosc::OscError) -> Self {
        OscIoError::OscError(source)
    }
}

fn read_osc(socket: &::std::net::UdpSocket) -> Result<rosc::OscMessage, OscIoError> {
    let mut buf = [0u8; 1024];

    let (n, _) = socket.recv_from(&mut buf)?;
    let pkg = rosc::decoder::decode(&buf[..n])?;

    match pkg {
        rosc::OscPacket::Message(msg) =>
            Ok(msg),
        _ => unimplemented!()
    }
}

fn main() {
    let (client, _status) = jack::Client::new("pianozen", jack::ClientOptions::NO_START_SERVER).unwrap();
    let mut out_port = client.register_port("mono", jack::AudioOut::default()).unwrap();

    let (tx, rx) = ::std::sync::mpsc::channel();

    let mut engine = Engine::new(client.sample_rate() as f32);

    let process = jack::ClosureProcessHandler::new(move |_: &jack::Client, ps: &jack::ProcessScope| -> jack::Control {
        let out = out_port.as_mut_slice(ps);

        if let Ok(msg) = rx.try_recv() {
            engine.process_osc(msg)
        }

        engine.fill_buf(out);

        jack::Control::Continue
    });

    let active_client = client.activate_async((), process).unwrap();

    let socket = ::std::net::UdpSocket::bind("127.0.0.1:3579").unwrap();

    loop {
        let msg = read_osc(&socket);

        if let Ok(msg) = msg {
            drop(tx.send(msg))
        }

        let dt = ::std::time::Duration::from_millis(8);
        ::std::thread::sleep(dt);
    }

    active_client.deactivate().unwrap();
}
