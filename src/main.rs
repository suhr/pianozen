#![allow(dead_code)]

extern crate jack;
extern crate miosc;
extern crate rosc;

use std::collections::VecDeque;

use primitives::*;

mod primitives;

const MIDI_0: f32 = 8.1757989156;

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

struct PowerMeter {
    sum: f32,
    length: usize,
    counter: usize,
}

impl PowerMeter {
    fn new(length: usize) -> Self {
        PowerMeter {
            sum: 0.0,
            length,
            counter: 0,
        }
    }
    fn feed(&mut self, input: f32) {
        self.sum += input * input;
        self.counter += 1;

        if self.counter == self.length {
            self.sum /= self.length as f32;
            self.counter = 1
        }
    }
    fn power(&self) -> f32 {
        self.sum / self.counter as f32
    }
}

struct Lagrange5Fd {
    ks: [f32; 6],
    zs: VecDeque<f32>,
}

impl Lagrange5Fd {
    fn compute_ks(delay: f32) -> [f32; 6] {
        let mut ks = [1.0; 6];

        for (n, k) in ks.iter_mut().enumerate() {
            for i in 0..=5 {
                if i == n { continue }

                *k *= (delay - i as f32) / (n as f32 - i as f32)
            }
        }

        ks
    }

    fn new(delay: f32) -> Self {
        let ks = Lagrange5Fd::compute_ks(delay);
        let zs = vec![0.0; 5].into();

        Lagrange5Fd { ks, zs }
    }

    fn tune(&mut self, delay: f32) {
        self.ks = Lagrange5Fd::compute_ks(delay)
    }

    fn signal(&mut self, input: f32) -> f32 {
        let mut s = self.ks[0] * input;

        for i in 1..6 {
            s += self.ks[i] * self.zs[i - 1]
        }

        drop(self.zs.pop_back());
        self.zs.push_front(input);

        s
    }
}

struct Waveguide {
    queue: VecDeque<f32>,
    delay: Lagrange5Fd,
    filter: Filter,
    leak: f32,
}

impl Waveguide {
    fn new(wavelength: usize) -> Self {
        let queue = vec![0.0; wavelength - 3].into();
        let delay = Lagrange5Fd::new(2.5);

        Waveguide {
            queue,
            delay,
            filter: Filter::new(),
            leak: 0.995,
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.leak = parameters.leak
    }

    fn tune(&mut self, wavelength: f32) {
        let len = wavelength - 3.0;

        self.queue.resize(len as _, 0.0);
        self.delay.tune(2.5 + len.fract());
    }

    fn tap(&self) -> f32 {
        *self.queue.back().unwrap()
    }

    fn signal(&mut self, input: f32) -> f32 {
        let z_out = self.queue.pop_front().unwrap();
        let z_out = self.delay.signal(z_out);
        let feedback = self.leak * self.filter.signal(z_out);

        let z_in = input + feedback;

        self.queue.push_back(z_in);

        z_out
    }
}

struct Exciter {
    noise: Noise,
    square: Square,
    adsr: Adsr,
    pmeter: PowerMeter,

    velocity: f32,
}

impl Exciter {
    fn new() -> Self {
        unimplemented!()
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let adsr = Adsr::from_parameters(parameters);
        let velocity = parameters.velocity.sqrt();

        let noise = Noise::new();
        let square = Square::new(150.0);
        let pmeter = PowerMeter::new(150);

        Exciter {
            noise, square, pmeter,
            adsr, velocity
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.adsr.load_parameters(parameters);
        self.velocity = parameters.velocity.sqrt();
    }

    fn tune(&mut self, wavelength: f32) {
        self.square.period = wavelength;
        self.pmeter.length = wavelength as _;
    }

    fn signal(&mut self, input: f32) -> f32 {
        let source = 0.90 * self.square.generate() + 0.10 * self.noise.generate();
        let excite = self.velocity * source * self.adsr.generate();
        self.pmeter.feed(input);

        let p_r = self.pmeter.power();
        let p_l = self.velocity.powi(2);
        let k = (p_l - p_r).max(0.0);

        k * excite
    }
}

struct Square {
    period: f32,
    phase: f32,
}

impl Square {
    fn new(wavelength: f32) -> Self {
        Square {
            period: wavelength,
            phase: 0.0
        }
    }

    fn generate(&mut self) -> f32 {
        let half = 0.5 * self.period;
        let signal =
            if self.phase < half {
                1.0 - 2.0 * self.phase / half
            } else {
                1.0 - 2.0 * (self.period - self.phase) / half
            };

        self.phase += 1.0;
        if self.phase >= self.period {
            self.phase -= self.period
        }

        signal
    }
}

struct Synth {
    exciter: Exciter,
    waveguide: Waveguide,
}

impl Synth {
    fn new() -> Self {
        let exciter = Exciter::new();
        let waveguide = Waveguide::new(150);

        Synth {
            exciter, waveguide,
        }
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let exciter = Exciter::from_parameters(parameters);
        let waveguide = Waveguide::new(150);

        Synth {
            exciter, waveguide,
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.exciter.load_parameters(parameters);
        self.waveguide.load_parameters(parameters);
    }

    fn note_on(&mut self, wavelength: f32) {
        self.exciter.tune(wavelength);
        self.waveguide.tune(wavelength);

        self.exciter.adsr.note_on()
    }

    fn note_off(&mut self) {
        self.exciter.adsr.note_off()
    }

    fn is_alive(&self) -> bool {
        self.exciter.adsr.is_active() || self.exciter.pmeter.power() > 1e-16
    }

    fn fill_buf(&mut self, buf: &mut [f32]) {
        if !self.exciter.adsr.is_active() {
            for v in buf { *v = 0.0 }
            return
        }

        for v in buf {
            let tap = self.waveguide.tap();
            let excite = self.exciter.signal(tap);
            let output = self.waveguide.signal(excite);

            *v = output
        }
    }

    fn add_to_buf(&mut self, buf: &mut [f32]) {
        if !self.is_alive() {
            return
        }

        for v in buf {
            let tap = self.waveguide.tap();
            let excite = self.exciter.signal(tap);
            let output = self.waveguide.signal(excite);

            *v += output
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    pub sample_rate: f32,

    pub leak: f32,

    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,

    pub velocity: f32,
}

struct Engine {
    synths: Vec<(i32, Synth)>,
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

        let synths = vec![];
        Engine {
            synths, parameters
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
        for (_, s) in &mut self.synths {
            s.load_parameters(self.parameters)
        }
    }

    fn process_miosc(&mut self, miosc: miosc::MioscMessage) {
        use miosc::MioscMessage::*;

        match miosc {
            NoteOn(id, pitch, _) => {
                let reference = self.parameters.sample_rate / MIDI_0;
                let ratio = (pitch / 12.0).exp2();

                if let Some((_, s)) = self.synths.iter_mut().find(|(i, _)| *i == id) {
                    s.note_on(reference / ratio);
                    return
                }

                let mut synth = Synth::from_parameters(self.parameters);
                synth.note_on(reference / ratio);
                self.synths.push((id, synth))
            },
            NoteOff(id) => {
                if let Some((_, s)) = self.synths.iter_mut().find(|(i, _)| *i == id) {
                    s.note_off()
                };
            },
            _ => (),
        }
    }

    fn fill_buf(&mut self, buf: &mut [f32]) {
        for v in buf.iter_mut() { *v = 0.0 }

        for (_, s) in &mut self.synths {
            s.add_to_buf(buf);
        }
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
