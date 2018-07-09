#![allow(dead_code)]

extern crate jack;
extern crate miosc;
extern crate rosc;

use std::collections::VecDeque;

use primitives::*;

mod primitives;

const MIDI_0: f32 = 8.1757989156;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum OscType {
    Triangle,
    Square,
    Sine,
}

impl OscType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "triangle" =>
                Some(OscType::Triangle),
            "square" =>
                Some(OscType::Square),
            "sine" =>
                Some(OscType::Sine),
            _ => None,
        }
    }
}

// A very naive oscillator
struct Oscillator {
    kind: OscType,
    period: f32,
    phase: f32,
    half_width: f32,
}

impl Oscillator {
    fn new() -> Self {
        Oscillator {
            kind: OscType::Triangle,
            period: 150.0,
            phase: 0.0,
            half_width: 1.0,
        }
    }

    fn triangle(&self, phase: f32) -> f32 {
        if phase < 0.5 {
            1.0 - 4.0 * phase
        } else {
            1.0 - 4.0 * (1.0 - phase)
        }
    }

    fn square(&self, phase: f32) -> f32 {
        if phase < 0.5 {
            1.0
        } else {
            -1.0
        }
    }

    fn sine(&self, phase: f32) -> f32 {
        (2.0 * ::std::f32::consts::PI * phase).sin()
    }

    fn generate(&mut self) -> f32 {
        let half = 0.5 * self.period;
        let w = self.half_width;

        let phase =
            if self.phase < half * w {
                self.phase / (w * self.period)
            } else {
                (0.5 * w + (self.phase - half * w) / ((2.0 - w) * self.period)).fract()
            };

        debug_assert!(phase.abs() <= 1.0);

        let signal =
            match self.kind {
                OscType::Triangle => self.triangle(phase),
                OscType::Square => self.square(phase),
                OscType::Sine => self.sine(phase),
            };

        self.phase += 1.0;
        if self.phase >= self.period {
            self.phase -= self.period
        }

        debug_assert!(signal.abs() <= 1.0);
        signal
    }
}

struct Oscs {
    osc1: Oscillator,
    osc2: Oscillator,

    tune1: f32,
    tune2: f32,
    mix: f32,
}

impl Oscs {
    fn new() -> Self {
        let osc1 = Oscillator::new();
        let osc2 = Oscillator::new();

        Oscs {
            osc1, osc2,
            tune1: 1.0,
            tune2: 1.0,
            mix: 0.0,
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.osc1.kind = parameters.osc1_type;
        self.osc2.kind = parameters.osc2_type;
        self.osc1.half_width = parameters.osc1_width;
        self.osc2.half_width = parameters.osc2_width;
        self.tune1 = (parameters.osc1_tune / 12.0).exp2();
        self.tune2 = (parameters.osc2_tune / 12.0).exp2();
        self.mix = parameters.oscs_mix
    }

    fn tune(&mut self, wavelength: f32) {
        self.osc1.period = wavelength * self.tune1;
        self.osc2.period = wavelength * self.tune2;
    }

    fn generate(&mut self) -> f32 {
        (1.0 - self.mix) * self.osc1.generate()
        + self.mix * self.osc2.generate()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Filter {
    a1: f32,
    z1: f32,
}

impl Filter {
    fn new() -> Self {
        Filter {
            a1: 0.15,
            z1: 0.0,
        }
    }

    fn delay(&self, wavelength: f32) -> f32 {
        let omega = 2.0 * ::std::f32::consts::PI / wavelength;

        let a1 = self.a1;
        let frac = a1 * omega.sin() / (1.0 - a1 * omega.cos());

        frac.atan() / omega
    }

    fn signal(&mut self, input: f32) -> f32 {
        let output = input * (1.0 - self.a1) + self.z1 * self.a1;
        self.z1 = output;

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

struct Damper {
    is_off: bool,
    damping: f32,
    counter: f32,
}

impl Damper {
    fn new() -> Self {
        Damper {
            is_off: true,
            damping: 0.05,
            counter: 0.0,
        }
    }

    fn activate(&mut self) {
        self.is_off = false
    }

    fn deactivate(&mut self) {
        self.is_off = true;
        self.counter = 0.0
    }

    fn signal(&mut self, input: f32) -> f32 {
        if self.is_off {
            input
        } else {
            if self.counter >= 440.0 {
                (1.0 - self.damping) * input
            } else {
                let k = self.damping * self.counter / 440.0;

                self.counter += 1.0;
                (1.0 - k) * input
            }
        }
    }
}

struct Waveguide {
    queue: VecDeque<f32>,
    delay: Lagrange5Fd,
    filter: Filter,
    leak: f32,
    damper: Damper,
}

impl Waveguide {
    fn new(wavelength: usize) -> Self {
        let queue = vec![0.0; wavelength - 3].into();
        let delay = Lagrange5Fd::new(2.5);

        Waveguide {
            queue,
            delay,
            filter: Filter::new(),
            leak: 0.994,
            damper: Damper::new(),
        }
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let mut waveguide = Waveguide::new(150);
        waveguide.load_parameters(parameters);

        waveguide
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.leak = parameters.leak;
        self.damper.damping = parameters.damper;
        self.filter.a1 = parameters.cutoff
    }

    fn note_on(&mut self) {
        self.damper.deactivate()
    }

    fn note_off(&mut self) {
        self.damper.activate()
    }

    fn tune(&mut self, wavelength: f32) {
        let fd = 2.5 + self.filter.delay(wavelength);
        let len = wavelength - fd;

        self.queue.resize(len as _, 0.0);
        self.delay.tune(2.5 + len.fract());
    }

    fn tap(&self) -> f32 {
        *self.queue.back().unwrap()
    }

    fn signal(&mut self, input: f32) -> f32 {
        let leak = self.damper.signal(self.leak);

        let z_out = self.queue.pop_front().unwrap();
        let z_out = self.delay.signal(z_out);
        let feedback = leak * self.filter.signal(z_out);

        let z_in = input + feedback;

        self.queue.push_back(z_in);

        z_out
    }
}

struct Exciter {
    noise: Noise,
    oscs: Oscs,
    adsr: Adsr,
    pmeter: PowerMeter,
    //filter: Filter,

    velocity: f32,
    noise_ratio: f32,
}

impl Exciter {
    fn new() -> Self {
        unimplemented!()
    }

    fn from_parameters(parameters: Parameters) -> Self {
        let adsr = Adsr::from_parameters(parameters);
        let velocity = parameters.velocity.powf(1.5);
        let noise_ratio = parameters.noise;

        let noise = Noise::new();
        let mut oscs = Oscs::new();
        let pmeter = PowerMeter::new(1100);

        oscs.load_parameters(parameters);

        Exciter {
            noise, oscs, pmeter,
            adsr, velocity, noise_ratio
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.adsr.load_parameters(parameters);
        self.oscs.load_parameters(parameters);
        self.velocity = parameters.velocity.powf(1.5);
        self.noise_ratio = parameters.noise;
    }

    fn tune(&mut self, wavelength: f32) {
        self.oscs.tune(wavelength)
    }

    fn signal(&mut self, input: f32) -> f32 {
        let nsr = self.noise_ratio;
        let source = (1.0 - nsr) * self.oscs.generate() + nsr * self.noise.generate();
        let excite = source * self.adsr.generate();
        self.pmeter.feed(input);

        let p_r = self.pmeter.power().sqrt();
        let p_l = self.velocity;
        let k = (p_l - p_r).max(0.0);

        k * excite
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
        let waveguide = Waveguide::from_parameters(parameters);

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

        self.exciter.adsr.note_on();
        self.waveguide.note_on();
    }

    fn note_off(&mut self) {
        self.exciter.adsr.note_off();
        self.waveguide.note_off();
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

    pub osc1_type: OscType,
    pub osc2_type: OscType,
    pub osc1_width: f32,
    pub osc2_width: f32,
    pub osc1_tune: f32,
    pub osc2_tune: f32,
    pub oscs_mix: f32,

    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,

    pub velocity: f32,
    pub noise: f32,

    pub leak: f32,
    pub damper: f32,
    pub cutoff: f32,
}

struct Engine {
    synths: Vec<(i32, Synth)>,
    parameters: Parameters,
}

impl Engine {
    fn new(sample_rate: f32) -> Self {
        let parameters = Parameters {
            sample_rate,

            osc1_type: OscType::Triangle,
            osc2_type: OscType::Triangle,
            osc1_width: 1.0,
            osc2_width: 1.0,
            osc1_tune: 1.0,
            osc2_tune: 1.0,
            oscs_mix: 0.0,

            attack: 5.0,
            decay: 10.0,
            sustain: 0.0,
            release: 10000.0,

            velocity: 0.25,
            noise: 0.15,

            leak: 0.994,
            damper: 0.04,
            cutoff: 0.15,
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
            "/osc1_type" => {
                if let Some(Ty::String(s)) = args.pop() {
                    if let Some(ty) = OscType::from_str(&s) {
                        self.parameters.osc1_type = ty
                    }
                }
            },
            "/osc2_type" => {
                if let Some(Ty::String(s)) = args.pop() {
                    if let Some(ty) = OscType::from_str(&s) {
                        self.parameters.osc2_type = ty
                    }
                }
            },
            "/osc1_width" =>
                if let Some(Ty::Float(osc1_width)) = args.pop() {
                    self.parameters.osc1_width = osc1_width
                },
            "/osc2_width" =>
                if let Some(Ty::Float(osc2_width)) = args.pop() {
                    self.parameters.osc2_width = osc2_width
                },
            "/osc1_tune" =>
                if let Some(Ty::Float(osc1_tune)) = args.pop() {
                    self.parameters.osc1_tune = osc1_tune
                },
            "/osc2_tune" =>
                if let Some(Ty::Float(osc2_tune)) = args.pop() {
                    self.parameters.osc2_tune = osc2_tune
                },
            "/oscs_mix" =>
                if let Some(Ty::Float(oscs_mix)) = args.pop() {
                    self.parameters.oscs_mix = oscs_mix
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
            "/noise" =>
                if let Some(Ty::Float(noise)) = args.pop() {
                    self.parameters.noise = noise
                },

            "/leak" => {
                if let Some(Ty::Float(leak)) = args.pop() {
                    self.parameters.leak = leak
                }
            },
            "/damper" =>
                if let Some(Ty::Float(damper)) = args.pop() {
                    self.parameters.damper = damper
                },
            "/cutoff" =>
                if let Some(Ty::Float(cutoff)) = args.pop() {
                    self.parameters.cutoff = cutoff
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
    let (client, _status) = jack::Client::new("xephys", jack::ClientOptions::NO_START_SERVER).unwrap();
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
