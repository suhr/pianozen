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

struct Waveguide {
    queue: VecDeque<f32>,
    filter: Filter,
    leak: f32,
}

impl Waveguide {
    fn new(wavelength: usize) -> Self {
        let queue = vec![0.0; wavelength * 2].into();

        Waveguide {
            queue,
            filter: Filter::new(),
            leak: 0.995,
        }
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.leak = parameters.leak
    }

    fn signal(&mut self, input: f32) -> f32 {
        let z_out = self.queue.pop_front().unwrap();
        let z_in = self.leak * self.filter.signal(z_out) + input;

        self.queue.push_back(z_in);

        z_out
    }
}

struct Synth {
    noise: Noise,
    adsr: Adsr,
    waveguide: Waveguide,
}

impl Synth {
    fn new() -> Self {
        let noise = Noise::new();
        let adsr = Adsr::new(200.0, 0.0, 0.0, 10000.0);
        let waveguide = Waveguide::new(100);

        Synth {
            noise, adsr, waveguide
        }
    }

    fn from_parameters(parameters: Parameters) -> Self {
        unimplemented!()
    }

    fn load_parameters(&mut self, parameters: Parameters) {
        self.waveguide.load_parameters(parameters)
    }

    fn note_on(&mut self) {
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
            let excite = 0.5 * self.noise.generate() * self.adsr.generate();
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
}

struct Engine {
    synth: Synth,
    parameters: Parameters,
}

impl Engine {
    fn new(sample_rate: f32) -> Self {
        let parameters = Parameters {
            sample_rate,

            leak: 0.995,

            attack: 200.0,
            decay: 0.0,
            sustain: 0.0,
            release: 0.0,
        };
        let synth = Synth::new();

        Engine {
            synth, parameters
        }
    }

    fn process_osc(&mut self, msg: rosc::OscMessage) {
        use rosc::OscType as Ty;

        if msg.args.is_none() { return }

        match &*msg.addr {
            "/leak" => {
                let mut args = msg.args.unwrap();

                if let Some(Ty::Float(leak)) = args.pop() {
                    self.parameters.leak = leak
                }
            },
            _ => {
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
                self.synth.note_on(),
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
