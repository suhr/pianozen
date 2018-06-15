use super::Parameters;

pub struct Noise {
    rand: u32,
}

impl Noise {
    pub fn new() -> Self {
        // XKCD random number
        let rand = 4;

        Noise {
            rand,
        }
    }

    pub fn generate(&mut self) -> f32 {
        // BSD variant of the linear congruential method
        let max = 2_147_483_647.0;

        self.rand = self.rand.wrapping_mul(1_103_515_245).wrapping_add(12345);
        self.rand %= 1 << 31;

        1.0 - 2.0 * self.rand as f32 / max
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdsrState {
    Idle,
    Press(u32),
    Sustain,
    Release(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adsr {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    state: AdsrState,
}

impl Adsr {
    pub fn new(attack: f32, decay: f32, sustain: f32, release: f32) -> Self {
        Adsr {
            attack, decay, sustain, release,
            state: AdsrState::Idle,
        }
    }

    pub fn from_parameters(parameters: Parameters) -> Self {
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

    pub fn load_parameters(&mut self, parameters: Parameters) {
        let ms = 1e-3 * parameters.sample_rate;
        let attack = parameters.attack * ms;
        let decay = parameters.decay * ms;
        let release = parameters.release * ms;

        self.attack = attack;
        self.decay = decay;
        self.sustain = parameters.sustain;
        self.release = release;
    }

    pub fn note_on(&mut self) {
        self.state = AdsrState::Press(0)
    }

    pub fn note_off(&mut self) {
        self.state = AdsrState::Release(0)
    }

    pub fn is_active(&self) -> bool {
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

    pub fn generate(&mut self) -> f32 {
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

struct Delay {
    queue: ::std::collections::VecDeque<f32>,
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