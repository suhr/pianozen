# Xephys

Xephys is a microtonal [digital waveguide](https://en.wikipedia.org/wiki/Digital_waveguide_synthesis) synthesier.

## Installation

Install [Rust](https://www.rust-lang.org/) and [JACK](http://jackaudio.org/) and then run

```
$ cargo install --git https://github.com/suhr/xephys.git
```

## How to use it

Xephys is a standalone synth that uses JACK for audio output and OSC for control. It has no support for MIDI. Instead it uses the Miosc protocol that works over OSC.

To convert MIDI messages into Miosc messages, you can use a little utily called “mimi”. You can install it with:

```
$ cargo install --git https://github.com/diaschisma/mimi.git
```

By default, it converts MIDI messages to pitches of the [31 tone equal temperament](https://en.wikipedia.org/wiki/31_equal_temperament). You can choose any equal temperament you want by running mimi with corresponding `--edo` flag. For example, to use the standard 12 tone equal temperament, run mimi as `mimi --edo 12`.

Synth parameters are changed by sending the corresponding OSC messages. The most convinient way to do it, is to use [Open Stage Control](https://osc.ammd.net/). This programm provides you a graphical interface to the OSC parameters. The repository contains file named `xephys.json`, which you can open in Open Stage Control to control the synth.

The default UDP port for OSC messages is `3579`.
